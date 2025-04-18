from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np
import json

class Trader:
    def __init__(self):
        self.product = "SQUID_INK"
        self.position_limit = 50  # 仓位限制
        
        # 保存价格历史
        self.price_history = []
        
        # 均值回归策略参数
        self.ma_window = 50     # 移动平均窗口
        self.std_window = 50    # 标准差计算窗口
        self.dev_threshold = 1  # 偏离阈值(标准差倍数)
        self.min_history = 51   # 最小历史数据长度要求
        
        # 跟踪PnL表现
        self.pnl_history = []
        self.last_price = None
    
    def calculate_mid_price(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]) -> float:
        """计算中间价格"""
        if not buy_orders or not sell_orders:
            return None
            
        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def calculate_signal(self):
        """简单均值回归策略信号"""
        if len(self.price_history) < self.min_history:
            return 0
        
        # 1. 计算移动平均
        ma = np.mean(self.price_history[-self.ma_window:])
        
        # 2. 计算标准差
        std = np.std(self.price_history[-self.std_window:])
        
        # 3. 当前价格
        current_price = self.price_history[-1]
        
        # 4. 计算z分数(偏离均值的标准差倍数)
        z_score = (current_price - ma) / std if std > 0 else 0
        
        # 5. 生成交易信号: 正z分数表示价格高于均值，应该做空
        # 负z分数表示价格低于均值，应该做多
        # 反向均值回归: 价格突破均值时，跟随趋势而非逆势交易
        # 检查最近的价格趋势
        short_trend = 0
        if len(self.price_history) >= 5:
            # 计算短期价格趋势 (最近5个周期)
            recent_prices = self.price_history[-5:]
            price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
            short_trend = np.mean(price_changes)
        
        # 结合趋势和均值回归信号
        # 如果趋势很强，就跟随趋势；如果趋势不明确，就做均值回归
        if abs(short_trend) > std * 0.05:  # 趋势明显
            signal = np.sign(short_trend) * min(abs(z_score), 3.0)  # 跟随趋势，但限制信号强度
        else:
            signal = -z_score  # 均值回归
        
        return signal
    
    def determine_position_size(self, signal: float) -> int:
        """根据信号强度确定目标仓位大小"""
        # 只有当信号超过阈值时才交易
        if abs(signal) < self.dev_threshold:
            return 0
            
        # 计算目标仓位
        # 信号越强，仓位越大
        position_scale = min(abs(signal) / (self.dev_threshold * 2), 1.0)
        target_position = int(np.sign(signal) * position_scale * self.position_limit)
        
        return target_position
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """执行交易策略"""
        result = {}
        
        # 如果没有SQUID_INK的订单深度，则返回空结果
        if self.product not in state.order_depths:
            return {self.product: []}, 0, ""
            
        order_depth = state.order_depths[self.product]
        current_position = state.position.get(self.product, 0)
        
        # 提取当前市场数据
        mid_price = self.calculate_mid_price(order_depth.buy_orders, order_depth.sell_orders)
        if mid_price is None:
            return {self.product: []}, 0, ""
        
        # 计算PnL (如果有上一个价格)
        if self.last_price is not None and current_position != 0:
            pnl_change = current_position * (mid_price - self.last_price)
            self.pnl_history.append(pnl_change)
        
        # 更新最后价格
        self.last_price = mid_price
        
        # 更新历史数据
        self.price_history.append(mid_price)
        
        # 如果历史数据不足，则不交易
        if len(self.price_history) < self.min_history:
            return {self.product: []}, 0, ""
        
        # 计算均值回归信号
        signal = self.calculate_signal()
        
        # 确定目标仓位
        target_position = self.determine_position_size(signal)
        
        # 计算需要调整的仓位
        position_change = target_position - current_position
        
        # 计算当前盈亏趋势
        if len(self.pnl_history) > 10:
            recent_pnl = self.pnl_history[-10:]
            pnl_trend = sum(recent_pnl)
            
            # 如果连续亏损，可能是策略方向有问题，调整信号方向
            if pnl_trend < -50 and position_change != 0:
                # 反转信号方向
                position_change = -position_change
                print(f"检测到持续亏损，反转交易方向! 原信号: {signal}, 原目标仓位: {target_position}, 新仓位变动: {position_change}")
        
        orders = []
        
        # 执行买入订单
        if position_change > 0:
            # 获取最优卖价(best ask)
            if order_depth.sell_orders:
                best_ask_price = min(order_depth.sell_orders.keys())
                best_ask_volume = abs(order_depth.sell_orders[best_ask_price])
                
                # 确定能以best ask价格买入的数量
                buy_volume = min(position_change, best_ask_volume)
                
                if buy_volume > 0:
                    # 以最优卖价(best ask)买入
                    orders.append(Order(self.product, int(best_ask_price), int(buy_volume)))
                    
                    # 如果还有更多需要买入的，可以使用次优价格
                    remaining_to_buy = position_change - buy_volume
                    if remaining_to_buy > 0:
                        # 尝试获取次优卖价
                        sell_prices = sorted(order_depth.sell_orders.keys())
                        if len(sell_prices) > 1:
                            next_ask_price = sell_prices[1]
                            next_ask_volume = abs(order_depth.sell_orders[next_ask_price])
                            buy_volume = min(remaining_to_buy, next_ask_volume)
                            if buy_volume > 0:
                                orders.append(Order(self.product, int(next_ask_price), int(buy_volume)))

        # 执行卖出订单
        elif position_change < 0:
            # 获取最优买价(best bid)
            if order_depth.buy_orders:
                best_bid_price = max(order_depth.buy_orders.keys())
                best_bid_volume = abs(order_depth.buy_orders[best_bid_price])
                
                # 确定能以best bid价格卖出的数量
                sell_volume = min(abs(position_change), best_bid_volume)
                
                if sell_volume > 0:
                    # 以最优买价(best bid)卖出
                    orders.append(Order(self.product, int(best_bid_price), -int(sell_volume)))
                    
                    # 如果还有更多需要卖出的，可以使用次优价格
                    remaining_to_sell = abs(position_change) - sell_volume
                    if remaining_to_sell > 0:
                        # 尝试获取次优买价
                        buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
                        if len(buy_prices) > 1:
                            next_bid_price = buy_prices[1]
                            next_bid_volume = abs(order_depth.buy_orders[next_bid_price])
                            sell_volume = min(remaining_to_sell, next_bid_volume)
                            if sell_volume > 0:
                                orders.append(Order(self.product, int(next_bid_price), -int(sell_volume)))
        
        result[self.product] = orders
        
        # 创建交易员数据
        trader_data = {
            "position": current_position,
            "signal": signal,
            "target_position": target_position,
            "price": mid_price,
            "ma": np.mean(self.price_history[-self.ma_window:]),
            "std": np.std(self.price_history[-self.std_window:])
        }
        
        # 将交易员数据转换为字符串
        trader_data_str = json.dumps(trader_data)
        
        # 打印调试信息
        cum_pnl = sum(self.pnl_history) if self.pnl_history else 0
        print(f"时间戳: {state.timestamp}, 价格: {mid_price:.2f}, 信号: {signal:.2f}, "
              f"目标仓位: {target_position}, 当前仓位: {current_position}, 调整: {position_change}, 累计PnL: {cum_pnl:.2f}")
        
        # 返回订单、转换和交易员数据
        conversions = 0  # 不执行转换
        return result, conversions, trader_data_str