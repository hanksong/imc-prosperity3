from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import numpy as np
import json
class Trader:
    def __init__(self):
        self.product = "SQUID_INK"  # 交易产品
        self.position_limit = 50     # 仓位限制
        
        # 保存价格历史
        self.price_history = []
        
        # 均值回归策略参数
        self.ma_window = 50       # 移动平均窗口
        self.std_window = 50      # 标准差计算窗口
        self.z_threshold = 1.5    # z分数阈值
        self.min_history = 51     # 最小历史数据要求
        
        # 跟踪绩效
        self.pnl_history = []
        self.last_price = None
        
        # 止损参数
        self.max_loss_pct = 0.001  # 相对于价格的最大损失比例
        self.stop_loss_active = False
        self.entry_price = None
    
    def calculate_mid_price(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]) -> float:
        """计算中间价格"""
        if not buy_orders or not sell_orders:
            return None
            
        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def calculate_signal(self):
        """均值回归策略信号"""
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
        
        # 5. 生成交易信号: 
        # 正z分数表示价格高于均值，应该做空(等待回归)
        # 负z分数表示价格低于均值，应该做多(等待回归)
        signal = -z_score  # 反向z分数，价格偏高时做空，价格偏低时做多
        
        # 检测价格波动率变化
        if len(self.price_history) >= self.std_window * 2:
            recent_std = np.std(self.price_history[-self.std_window:])
            prev_std = np.std(self.price_history[-self.std_window*2:-self.std_window])
            
            # 如果波动率突然增加，降低信号强度(市场可能变得不稳定)
            if recent_std > prev_std * 1.5:
                signal *= 0.5
        
        return signal
    
    def check_stop_loss(self, current_price, current_position):
        """检查止损条件"""
        if self.entry_price is None or current_position == 0:
            self.stop_loss_active = False
            return False
            
        # 计算当前持仓的亏损比例
        if current_position > 0:  # 多头仓位
            loss_pct = (self.entry_price - current_price) / self.entry_price
            if loss_pct > self.max_loss_pct:
                self.stop_loss_active = True
                return True
        elif current_position < 0:  # 空头仓位
            loss_pct = (current_price - self.entry_price) / self.entry_price
            if loss_pct > self.max_loss_pct:
                self.stop_loss_active = True
                return True
                
        return False
    
    def determine_position_size(self, signal: float, current_position: int) -> int:
        """根据信号强度确定目标仓位大小"""
        # 只有当信号超过阈值时才交易
        if abs(signal) < self.z_threshold:
            return current_position  # 维持当前仓位
            
        # 计算目标仓位
        # 信号越强，仓位越大，但上限是仓位限制
        position_scale = min(abs(signal) / (self.z_threshold * 2), 1.0)
        target_position = int(np.sign(signal) * position_scale * self.position_limit)
        
        # 如果止损触发，则平仓
        if self.stop_loss_active:
            return 0
            
        return target_position
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """执行交易策略"""
        result = {}
        
        # 如果没有目标产品的订单深度，则返回空结果
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
        
        # 检查止损条件
        if self.check_stop_loss(mid_price, current_position):
            target_position = 0  # 触发止损，平仓
        else:
            # 计算均值回归信号
            signal = self.calculate_signal()
            
            # 确定目标仓位
            target_position = self.determine_position_size(signal, current_position)
        
        # 计算需要调整的仓位
        position_change = target_position - current_position
        
        # 自适应参数调整
        # 根据最近的盈亏表现动态调整策略参数
        if len(self.pnl_history) > 20:
            recent_pnl = sum(self.pnl_history[-20:])
            
            # 如果策略表现不佳，可能需要调整参数
            if recent_pnl < -50:
                # 增加z分数阈值，减少交易频率
                self.z_threshold = min(self.z_threshold * 1.2, 3.0)
                
                # 减小目标仓位规模
                position_change = int(position_change * 0.7)
            
            # 如果策略表现良好，可以更积极
            elif recent_pnl > 50:
                # 适度减小z分数阈值
                self.z_threshold = max(self.z_threshold * 0.9, 1.0)
        
        orders = []
        
        # 在首次开仓或平仓后重新开仓时记录入场价格
        if current_position == 0 and position_change != 0:
            self.entry_price = mid_price
            self.stop_loss_active = False
        
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
            "signal": "平仓" if self.stop_loss_active else self.calculate_signal(),
            "target_position": target_position,
            "price": mid_price,
            "ma": np.mean(self.price_history[-self.ma_window:]),
            "std": np.std(self.price_history[-self.std_window:]),
            "z_threshold": self.z_threshold,
            "stop_loss": self.stop_loss_active
        }
        
        # 将交易员数据转换为字符串
        trader_data_str = json.dumps(trader_data)
        
        # 打印调试信息
        cum_pnl = sum(self.pnl_history) if self.pnl_history else 0
        print(f"时间戳: {state.timestamp}, 价格: {mid_price:.2f}, 信号: {trader_data['signal']}, "
              f"目标仓位: {target_position}, 当前仓位: {current_position}, 调整: {position_change}, 累计PnL: {cum_pnl:.2f}")
        
        # 返回订单、转换和交易员数据
        conversions = 0  # 不执行转换
        return result, conversions, trader_data_str 