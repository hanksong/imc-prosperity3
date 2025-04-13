from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import numpy as np
import statistics
import json

class Trader:
    def __init__(self):
        self.product = "JAMS"
        self.position_limit = 350  # 仓位限制
        
        # 保存价格历史和订单深度历史用于计算因子
        self.price_history = []
        self.bid_volume_history = []
        self.ask_volume_history = []


        # magic number
        self.mean_jams = -7.578047e-07
        self.std_jams = 7.618271e-05
        
    def calculate_mid_price(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]) -> float:
        """计算中间价格, Volume Weighted 1 Mid Price"""
        if not buy_orders or not sell_orders:
            return None
            
        best_bid = max(buy_orders.keys())
        best_ask = min(sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def calculate_momentum_1(self) -> float:
        if len(self.price_history) < self.ma_short + 1:
            return 0
    
        current_price = self.price_history[-1]
        prev_price = self.price_history[-2]
        
        return -(current_price - prev_price) / prev_price if prev_price != 0 else 0
    
    def determine_position_size(self, signal: float, current_position: int) -> int:
        """根据信号强度确定目标仓位大小"""
        # 信号值在-1到1之间，转换为目标仓位
        target_percent = 1
        # 转换为Python的int类型，避免numpy.int64类型
        target_position = int(target_percent * self.position_limit)
        
        # 限制在仓位范围内，并确保返回标准Python整数
        return int(np.clip(target_position, -self.position_limit, self.position_limit))
    
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
        
        # 更新历史数据
        self.price_history.append(mid_price)
        
        total_bid_volume = sum(abs(amount) for amount in order_depth.buy_orders.values())
        total_ask_volume = sum(abs(amount) for amount in order_depth.sell_orders.values())
        self.bid_volume_history.append(total_bid_volume)
        self.ask_volume_history.append(total_ask_volume)
        
        # 如果历史数据不足，则不交易
        if len(self.price_history) < self.ma_long:
            return {self.product: []}, 0, ""
        
        # 计算综合信号
        signal = self.calculate_momentum_1()
        
        # 确定目标仓位
        target_position = self.determine_position_size(signal, current_position)
        
        # 计算需要调整的仓位
        position_change = target_position - current_position
        
        orders = []
        
        # 执行买入订单
        if position_change > 0:
            # 按价格排序的卖单
            sorted_sells = sorted(order_depth.sell_orders.items())
            remaining_to_buy = position_change
            
            for price, volume in sorted_sells:
                if remaining_to_buy <= 0:
                    break
                    
                buy_volume = min(remaining_to_buy, abs(volume))
                if buy_volume > 0:
                    # 确保价格和数量都是原生Python整数
                    orders.append(Order(self.product, int(price), int(buy_volume)))
                    remaining_to_buy -= buy_volume
        
        # 执行卖出订单
        elif position_change < 0:
            # 按价格降序排序的买单
            sorted_buys = sorted(order_depth.buy_orders.items(), reverse=True)
            remaining_to_sell = abs(position_change)
            
            for price, volume in sorted_buys:
                if remaining_to_sell <= 0:
                    break
                    
                sell_volume = min(remaining_to_sell, abs(volume))
                if sell_volume > 0:
                    # 确保价格和数量都是原生Python整数
                    orders.append(Order(self.product, int(price), -int(sell_volume)))
                    remaining_to_sell -= sell_volume
        
        result[self.product] = orders
        
        # 创建交易员数据
        trader_data = {
            "position": current_position,
            "signal": signal,
            "target_position": target_position,
            "price": mid_price
        }
        
        # 将交易员数据转换为字符串
        trader_data_str = json.dumps(trader_data)
        
        # 返回订单、转换和交易员数据
        conversions = 1  # 不执行转换
        return result, conversions, trader_data_str