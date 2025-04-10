#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datamodel import Order, ProsperityEncoder, Symbol, Trade, TradingState, OrderDepth
from typing import Dict, List
import statistics
import json
import numpy as np

class Trader:
    def __init__(self):
        # 初始化价格历史记录
        self.kelp_prices = []
        self.ink_prices = []
        self.resin_prices = []
        
        # 产品名称
        self.products = {
            "kelp": "KELP",
            "ink": "SQUID_INK",
            "resin": "RAINFOREST_RESIN"
        }
        
        # 每个产品的持仓限制
        self.position_limit = 20
        
        # 相关性策略参数
        self.correlation = -0.314151  # 已知的负相关性
        self.correlation_strength = 1.5  # 放大相关性的影响
        self.window_size = 10  # 计算相关性的窗口大小
        self.std_dev_threshold = 0.8  # 标准差阈值
        
        # 交易量参数
        self.base_quantity = 3  # 基础交易量
        self.max_quantity = 5  # 最大交易量
        
        # 历史数据
        self.kelp_ink_ratio = []  # KELP与SQUID_INK的价格比率
        self.kelp_resin_ratio = []  # KELP与RAINFOREST_RESIN的价格比率
        self.ink_resin_ratio = []  # SQUID_INK与RAINFOREST_RESIN的价格比率
        
        # 交易状态
        self.trading_cycle = 0
        self.last_kelp_signal = 0  # 1表示看涨，-1表示看跌，0表示中性
        self.last_ink_signal = 0
        self.last_resin_signal = 0
        
    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """计算中间价格"""
        if not order_depth:
            return None
            
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif len(order_depth.buy_orders) > 0:
            return max(order_depth.buy_orders.keys())
        elif len(order_depth.sell_orders) > 0:
            return min(order_depth.sell_orders.keys())
        else:
            return None
    
    def update_ratios(self, kelp_price, ink_price, resin_price):
        """更新价格比率历史"""
        if kelp_price and ink_price:
            self.kelp_ink_ratio.append(kelp_price / ink_price)
            
        if kelp_price and resin_price:
            self.kelp_resin_ratio.append(kelp_price / resin_price)
            
        if ink_price and resin_price:
            self.ink_resin_ratio.append(ink_price / resin_price)
            
        # 保持窗口大小
        if len(self.kelp_ink_ratio) > self.window_size:
            self.kelp_ink_ratio = self.kelp_ink_ratio[-self.window_size:]
        if len(self.kelp_resin_ratio) > self.window_size:
            self.kelp_resin_ratio = self.kelp_resin_ratio[-self.window_size:]
        if len(self.ink_resin_ratio) > self.window_size:
            self.ink_resin_ratio = self.ink_resin_ratio[-self.window_size:]
    
    def calculate_z_score(self, current_value, history):
        """计算当前值的z-score"""
        if len(history) < 5:  # 至少需要5个历史数据点
            return 0
            
        mean_value = np.mean(history)
        std_value = np.std(history)
        
        if std_value == 0:
            return 0
            
        return (current_value - mean_value) / std_value
    
    def generate_signals(self, kelp_price, ink_price, resin_price):
        """根据价格之间的负相关性生成交易信号"""
        kelp_signal = 0
        ink_signal = 0
        resin_signal = 0
        
        # 计算当前比率
        if kelp_price and ink_price:
            current_kelp_ink = kelp_price / ink_price
            kelp_ink_z = self.calculate_z_score(current_kelp_ink, self.kelp_ink_ratio)
            
            # 利用负相关性：当KELP/INK比率高时，预期KELP将下跌，INK将上涨
            if abs(kelp_ink_z) > self.std_dev_threshold:
                # 比率异常高，预期回归：做空KELP，做多INK
                if kelp_ink_z > 0:
                    kelp_signal = -1
                    ink_signal = 1
                # 比率异常低，预期回归：做多KELP，做空INK
                elif kelp_ink_z < 0:
                    kelp_signal = 1
                    ink_signal = -1
        
        # 计算KELP与RESIN的关系
        if kelp_price and resin_price:
            current_kelp_resin = kelp_price / resin_price
            kelp_resin_z = self.calculate_z_score(current_kelp_resin, self.kelp_resin_ratio)
            
            if abs(kelp_resin_z) > self.std_dev_threshold:
                # 调整KELP信号，权重0.5
                kelp_adj = -1 if kelp_resin_z > 0 else 1
                kelp_signal = (kelp_signal + kelp_adj * 0.5) if kelp_signal != 0 else kelp_adj * 0.5
                
                # 调整RESIN信号，权重1.0
                resin_signal = 1 if kelp_resin_z > 0 else -1
        
        # 计算INK与RESIN的关系
        if ink_price and resin_price:
            current_ink_resin = ink_price / resin_price
            ink_resin_z = self.calculate_z_score(current_ink_resin, self.ink_resin_ratio)
            
            if abs(ink_resin_z) > self.std_dev_threshold:
                # 调整INK信号，权重0.5
                ink_adj = -1 if ink_resin_z > 0 else 1
                ink_signal = (ink_signal + ink_adj * 0.5) if ink_signal != 0 else ink_adj * 0.5
                
                # 强化RESIN信号，权重0.5
                resin_adj = 1 if ink_resin_z > 0 else -1
                resin_signal = (resin_signal + resin_adj * 0.5) if resin_signal != 0 else resin_adj * 0.5
        
        # 信号强度限制在[-1, 1]范围内
        kelp_signal = max(-1, min(1, kelp_signal))
        ink_signal = max(-1, min(1, ink_signal))
        resin_signal = max(-1, min(1, resin_signal))
        
        # 平滑信号，与上一个信号融合（70%当前信号，30%上一个信号）
        kelp_signal = 0.7 * kelp_signal + 0.3 * self.last_kelp_signal
        ink_signal = 0.7 * ink_signal + 0.3 * self.last_ink_signal
        resin_signal = 0.7 * resin_signal + 0.3 * self.last_resin_signal
        
        # 更新最后的信号
        self.last_kelp_signal = kelp_signal
        self.last_ink_signal = ink_signal
        self.last_resin_signal = resin_signal
        
        return kelp_signal, ink_signal, resin_signal
    
    def calculate_trade_quantity(self, signal_strength, position, is_buy):
        """根据信号强度和当前持仓计算交易数量"""
        # 信号强度转换为[1, base_quantity]范围
        quantity = abs(signal_strength) * (self.base_quantity - 1) + 1
        
        # 考虑持仓限制
        if is_buy:
            available = self.position_limit - position
        else:
            available = self.position_limit + position
        
        # 限制交易量
        return min(int(quantity), available, self.max_quantity)
    
    def create_orders_based_on_signal(self, product, order_depth, position, signal):
        """根据信号创建订单"""
        if not order_depth or signal == 0:
            return []
            
        orders = []
        
        # 信号为正，买入
        if signal > 0 and position < self.position_limit:
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = abs(order_depth.sell_orders[best_ask])
                
                # 计算交易量
                buy_qty = self.calculate_trade_quantity(signal, position, True)
                buy_qty = min(buy_qty, best_ask_volume)
                
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
        
        # 信号为负，卖出
        elif signal < 0 and position > -self.position_limit:
            if len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                
                # 计算交易量
                sell_qty = self.calculate_trade_quantity(abs(signal), position, False)
                sell_qty = min(sell_qty, best_bid_volume)
                
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
        
        return orders
    
    def run(self, state: TradingState):
        """
        基于负相关性的交易策略
        """
        # 自增交易周期
        self.trading_cycle += 1
        
        # 初始化结果字典
        result = {}
        
        # 获取三个产品的订单深度
        kelp_depth = state.order_depths.get(self.products["kelp"], OrderDepth())
        ink_depth = state.order_depths.get(self.products["ink"], OrderDepth())
        resin_depth = state.order_depths.get(self.products["resin"], OrderDepth())
        
        # 计算中间价格
        kelp_mid_price = self.calculate_mid_price(kelp_depth)
        ink_mid_price = self.calculate_mid_price(ink_depth)
        resin_mid_price = self.calculate_mid_price(resin_depth)
        
        # 更新历史价格
        if kelp_mid_price:
            self.kelp_prices.append(kelp_mid_price)
        if ink_mid_price:
            self.ink_prices.append(ink_mid_price)
        if resin_mid_price:
            self.resin_prices.append(resin_mid_price)
        
        # 更新价格比率
        self.update_ratios(kelp_mid_price, ink_mid_price, resin_mid_price)
        
        # 获取当前持仓
        kelp_position = state.position.get(self.products["kelp"], 0)
        ink_position = state.position.get(self.products["ink"], 0)
        resin_position = state.position.get(self.products["resin"], 0)
        
        # 检查是否需要平仓（接近持仓限制）
        kelp_risk = abs(kelp_position) > self.position_limit * 0.9
        ink_risk = abs(ink_position) > self.position_limit * 0.9
        resin_risk = abs(resin_position) > self.position_limit * 0.9
        
        # 生成信号
        kelp_signal, ink_signal, resin_signal = self.generate_signals(
            kelp_mid_price, ink_mid_price, resin_mid_price
        )
        
        # 如果有风险持仓，优先平仓
        if kelp_risk:
            # 如果持仓过多，强制卖出
            if kelp_position > 0:
                kelp_signal = -1.0
            # 如果持仓过少（负值过大），强制买入 
            else:
                kelp_signal = 1.0
        
        if ink_risk:
            if ink_position > 0:
                ink_signal = -1.0
            else:
                ink_signal = 1.0
                
        if resin_risk:
            if resin_position > 0:
                resin_signal = -1.0
            else:
                resin_signal = 1.0
        
        # 创建订单
        kelp_orders = self.create_orders_based_on_signal(
            self.products["kelp"], kelp_depth, kelp_position, kelp_signal
        )
        
        ink_orders = self.create_orders_based_on_signal(
            self.products["ink"], ink_depth, ink_position, ink_signal
        )
        
        resin_orders = self.create_orders_based_on_signal(
            self.products["resin"], resin_depth, resin_position, resin_signal
        )
        
        # 将订单添加到结果中
        if kelp_orders:
            result[self.products["kelp"]] = kelp_orders
        if ink_orders:
            result[self.products["ink"]] = ink_orders
        if resin_orders:
            result[self.products["resin"]] = resin_orders
        
        # 交易员数据
        trader_data = json.dumps({
            "cycle": self.trading_cycle,
            "kelp_signal": round(kelp_signal, 2),
            "ink_signal": round(ink_signal, 2),
            "resin_signal": round(resin_signal, 2),
            "kelp_position": kelp_position,
            "ink_position": ink_position,
            "resin_position": resin_position
        })
        
        # 返回订单、转换量和交易员数据
        conversions = 0
        return result, conversions, trader_data


if __name__ == "__main__":
    # 这里可以编写简单的测试代码
    trader = Trader()
    print("负相关性策略初始化完成") 