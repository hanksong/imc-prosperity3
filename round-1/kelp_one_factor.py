#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datamodel import Order, ProsperityEncoder, Symbol, Trade, TradingState, OrderDepth
from typing import Dict, List
import json
import numpy as np

class Trader:
    def __init__(self):
        # 产品名称
        self.kelp = "KELP"
        
        # 持仓限制
        self.position_limit = 20
        
        # 记录交易状态
        self.trading_cycle = 0
        
        # 历史数据
        self.history = {
            "mid_prices": [],
            "mid_depth_prices": [],
            "price_deviations": [],
            "signals": []
        }
        
        # 因子权重 (根据IC值设置)
        self.factor_weight = -1.0  # price_deviation负相关
        
        # 交易信号阈值，较小以增加交易频率
        self.signal_threshold = 0.05
        
        # 交易基础数量，较大以充分利用持仓限制
        self.base_quantity = 8
        
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
    
    def calculate_mid_depth_price(self, order_depth: OrderDepth) -> float:
        """计算加权中间价格"""
        if not order_depth:
            return None
        
        total_volume = 0
        weighted_sum = 0
        
        for price, volume in order_depth.buy_orders.items():
            weighted_sum += price * abs(volume)
            total_volume += abs(volume)
            
        for price, volume in order_depth.sell_orders.items():
            weighted_sum += price * abs(volume)
            total_volume += abs(volume)
            
        if total_volume == 0:
            return self.calculate_mid_price(order_depth)
            
        return weighted_sum / total_volume
    
    def calculate_price_deviation(self, order_depth: OrderDepth, mid_price: float) -> float:
        """计算价格偏离度因子"""
        if not order_depth or mid_price is None:
            return 0
        
        # 计算价格偏离度
        mid_depth_price = self.calculate_mid_depth_price(order_depth)
        if mid_depth_price:
            # 更新历史数据
            self.history["mid_depth_prices"].append(mid_depth_price)
            return (mid_price - mid_depth_price) / mid_depth_price
        
        return 0
    
    def calculate_signal(self, price_deviation):
        """根据价格偏离度计算交易信号"""
        signal = price_deviation * self.factor_weight
        
        # 将信号记录到历史数据
        self.history["signals"].append(signal)
        
        return signal
    
    def normalize_signal(self, signal):
        """归一化信号"""
        # 如果历史数据少于10个周期，不做额外处理
        if len(self.history["signals"]) < 10:
            return signal
            
        # 使用最近10个周期信号的标准差来归一化
        recent_signals = self.history["signals"][-10:]
        
        if len(recent_signals) > 1:
            std = np.std(recent_signals)
            if std > 0:
                return signal / std
        
        return signal
    
    def should_take_profit(self, position, signal, mid_price):
        """判断是否应该平仓获利"""
        # 如果历史价格不足，不执行获利操作
        if len(self.history["mid_prices"]) < 10:
            return False
        
        # 计算过去10步的收益率
        past_price = self.history["mid_prices"][-10]
        return_10 = (mid_price / past_price) - 1
        
        # 如果是多头且return10显著为正，则平仓获利
        if position > 0 and return_10 > 0.005:  # 0.5%的获利目标
            return True
        
        # 如果是空头且return10显著为负，则平仓获利
        if position < 0 and return_10 < -0.005:  # -0.5%的获利目标
            return True
        
        return False

    def create_orders(self, order_depth, position, signal, mid_price):
        """根据信号创建订单"""
        if not order_depth or mid_price is None:
            return []
            
        orders = []
        
        # 检查是否应该获利平仓
        if self.should_take_profit(position, signal, mid_price):
            # 如果是多头，平多获利
            if position > 0 and len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = abs(order_depth.buy_orders[best_bid])
                sell_qty = min(position, best_bid_volume)
                
                if sell_qty > 0:
                    return [Order(self.kelp, best_bid, -sell_qty)]
            
            # 如果是空头，平空获利
            elif position < 0 and len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = abs(order_depth.sell_orders[best_ask])
                buy_qty = min(-position, best_ask_volume)
                
                if buy_qty > 0:
                    return [Order(self.kelp, best_ask, buy_qty)]
                    
            # 如果已经决定平仓获利，不执行下面的建仓操作
            return orders
        
        # 如果信号强度不足，不交易
        if abs(signal) < self.signal_threshold:
            return []
            
        # 信号为正，预期价格上涨，买入
        if signal > 0:
            # 确保不超过持仓限制
            if position < self.position_limit and len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = abs(order_depth.sell_orders[best_ask])
                
                # 计算可买入数量，信号越强交易量越大
                available_qty = self.position_limit - position
                signal_scale = min(1.0, signal / 0.2)  # 限制在0-1之间
                desired_qty = max(1, round(self.base_quantity * signal_scale))
                buy_qty = min(best_ask_volume, available_qty, desired_qty)
                
                if buy_qty > 0:
                    orders.append(Order(self.kelp, best_ask, buy_qty))
                    
        # 信号为负，预期价格下跌，卖出
        elif signal < 0:
            # 确保不超过持仓限制
            if position > -self.position_limit and len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = abs(order_depth.buy_orders[best_bid])
                
                # 计算可卖出数量，信号越强交易量越大
                available_qty = self.position_limit + position
                signal_scale = min(1.0, abs(signal) / 0.2)  # 限制在0-1之间
                desired_qty = max(1, round(self.base_quantity * signal_scale))
                sell_qty = min(best_bid_volume, available_qty, desired_qty)
                
                if sell_qty > 0:
                    orders.append(Order(self.kelp, best_bid, -sell_qty))
                    
        return orders
    
    def add_risk_management_orders(self, result, order_depth, position):
        """添加风险管理订单，减少持仓"""
        if self.kelp not in result:
            result[self.kelp] = []
            
        # 如果持仓过多，尝试卖出
        if position > self.position_limit * 0.9 and len(order_depth.buy_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = abs(order_depth.buy_orders[best_bid])
            # 平仓数量
            sell_qty = min(5, best_bid_volume, position - self.position_limit * 0.7)
            if sell_qty > 0:
                result[self.kelp].append(Order(self.kelp, best_bid, -sell_qty))
            
        # 如果持仓过少（负值过大），尝试买入
        elif position < -self.position_limit * 0.9 and len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = abs(order_depth.sell_orders[best_ask])
            # 平仓数量
            buy_qty = min(5, best_ask_volume, self.position_limit * 0.7 + position)
            if buy_qty > 0:
                result[self.kelp].append(Order(self.kelp, best_ask, buy_qty))
    
    def run(self, state: TradingState):
        """
        基于price_deviation单因子的KELP交易策略
        """
        # 自增交易周期
        self.trading_cycle += 1
        
        # 初始化结果字典
        result = {}
        
        # 获取KELP的订单深度
        kelp_depth = state.order_depths.get(self.kelp, OrderDepth())
        
        # 计算中间价格
        kelp_mid_price = self.calculate_mid_price(kelp_depth)
        
        # 更新历史价格
        if kelp_mid_price:
            self.history["mid_prices"].append(kelp_mid_price)
        
        # 获取当前持仓
        kelp_position = state.position.get(self.kelp, 0)
        
        # 检查是否有风险持仓
        kelp_risk = abs(kelp_position) > self.position_limit * 0.85
        
        # 风险管理：优先处理，如果持仓接近限制，先平仓
        if kelp_risk:
            self.add_risk_management_orders(result, kelp_depth, kelp_position)
        else:
            # 计算price_deviation因子
            price_deviation = self.calculate_price_deviation(kelp_depth, kelp_mid_price)
            
            # 计算交易信号
            signal = self.calculate_signal(price_deviation)
            
            # 归一化信号
            signal = self.normalize_signal(signal)
            
            # 创建订单
            kelp_orders = self.create_orders(kelp_depth, kelp_position, signal, kelp_mid_price)
            
            # 将订单添加到结果中
            if kelp_orders:
                if self.kelp not in result:
                    result[self.kelp] = []
                result[self.kelp].extend(kelp_orders)
        
        # 交易员数据
        trader_data = json.dumps({
            "cycle": self.trading_cycle,
            "kelp_position": kelp_position,
            "price_deviation": price_deviation if 'price_deviation' in locals() else 0,
            "signal": signal if 'signal' in locals() else 0,
            "mid_price": kelp_mid_price if kelp_mid_price else 0,
            "risk_mode": kelp_risk
        })
        
        # 返回订单、转换量和交易员数据
        conversions = 0
        return result, conversions, trader_data 