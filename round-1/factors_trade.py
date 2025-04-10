#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datamodel import Order, ProsperityEncoder, Symbol, Trade, TradingState, OrderDepth
from typing import Dict, List
import statistics
import json
import numpy as np

class Trader:
    def __init__(self):
        # 产品名称
        self.products = {
            "kelp": "KELP",
            "ink": "SQUID_INK",
            "resin": "RAINFOREST_RESIN"
        }
        
        # 每个产品的持仓限制
        self.position_limit = 20
        
        # 记录交易状态
        self.trading_cycle = 0
        
        # 历史数据
        self.history = {
            self.products["kelp"]: {
                "mid_prices": [],
                "bid_volumes": [],
                "ask_volumes": [],
                "spreads": [],
                "mid_depth_prices": [],
                "pressure_ratios": [],
                "price_deviations": [],
                "order_imbalances": []
            },
            self.products["ink"]: {
                "mid_prices": [],
                "bid_volumes": [],
                "ask_volumes": [],
                "spreads": [],
                "mid_depth_prices": [],
                "pressure_ratios": [],
                "price_deviations": [],
                "order_imbalances": []
            },
            self.products["resin"]: {
                "mid_prices": [],
                "bid_volumes": [],
                "ask_volumes": [],
                "spreads": [],
                "mid_depth_prices": [],
                "pressure_ratios": [],
                "price_deviations": [],
                "order_imbalances": []
            }
        }
        
        # 因子权重 (根据IC值的相对大小设置)
        self.factor_weights = {
            "pressure_ratio": -0.5,  # 负相关
            "price_deviation": -0.3,  # 负相关
            "order_imbalance": 0.2   # 正相关
        }
        
        # 交易信号阈值
        self.signal_threshold = 0.1
        
        # 交易基础数量
        self.base_quantity = 5
        
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
    
    def calculate_factors(self, order_depth: OrderDepth, mid_price: float, product: str):
        """计算三个关键因子"""
        # 初始化因子值
        factors = {
            "pressure_ratio": 0,
            "price_deviation": 0,
            "order_imbalance": 0
        }
        
        if not order_depth or mid_price is None:
            return factors
        
        # 计算买卖压力
        buy_pressure = sum(abs(volume) for volume in order_depth.buy_orders.values())
        sell_pressure = sum(abs(volume) for volume in order_depth.sell_orders.values())
        
        # 避免除零
        if sell_pressure > 0:
            factors["pressure_ratio"] = buy_pressure / sell_pressure
        else:
            factors["pressure_ratio"] = buy_pressure * 1.0 if buy_pressure > 0 else 0
        
        # 计算价格偏离度
        mid_depth_price = self.calculate_mid_depth_price(order_depth)
        if mid_depth_price:
            factors["price_deviation"] = (mid_price - mid_depth_price) / mid_depth_price
            
            # 更新历史数据
            self.history[product]["mid_depth_prices"].append(mid_depth_price)
        
        # 计算订单不平衡度
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_volume = abs(order_depth.buy_orders[best_bid])
            best_ask_volume = abs(order_depth.sell_orders[best_ask])
            
            total_top_volume = best_bid_volume + best_ask_volume
            if total_top_volume > 0:
                factors["order_imbalance"] = (best_bid_volume - best_ask_volume) / total_top_volume
        
        # 更新历史数据
        self.history[product]["pressure_ratios"].append(factors["pressure_ratio"])
        self.history[product]["price_deviations"].append(factors["price_deviation"])
        self.history[product]["order_imbalances"].append(factors["order_imbalance"])
        
        return factors
    
    def calculate_signal(self, factors):
        """根据因子计算交易信号"""
        signal = 0
        
        # 加权组合因子
        for factor_name, factor_value in factors.items():
            if factor_name in self.factor_weights:
                signal += factor_value * self.factor_weights[factor_name]
        
        return signal
    
    def normalize_signal(self, signal, product):
        """归一化信号"""
        # 如果历史数据少于5个周期，不做额外处理
        if len(self.history[product]["pressure_ratios"]) < 5:
            return signal
            
        # 使用最近5个周期信号的标准差来归一化
        recent_signals = []
        for i in range(min(5, len(self.history[product]["pressure_ratios"]))):
            recent_pressure = self.history[product]["pressure_ratios"][-(i+1)]
            recent_deviation = self.history[product]["price_deviations"][-(i+1)]
            recent_imbalance = self.history[product]["order_imbalances"][-(i+1)]
            
            recent_factors = {
                "pressure_ratio": recent_pressure,
                "price_deviation": recent_deviation,
                "order_imbalance": recent_imbalance
            }
            
            recent_signals.append(self.calculate_signal(recent_factors))
        
        if len(recent_signals) > 1:
            std = np.std(recent_signals)
            if std > 0:
                return signal / std
        
        return signal

    def create_orders(self, product, order_depth, position, signal):
        """根据信号创建订单"""
        if not order_depth:
            return []
            
        orders = []
        
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
                    orders.append(Order(product, best_ask, buy_qty))
                    
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
                    orders.append(Order(product, best_bid, -sell_qty))
                    
        return orders
    
    def add_risk_management_orders(self, result, product, order_depth, position):
        """添加风险管理订单，减少持仓"""
        if not product in result:
            result[product] = []
            
        # 如果持仓过多，尝试卖出
        if position > self.position_limit * 0.9 and len(order_depth.buy_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = abs(order_depth.buy_orders[best_bid])
            # 平仓数量
            sell_qty = min(3, best_bid_volume, position - self.position_limit * 0.7)
            if sell_qty > 0:
                result[product].append(Order(product, best_bid, -sell_qty))
            
        # 如果持仓过少（负值过大），尝试买入
        elif position < -self.position_limit * 0.9 and len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = abs(order_depth.sell_orders[best_ask])
            # 平仓数量
            buy_qty = min(3, best_ask_volume, self.position_limit * 0.7 + position)
            if buy_qty > 0:
                result[product].append(Order(product, best_ask, buy_qty))
    
    def run(self, state: TradingState):
        """
        基于因子的交易策略
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
            self.history[self.products["kelp"]]["mid_prices"].append(kelp_mid_price)
        if ink_mid_price:
            self.history[self.products["ink"]]["mid_prices"].append(ink_mid_price)
        if resin_mid_price:
            self.history[self.products["resin"]]["mid_prices"].append(resin_mid_price)
        
        # 获取当前持仓
        kelp_position = state.position.get(self.products["kelp"], 0)
        ink_position = state.position.get(self.products["ink"], 0)
        resin_position = state.position.get(self.products["resin"], 0)
        
        # 检查是否有风险持仓
        kelp_risk = abs(kelp_position) > self.position_limit * 0.85
        ink_risk = abs(ink_position) > self.position_limit * 0.85
        # resin不参与交易，所以不检查风险
        
        # 风险管理：优先处理，如果持仓接近限制，先平仓
        if kelp_risk:
            self.add_risk_management_orders(result, self.products["kelp"], kelp_depth, kelp_position)
        if ink_risk:
            self.add_risk_management_orders(result, self.products["ink"], ink_depth, ink_position)
        # resin不参与交易，所以不执行风险管理
            
        # 如果没有风险仓位，才进行正常交易
        if not (kelp_risk or ink_risk):
            # 计算各产品的因子
            kelp_factors = self.calculate_factors(kelp_depth, kelp_mid_price, self.products["kelp"])
            ink_factors = self.calculate_factors(ink_depth, ink_mid_price, self.products["ink"])
            # 仍然计算resin因子，用于记录数据，但不会用于交易
            resin_factors = self.calculate_factors(resin_depth, resin_mid_price, self.products["resin"])
            
            # 计算交易信号
            kelp_signal = self.calculate_signal(kelp_factors)
            ink_signal = self.calculate_signal(ink_factors)
            resin_signal = self.calculate_signal(resin_factors)
            
            # 归一化信号
            kelp_signal = self.normalize_signal(kelp_signal, self.products["kelp"])
            ink_signal = self.normalize_signal(ink_signal, self.products["ink"])
            resin_signal = self.normalize_signal(resin_signal, self.products["resin"])
            
            # 创建订单 - 只为KELP和SQUID_INK创建订单
            kelp_orders = self.create_orders(
                self.products["kelp"], kelp_depth, kelp_position, kelp_signal
            )
            ink_orders = self.create_orders(
                self.products["ink"], ink_depth, ink_position, ink_signal
            )
            # 不为resin创建订单
            
            # 将订单添加到结果中
            if kelp_orders:
                if self.products["kelp"] not in result:
                    result[self.products["kelp"]] = []
                result[self.products["kelp"]].extend(kelp_orders)
                
            if ink_orders:
                if self.products["ink"] not in result:
                    result[self.products["ink"]] = []
                result[self.products["ink"]].extend(ink_orders)
            
            # 不添加resin订单
        
        # 交易员数据
        trader_data = json.dumps({
            "cycle": self.trading_cycle,
            "kelp_position": kelp_position,
            "ink_position": ink_position,
            "resin_position": resin_position,
            "kelp_signal": kelp_signal if 'kelp_signal' in locals() else 0,
            "ink_signal": ink_signal if 'ink_signal' in locals() else 0,
            "resin_signal": resin_signal if 'resin_signal' in locals() else 0,
            "risk_mode": kelp_risk or ink_risk
        })
        
        # 返回订单、转换量和交易员数据
        conversions = 0
        return result, conversions, trader_data 