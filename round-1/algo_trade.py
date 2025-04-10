#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datamodel import Order, ProsperityEncoder, Symbol, Trade, TradingState, OrderDepth
from typing import Dict, List
import statistics
import json

class Trader:
    def __init__(self):
        # 初始化价格历史记录
        self.kelp_prices = []
        self.ink_prices = []
        self.resin_prices = []
        
        # 产品名称
        self.products = {
            "kelp": "KELP",
            "ink": "SQUID_INK",  # 注意这里是SQUID_INK不是SQUILD_INK
            "resin": "RAINFOREST_RESIN"
        }
        
        # 每个产品的持仓限制
        self.position_limit = 20
        
        # 价格阈值 - 增大阈值以减少交易频率，但增加每次交易的盈利潜力
        self.kelp_ink_ratio_threshold = 0.03  # 3%价差
        self.resin_threshold = 0.005  # 0.5%价差
        
        # 记录交易状态
        self.trading_cycle = 0
        
        # 交易量设置 - 更积极的交易
        self.trade_quantity = 3  # 默认每次交易3个单位
        
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
    
    def calculate_fair_value(self, product, mid_price, kelp_mid, ink_mid, resin_mid):
        """根据三角关系计算产品的公平价值 - 使用更有效的关系"""
        if product == self.products["kelp"]:
            if ink_mid and resin_mid:
                # 使用SQUID_INK和RAINFOREST_RESIN的关系估算KELP的公平价值
                # 调整计算方式，增强价差
                return (ink_mid * 1.01 + resin_mid * 0.99) / 2
        elif product == self.products["ink"]:
            if kelp_mid and resin_mid:
                # 使用KELP和RAINFOREST_RESIN的关系估算SQUID_INK的公平价值
                # 调整计算方式，增强价差
                return (kelp_mid * 0.99 + resin_mid * 1.01) / 2
        elif product == self.products["resin"]:
            if kelp_mid and ink_mid:
                # 使用KELP和SQUID_INK的关系估算RAINFOREST_RESIN的公平价值
                # 调整计算方式，增强价差
                return (kelp_mid * 1.01 + ink_mid * 0.99) / 2
        
        # 默认返回当前中间价
        return mid_price
        
    def create_orders(self, product, order_depth, position, fair_value, mid_price):
        """根据公平价值创建订单 - 反向交易策略"""
        if not order_depth or not mid_price or not fair_value:
            return []
            
        orders = []
        price_diff_pct = abs(mid_price - fair_value) / fair_value
        
        # 设置价差阈值，根据产品不同可能有不同的阈值
        threshold = self.resin_threshold
        if product in [self.products["kelp"], self.products["ink"]]:
            threshold = self.kelp_ink_ratio_threshold
            
        # 如果价差小于阈值，不交易
        if price_diff_pct < threshold:
            return []
            
        # 反向交易：如果市场价格低于公平价值，卖出 (原来是买入)
        if mid_price < fair_value:
            # 确保不超过持仓限制
            if position > -self.position_limit and len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                
                # 计算可卖出数量
                available_qty = self.position_limit + position
                sell_qty = min(best_bid_volume, available_qty, self.trade_quantity)  # 使用更积极的交易量
                
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    
        # 反向交易：如果市场价格高于公平价值，买入 (原来是卖出)
        elif mid_price > fair_value:
            # 确保不超过持仓限制
            if position < self.position_limit and len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                
                # 计算可买入数量
                available_qty = self.position_limit - position
                buy_qty = min(abs(best_ask_volume), available_qty, self.trade_quantity)  # 使用更积极的交易量
                
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    
        return orders
    
    def run(self, state: TradingState):
        """
        实现三角套利策略 - 反向交易版本
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
        
        # 获取当前持仓
        kelp_position = state.position.get(self.products["kelp"], 0)
        ink_position = state.position.get(self.products["ink"], 0)
        resin_position = state.position.get(self.products["resin"], 0)
        
        # 检查是否有风险持仓
        kelp_risk = abs(kelp_position) > self.position_limit * 0.85
        ink_risk = abs(ink_position) > self.position_limit * 0.85
        resin_risk = abs(resin_position) > self.position_limit * 0.85
        
        # 风险管理：优先处理，如果持仓接近限制，先平仓
        if kelp_risk:
            self.add_risk_management_orders(result, self.products["kelp"], kelp_depth, kelp_position)
        if ink_risk:
            self.add_risk_management_orders(result, self.products["ink"], ink_depth, ink_position)
        if resin_risk:
            self.add_risk_management_orders(result, self.products["resin"], resin_depth, resin_position)
            
        # 如果没有风险仓位，才进行正常交易
        if not (kelp_risk or ink_risk or resin_risk):
            # 计算每个产品的公平价值
            kelp_fair_value = self.calculate_fair_value(
                self.products["kelp"], kelp_mid_price, kelp_mid_price, ink_mid_price, resin_mid_price
            )
            ink_fair_value = self.calculate_fair_value(
                self.products["ink"], ink_mid_price, kelp_mid_price, ink_mid_price, resin_mid_price
            )
            resin_fair_value = self.calculate_fair_value(
                self.products["resin"], resin_mid_price, kelp_mid_price, ink_mid_price, resin_mid_price
            )
            
            # 创建订单
            kelp_orders = self.create_orders(
                self.products["kelp"], kelp_depth, kelp_position, kelp_fair_value, kelp_mid_price
            )
            ink_orders = self.create_orders(
                self.products["ink"], ink_depth, ink_position, ink_fair_value, ink_mid_price
            )
            resin_orders = self.create_orders(
                self.products["resin"], resin_depth, resin_position, resin_fair_value, resin_mid_price
            )
            
            # 将订单添加到结果中
            if kelp_orders:
                if self.products["kelp"] not in result:
                    result[self.products["kelp"]] = []
                result[self.products["kelp"]].extend(kelp_orders)
                
            if ink_orders:
                if self.products["ink"] not in result:
                    result[self.products["ink"]] = []
                result[self.products["ink"]].extend(ink_orders)
                
            if resin_orders:
                if self.products["resin"] not in result:
                    result[self.products["resin"]] = []
                result[self.products["resin"]].extend(resin_orders)
        
        # 交易员数据
        trader_data = json.dumps({
            "cycle": self.trading_cycle,
            "kelp_position": kelp_position,
            "ink_position": ink_position,
            "resin_position": resin_position,
            "risk_mode": kelp_risk or ink_risk or resin_risk
        })
        
        # 返回订单、转换量和交易员数据
        conversions = 0
        return result, conversions, trader_data
        
    def add_risk_management_orders(self, result, product, order_depth, position):
        """添加风险管理订单，减少持仓"""
        if not product in result:
            result[product] = []
            
        # 如果持仓过多，尝试卖出 - 更积极的风险管理，使用90%阈值
        if position > self.position_limit * 0.9 and len(order_depth.buy_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            # 增加平仓数量到2个单位
            result[product].append(Order(product, best_bid, -min(2, best_bid_volume)))
            
        # 如果持仓过少（负值过大），尝试买入 - 更积极的风险管理，使用90%阈值
        elif position < -self.position_limit * 0.9 and len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]
            # 增加平仓数量到2个单位
            result[product].append(Order(product, best_ask, min(2, abs(best_ask_volume))))