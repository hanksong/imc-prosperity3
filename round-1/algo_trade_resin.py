#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datamodel import Order, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List
import json

class Trader:
    def __init__(self):
        self.resin_position = 0
        self.position_limit = 20
        self.buy_price = 9996  # 买入价格
        self.sell_price = 10004  # 卖出价格
        self.product = "RAINFOREST_RESIN"
        self.order_id = 0
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        专注于交易RAINFOREST_RESIN的简单策略
        
        Args:
            state: 当前交易状态
            
        Returns:
            Dict[str, List[Order]]: 订单字典
        """
        # 初始化订单结果
        result = {}
        
        # 记录当前持仓
        position = state.position.get(self.product, 0)
        self.resin_position = position
        
        # 打印日志
        # print(f"开始交易 {self.product}, 当前持仓: {position}")
        
        # 检查RAINFOREST_RESIN是否有市场深度数据
        if self.product in state.order_depths:
            # 获取RAINFOREST_RESIN的订单深度
            order_depth = state.order_depths[self.product]
            
            # 初始化当前产品的订单列表
            orders: List[Order] = []
            
            # 获取市场买单和卖单
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders
            
            # 计算中间价格（如果有买卖盘）
            mid_price = None
            if buy_orders and sell_orders:
                best_bid = max(buy_orders.keys())
                best_ask = min(sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                # print(f"{self.product} 中间价: {mid_price}")
            
            # 策略1: 如果有卖单价格低于我们的买入价格，直接买入
            if sell_orders:
                best_ask = min(sell_orders.keys())
                if best_ask <= self.buy_price and position < self.position_limit:
                    # 可用数量 = 仓位限制 - 当前持仓
                    available_qty = self.position_limit - position
                    # 最大可买数量 = 卖单数量和可用数量的较小值
                    buy_qty = min(abs(sell_orders[best_ask]), available_qty)
                    
                    if buy_qty > 0:
                        orders.append(Order(self.product, best_ask, buy_qty))
                        # print(f"以 {best_ask} 价格买入 {buy_qty} 个 {self.product}")
            
            # 策略2: 如果有买单价格高于我们的卖出价格，直接卖出
            if buy_orders:
                best_bid = max(buy_orders.keys())
                if best_bid >= self.sell_price and position > -self.position_limit:
                    # 可用数量 = 仓位限制 - 当前持仓的负值
                    available_qty = self.position_limit + position
                    # 最大可卖数量 = 买单数量和可用数量的较小值
                    sell_qty = min(buy_orders[best_bid], available_qty)
                    
                    if sell_qty > 0:
                        orders.append(Order(self.product, best_bid, -sell_qty))
                        # print(f"以 {best_bid} 价格卖出 {sell_qty} 个 {self.product}")
            
            # 策略3: 主动挂单
            # 如果当前持仓未达上限，在买入价格挂买单
            if position < self.position_limit and mid_price and mid_price > self.buy_price:
                buy_qty = self.position_limit - position
                if buy_qty > 0:
                    orders.append(Order(self.product, self.buy_price, buy_qty))
                    # print(f"挂买单: {self.buy_price} x {buy_qty}")
            
            # 如果当前持仓未达下限，在卖出价格挂卖单
            if position > -self.position_limit and mid_price and mid_price < self.sell_price:
                sell_qty = self.position_limit + position
                if sell_qty > 0:
                    orders.append(Order(self.product, self.sell_price, -sell_qty))
                    # print(f"挂卖单: {self.sell_price} x {sell_qty}")
            
            # 将订单添加到结果中
            result[self.product] = orders
        
        # 返回我们想要执行的订单列表、转换量和交易员数据
        # 修复错误：返回三个值而不是两个
        traderData = "RAINFOREST_RESIN_TRADER"  # 交易员状态数据
        conversions = 0  # 转换量，暂时设为0
        return result, conversions, traderData