#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datamodel import OrderDepth, TradingState, Order, Listing
from algo_trade import Trader
import unittest
from unittest.mock import MagicMock

class TestAlgoTrade(unittest.TestCase):
    
    def setUp(self):
        self.trader = Trader()
    
    def test_calculate_mid_price(self):
        """测试计算中间价格函数"""
        order_depth = OrderDepth()
        
        # 测试空订单簿
        self.assertIsNone(self.trader.calculate_mid_price(order_depth))
        
        # 测试只有买单
        order_depth.buy_orders = {100: 5, 99: 10}
        self.assertEqual(self.trader.calculate_mid_price(order_depth), 100)
        
        # 测试只有卖单
        order_depth.buy_orders = {}
        order_depth.sell_orders = {101: 5, 102: 10}
        self.assertEqual(self.trader.calculate_mid_price(order_depth), 101)
        
        # 测试有买单和卖单
        order_depth.buy_orders = {100: 5}
        order_depth.sell_orders = {101: 5}
        self.assertEqual(self.trader.calculate_mid_price(order_depth), 100.5)
    
    def test_calculate_z_score(self):
        """测试计算z-score函数"""
        # 测试数据不足
        self.assertEqual(self.trader.calculate_z_score(10), 0)
        
        # 添加足够的数据
        self.trader.price_diffs = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        # 测试标准差为0的情况
        self.assertEqual(self.trader.calculate_z_score(10), 0)
        
        # 测试正常情况
        self.trader.price_diffs = [10, 12, 8, 11, 9, 10, 11, 9, 10, 10]
        # 均值应该是10，标准差约为1.15
        # 当价差为12时，z-score应该约为(12-10)/1.15 ≈ 1.74
        self.assertAlmostEqual(self.trader.calculate_z_score(12), 1.74, places=1)
    
    def test_run_data_collection(self):
        """测试运行函数的数据收集部分"""
        # 创建模拟的交易状态
        mock_state = MagicMock(spec=TradingState)
        mock_state.position = {"KELP": 0, "SQUID_INK": 0}
        
        # 创建带有买卖订单的订单深度
        kelp_depth = OrderDepth()
        kelp_depth.buy_orders = {2000: 10}
        kelp_depth.sell_orders = {2005: 10}
        
        ink_depth = OrderDepth()
        ink_depth.buy_orders = {1995: 10}
        ink_depth.sell_orders = {2000: 10}
        
        mock_state.order_depths = {"KELP": kelp_depth, "SQUID_INK": ink_depth}
        
        # 运行交易函数
        result, _, _ = self.trader.run(mock_state)
        
        # 验证价格数据是否被收集
        self.assertEqual(len(self.trader.kelp_prices), 1)
        self.assertEqual(len(self.trader.ink_prices), 1)
        self.assertEqual(len(self.trader.price_diffs), 1)
        
        # 价格应该是买卖订单的中间价
        self.assertEqual(self.trader.kelp_prices[0], 2002.5)
        self.assertEqual(self.trader.ink_prices[0], 1997.5)
        
        # 价差应该是KELP - SQUID_INK
        self.assertEqual(self.trader.price_diffs[0], 5.0)
    
    def test_trade_execution(self):
        """测试交易执行逻辑"""
        # 创建模拟的交易状态
        mock_state = MagicMock(spec=TradingState)
        mock_state.position = {"KELP": 0, "SQUID_INK": 0}
        
        # 让我们先积累一些历史数据
        self.trader.kelp_prices = [2002.5] * 10
        self.trader.ink_prices = [1997.5] * 10
        self.trader.price_diffs = [5.0] * 10
        
        # 现在模拟一个异常的价差情况
        # KELP价格上涨，SQUID_INK价格下跌，使价差变大
        kelp_depth = OrderDepth()
        kelp_depth.buy_orders = {2010: 10}
        kelp_depth.sell_orders = {2015: 10}
        
        ink_depth = OrderDepth()
        ink_depth.buy_orders = {1985: 10}
        ink_depth.sell_orders = {1990: 10}
        
        mock_state.order_depths = {"KELP": kelp_depth, "SQUID_INK": ink_depth}
        
        # 运行交易函数
        result, _, _ = self.trader.run(mock_state)
        
        # 验证是否生成了交易订单
        # 价差扩大，应该卖出KELP，买入SQUID_INK
        self.assertIn("KELP", result)
        self.assertIn("SQUID_INK", result)
        
        kelp_orders = result["KELP"]
        ink_orders = result["SQUID_INK"]
        
        # 检查订单方向
        self.assertTrue(any(order.quantity < 0 for order in kelp_orders))  # 卖出KELP
        self.assertTrue(any(order.quantity > 0 for order in ink_orders))   # 买入SQUID_INK

if __name__ == "__main__":
    unittest.main() 