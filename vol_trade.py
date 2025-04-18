import pandas as pd
import numpy as np
import math
from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation
from typing import List, Dict

# 正态分布累积分布函数的近似实现
def norm_cdf(x):
    """
    标准正态分布累积函数的近似实现
    替代scipy.stats.norm.cdf
    """
    # 常数
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    # 保存x的符号
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)
    
    # 公式近似
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return 0.5 * (1.0 + sign * y)

# 计算隐含波动率的函数
def calculate_implied_volatility(option_price, S, K, T, r, option_type='call'):
    """
    计算隐含波动率
    
    参数:
    option_price: 期权价格
    S: 标的资产当前价格
    K: 期权行权价
    T: 到期时间（年）
    r: 无风险利率
    option_type: 期权类型（'call'或'put'）
    """
    def black_scholes(sigma):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
        return price - option_price
    
    # 使用二分法求解
    a, b = 0.0001, 3  # 波动率范围
    
    if black_scholes(a) * black_scholes(b) > 0:
        return None  # 无解或超出范围
    
    while (b - a) > 0.0001:
        c = (a + b) / 2
        if black_scholes(c) == 0:
            return c
        if black_scholes(a) * black_scholes(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2

# 计算期权理论价格的函数
def calculate_option_price(S, K, T, r, sigma, option_type='call'):
    """
    使用Black-Scholes模型计算期权价格
    
    参数:
    S: 标的资产当前价格
    K: 期权行权价
    T: 到期时间（年）
    r: 无风险利率
    sigma: 波动率
    option_type: 期权类型（'call'或'put'）
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    
    return price

# 提取行权价
def extract_strike_price(product):
    """从产品名称中提取行权价"""
    if 'VOUCHER' in product:
        parts = product.split('_')
        if len(parts) > 2:
            return int(parts[-1])
    return None

# 计算到期时间
def calculate_time_to_maturity(current_day, expiry_day=7):
    """计算到期时间（以年为单位）"""
    return (expiry_day - current_day) / 365

# 计算标准化价内值 m_t
def calculate_moneyness(S, K, T):
    """
    计算标准化价内值 m_t = log(K/S)/sqrt(T)
    
    参数:
    S: 标的资产价格
    K: 行权价
    T: 到期时间（年）
    """
    return math.log(K/S) / math.sqrt(T)

# 拟合波动率微笑
def fit_volatility_smile(m_t_values, iv_values):
    """
    拟合波动率微笑曲线
    
    参数:
    m_t_values: 标准化价内值数组
    iv_values: 隐含波动率数组
    
    返回:
    拟合的二次多项式函数
    """
    if len(m_t_values) < 3:
        return None
    
    try:
        # 二次多项式拟合
        coeffs = np.polyfit(m_t_values, iv_values, 2)
        poly = np.poly1d(coeffs)
        return poly
    except:
        return None

class Trader:
    def __init__(self):
        self.day = 3  # 当前是第3天
        self.expiry_day = 7  # 期权在第7天到期
        self.position_limits = {
            'VOLCANIC_ROCK': 20,
            'VOLCANIC_ROCK_VOUCHER_9500': 20,
            'VOLCANIC_ROCK_VOUCHER_9750': 20,
            'VOLCANIC_ROCK_VOUCHER_10000': 20,
            'VOLCANIC_ROCK_VOUCHER_10250': 20,
            'VOLCANIC_ROCK_VOUCHER_10500': 20
        }
        self.r = 0.01  # 无风险利率
        self.volatility_history = {}  # 存储历史波动率数据
        self.price_history = {}  # 存储历史价格数据
        self.theoretical_prices = {}  # 存储理论价格
        self.market_making_spread = 1  # 做市商买卖价差
        self.min_edge = 2  # 最小套利边际
        self.max_delta_exposure = 10  # 最大Delta敞口
        
        # 新增的m_t数据结构
        self.m_t_data = {}  # 存储标准化价内值数据
        self.smile_fit = None  # 存储拟合的波动率微笑曲线
        self.base_iv_history = []  # 存储基准IV历史数据
        self.base_iv_trend = 0  # 基准IV趋势（1:上升, -1:下降, 0:稳定）
        self.iv_deviation = {}  # 各期权的IV偏离程度
        
    def store_price_data(self, state: TradingState):
        """存储当前价格数据"""
        timestamp = state.timestamp
        
        for product in state.order_depths:
            if product not in self.price_history:
                self.price_history[product] = []
                
            order_depth = state.order_depths[product]
            
            # 计算中间价
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
            elif order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = best_bid
            elif order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = best_ask
            else:
                continue  # 没有报价
                
            self.price_history[product].append({
                'timestamp': timestamp,
                'mid_price': mid_price
            })
    
    def calculate_implied_volatilities(self, state: TradingState):
        """计算所有期权的隐含波动率和m_t"""
        # 确保有标的资产价格
        if 'VOLCANIC_ROCK' not in self.price_history or not self.price_history['VOLCANIC_ROCK']:
            return
            
        # 获取最新的标的资产价格
        S = self.price_history['VOLCANIC_ROCK'][-1]['mid_price']
        T = calculate_time_to_maturity(self.day, self.expiry_day)
        
        # 存储当前时间戳的m_t和iv数据用于拟合波动率微笑
        current_m_t = []
        current_iv = []
        
        # 遍历所有期权
        for product in state.order_depths:
            if 'VOUCHER' not in product:
                continue
                
            # 获取行权价
            K = extract_strike_price(product)
            if K is None:
                continue
                
            # 确保有期权价格数据
            if product not in self.price_history or not self.price_history[product]:
                continue
                
            # 获取最新的期权价格
            option_price = self.price_history[product][-1]['mid_price']
            
            # 确定期权类型
            option_type = 'call' if K > S else 'put'
            
            # 计算标准化价内值 m_t
            m_t = calculate_moneyness(S, K, T)
            
            # 计算隐含波动率
            try:
                iv = calculate_implied_volatility(option_price, S, K, T, self.r, option_type)
                
                if iv is not None:
                    if product not in self.volatility_history:
                        self.volatility_history[product] = []
                        self.m_t_data[product] = []
                        
                    # 存储波动率数据
                    self.volatility_history[product].append({
                        'timestamp': state.timestamp,
                        'implied_volatility': iv,
                        'underlying_price': S,
                        'option_price': option_price
                    })
                    
                    # 存储m_t数据
                    self.m_t_data[product].append({
                        'timestamp': state.timestamp,
                        'm_t': m_t,
                        'implied_volatility': iv
                    })
                    
                    # 添加到当前数据集用于拟合
                    current_m_t.append(m_t)
                    current_iv.append(iv)
            except:
                pass
        
        # 拟合波动率微笑
        if len(current_m_t) >= 3:
            self.smile_fit = fit_volatility_smile(current_m_t, current_iv)
            
            # 如果拟合成功，计算基准IV
            if self.smile_fit is not None:
                base_iv = self.smile_fit(0)  # m_t=0时的IV
                self.base_iv_history.append({
                    'timestamp': state.timestamp,
                    'base_iv': base_iv
                })
                
                # 更新基准IV趋势
                if len(self.base_iv_history) >= 2:
                    prev_base_iv = self.base_iv_history[-2]['base_iv']
                    if base_iv > prev_base_iv * 1.02:  # 上升超过2%
                        self.base_iv_trend = 1
                    elif base_iv < prev_base_iv * 0.98:  # 下降超过2%
                        self.base_iv_trend = -1
                    else:
                        self.base_iv_trend = 0
                
                # 计算每个期权的IV偏离
                for product in self.m_t_data:
                    if self.m_t_data[product]:
                        latest_m_t = self.m_t_data[product][-1]['m_t']
                        latest_iv = self.m_t_data[product][-1]['implied_volatility']
                        expected_iv = self.smile_fit(latest_m_t)
                        
                        # 计算偏离程度（实际IV与拟合IV的比值）
                        self.iv_deviation[product] = latest_iv / expected_iv if expected_iv > 0 else 1.0
    
    def calculate_theoretical_prices(self, state: TradingState):
        """计算所有期权的理论价格（基于拟合的微笑曲线）"""
        # 确保有标的资产价格
        if 'VOLCANIC_ROCK' not in self.price_history or not self.price_history['VOLCANIC_ROCK']:
            return
            
        # 获取最新的标的资产价格
        S = self.price_history['VOLCANIC_ROCK'][-1]['mid_price']
        T = calculate_time_to_maturity(self.day, self.expiry_day)
        
        # 初始化理论价格字典
        self.theoretical_prices = {}
        
        # 如果有拟合的波动率微笑曲线
        if self.smile_fit is not None:
            # 遍历所有期权
            for product in state.order_depths:
                if 'VOUCHER' not in product:
                    continue
                    
                # 获取行权价
                K = extract_strike_price(product)
                if K is None:
                    continue
                
                # 计算m_t
                m_t = calculate_moneyness(S, K, T)
                
                # 使用拟合的波动率微笑计算理论隐含波动率
                theo_iv = self.smile_fit(m_t)
                
                # 确定期权类型
                option_type = 'call' if K > S else 'put'
                
                # 计算理论价格
                try:
                    price = calculate_option_price(S, K, T, self.r, theo_iv, option_type)
                    self.theoretical_prices[product] = price
                except:
                    # 如果使用拟合IV失败，尝试使用历史IV
                    if product in self.volatility_history and self.volatility_history[product]:
                        sigma = self.volatility_history[product][-1]['implied_volatility']
                        try:
                            price = calculate_option_price(S, K, T, self.r, sigma, option_type)
                            self.theoretical_prices[product] = price
                        except:
                            pass
        else:
            # 使用传统方法计算理论价格
            for product in state.order_depths:
                if 'VOUCHER' not in product:
                    continue
                    
                # 获取行权价
                K = extract_strike_price(product)
                if K is None:
                    continue
                    
                # 确保有波动率数据
                if product not in self.volatility_history or not self.volatility_history[product]:
                    continue
                    
                # 使用最新的隐含波动率
                sigma = self.volatility_history[product][-1]['implied_volatility']
                
                # 确定期权类型
                option_type = 'call' if K > S else 'put'
                
                # 计算理论价格
                try:
                    price = calculate_option_price(S, K, T, self.r, sigma, option_type)
                    self.theoretical_prices[product] = price
                except:
                    pass
    
    def get_best_price(self, order_depth: OrderDepth, side: str):
        """获取最优价格"""
        if side == 'buy' and order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif side == 'sell' and order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return None
    
    def calculate_position_delta(self, state: TradingState):
        """计算当前头寸的Delta敞口"""
        # 确保有标的资产价格
        if 'VOLCANIC_ROCK' not in self.price_history or not self.price_history['VOLCANIC_ROCK']:
            return 0
            
        # 获取最新的标的资产价格
        S = self.price_history['VOLCANIC_ROCK'][-1]['mid_price']
        T = calculate_time_to_maturity(self.day, self.expiry_day)
        
        total_delta = 0
        
        # 标的资产的Delta为1
        total_delta += state.position.get('VOLCANIC_ROCK', 0)
        
        # 计算期权的Delta
        for product, position in state.position.items():
            if 'VOUCHER' not in product or position == 0:
                continue
                
            # 获取行权价
            K = extract_strike_price(product)
            if K is None:
                continue
                
            # 确保有波动率数据
            if product not in self.volatility_history or not self.volatility_history[product]:
                continue
                
            # 使用最新的隐含波动率
            sigma = self.volatility_history[product][-1]['implied_volatility']
            
            # 计算Delta
            d1 = (np.log(S / K) + (self.r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            
            if K > S:  # Call
                delta = norm_cdf(d1)
            else:  # Put
                delta = norm_cdf(d1) - 1
                
            # 累加Delta敞口
            total_delta += position * delta
            
        return total_delta
    
    def determine_option_strategy(self, state: TradingState):
        """确定期权策略（基于m_t和波动率微笑分析）"""
        orders = {}
        
        # 计算总体Delta敞口
        total_delta = self.calculate_position_delta(state)
        
        for product in state.order_depths:
            orders[product] = []
            order_depth = state.order_depths[product]
            
            # 标的资产策略
            if product == 'VOLCANIC_ROCK':
                # 如果Delta敞口过大，进行对冲
                if abs(total_delta) > self.max_delta_exposure:
                    # Delta为正时卖出标的，Delta为负时买入标的
                    if total_delta > 0 and order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        qty_to_sell = min(order_depth.buy_orders[best_bid], 
                                        self.position_limits[product] + state.position.get(product, 0),
                                        int(total_delta))
                        if qty_to_sell > 0:
                            orders[product].append(Order(product, best_bid, -qty_to_sell))
                            
                    elif total_delta < 0 and order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        qty_to_buy = min(order_depth.sell_orders[best_ask], 
                                       self.position_limits[product] - state.position.get(product, 0),
                                       int(-total_delta))
                        if qty_to_buy > 0:
                            orders[product].append(Order(product, best_ask, qty_to_buy))
                
                # 基于基准IV趋势的方向性交易
                if self.base_iv_trend != 0 and len(self.base_iv_history) >= 3:
                    current_pos = state.position.get(product, 0)
                    
                    # 如果基准IV上升趋势，预期波动性增加，买入标的
                    if self.base_iv_trend > 0 and current_pos < self.position_limits[product] * 0.5:
                        if order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            qty_to_buy = min(order_depth.sell_orders[best_ask], 
                                          self.position_limits[product] - current_pos,
                                          5)  # 限制每次交易数量
                            if qty_to_buy > 0:
                                orders[product].append(Order(product, best_ask, qty_to_buy))
                    
                    # 如果基准IV下降趋势，预期波动性减少，卖出标的
                    elif self.base_iv_trend < 0 and current_pos > -self.position_limits[product] * 0.5:
                        if order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            qty_to_sell = min(order_depth.buy_orders[best_bid], 
                                           self.position_limits[product] + current_pos,
                                           5)  # 限制每次交易数量
                            if qty_to_sell > 0:
                                orders[product].append(Order(product, best_bid, -qty_to_sell))
                                
                continue  # 处理完标的资产，继续下一个产品
            
            # 期权策略
            if 'VOUCHER' not in product:
                continue
                
            # 使用理论价格进行套利
            if product in self.theoretical_prices:
                theo_price = self.theoretical_prices[product]
                
                # 考虑IV偏离因素调整套利边际
                edge_factor = 1.0
                if product in self.iv_deviation:
                    # IV异常高的期权提高卖出边际，IV异常低的期权提高买入边际
                    if self.iv_deviation[product] > 1.1:  # IV比预期高10%以上
                        edge_factor = 0.8  # 降低买入边际，提高卖出边际
                    elif self.iv_deviation[product] < 0.9:  # IV比预期低10%以上
                        edge_factor = 1.2  # 提高买入边际，降低卖出边际
                
                # 买入低估期权
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    if theo_price > best_ask + self.min_edge * edge_factor:
                        current_position = state.position.get(product, 0)
                        pos_limit = self.position_limits[product]
                        
                        qty_to_buy = min(order_depth.sell_orders[best_ask], 
                                       pos_limit - current_position)
                        
                        if qty_to_buy > 0:
                            orders[product].append(Order(product, best_ask, qty_to_buy))
                
                # 卖出高估期权
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    if theo_price < best_bid - self.min_edge / edge_factor:
                        current_position = state.position.get(product, 0)
                        pos_limit = self.position_limits[product]
                        
                        qty_to_sell = min(order_depth.buy_orders[best_bid],
                                        pos_limit + current_position)
                        
                        if qty_to_sell > 0:
                            orders[product].append(Order(product, best_bid, -qty_to_sell))
            else:
                # 如果没有理论价格，使用做市商策略
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    spread = best_ask - best_bid
                    
                    # 只在价差足够大时做市
                    if spread > self.market_making_spread * 2:
                        mid_price = (best_bid + best_ask) / 2
                        new_bid = int(mid_price - self.market_making_spread / 2)
                        new_ask = int(mid_price + self.market_making_spread / 2)
                        
                        # 确保不越过当前最优价格
                        if new_bid < best_bid and new_ask > best_ask:
                            # 确定买卖数量
                            current_position = state.position.get(product, 0)
                            pos_limit = self.position_limits[product]
                            
                            buy_quantity = min(5, pos_limit - current_position)
                            sell_quantity = min(5, pos_limit + current_position)
                            
                            if buy_quantity > 0:
                                orders[product].append(Order(product, new_bid, buy_quantity))
                            if sell_quantity > 0:
                                orders[product].append(Order(product, new_ask, -sell_quantity))
        
        # 扁平化订单
        result = {}
        for product, product_orders in orders.items():
            if product_orders:
                result[product] = product_orders
                
        return result
    
    def run(self, state: TradingState):
        """
        主交易函数
        """
        # 存储当前价格数据
        self.store_price_data(state)
        
        # 计算隐含波动率和标准化价内值
        self.calculate_implied_volatilities(state)
        
        # 计算期权理论价格
        self.calculate_theoretical_prices(state)
        
        # 确定交易策略并生成订单
        result = self.determine_option_strategy(state)
        
        # 用于下一轮执行的trader数据
        # 将波动率和基准IV数据编码为字符串
        trader_data = str({
            'day': self.day,
            'volatility_history_length': {product: len(history) for product, history in self.volatility_history.items()},
            'base_iv_trend': self.base_iv_trend,
            'base_iv_history_length': len(self.base_iv_history)
        })
        
        # 无转换
        conversions = 0
        
        return result, conversions, trader_data 