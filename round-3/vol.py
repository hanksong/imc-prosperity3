import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# 更快的正态分布累积分布函数实现
def norm_cdf_fast(x):
    """
    标准正态分布累积函数的快速近似实现
    使用误差函数(erf)近似，比传统近似算法更快
    """
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))

# 计算隐含波动率的函数 - 使用牛顿迭代法加速
def calculate_implied_volatility(option_price, S, K, T, r, option_type='call'):
    """
    使用牛顿迭代法计算隐含波动率（比二分法更快）
    
    参数:
    option_price: 期权价格
    S: 标的资产当前价格
    K: 期权行权价
    T: 到期时间（年）
    r: 无风险利率
    option_type: 期权类型（'call'或'put'）
    """
    # 如果期权价格接近于0或无意义，直接返回
    if option_price <= 0.001:
        return 0.001
    
    # BS公式价格计算
    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm_cdf_fast(d1) - K * np.exp(-r * T) * norm_cdf_fast(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm_cdf_fast(-d2) - S * norm_cdf_fast(-d1)
        
        return price
    
    # BS公式对sigma的导数
    def bs_vega(sigma):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
        return vega
    
    # 初始猜测值 - 使用简单估计
    sigma = np.sqrt(2 * np.pi / T) * option_price / S
    
    # 确保初始猜测值在合理范围内
    sigma = max(0.001, min(sigma, 2.0))
    
    # 牛顿迭代
    max_iterations = 20
    precision = 0.00001
    
    for i in range(max_iterations):
        price = bs_price(sigma)
        price_diff = price - option_price
        
        # 如果差异足够小，返回结果
        if abs(price_diff) < precision:
            return sigma
        
        # 计算Vega并更新sigma
        vega = bs_vega(sigma)
        
        # 防止Vega接近于0导致的不稳定性
        if abs(vega) < 1e-10:
            vega = 1e-10
            
        # 牛顿迭代步骤
        sigma = sigma - price_diff / vega
        
        # 限制sigma不要太大或太小
        sigma = max(0.001, min(sigma, 5.0))
    
    # 如果达到最大迭代次数，返回当前最佳估计值
    return sigma

# 使用缓存优化多次计算相同参数的情况
# 为了在数据较多时保持速度，我们使用LRU缓存来存储最近的计算结果
iv_cache = {}
def calculate_implied_volatility_cached(option_price, S, K, T, r, option_type='call'):
    """
    带缓存的隐含波动率计算，避免重复计算
    """
    # 创建缓存键
    cache_key = (round(option_price, 4), round(S, 4), K, round(T, 6), r, option_type)
    
    # 检查缓存
    if cache_key in iv_cache:
        return iv_cache[cache_key]
    
    # 计算IV
    iv = calculate_implied_volatility(option_price, S, K, T, r, option_type)
    
    # 存储到缓存
    iv_cache[cache_key] = iv
    
    # 限制缓存大小，防止内存泄漏
    if len(iv_cache) > 10000:
        # 清除一半的缓存项
        keys_to_remove = list(iv_cache.keys())[:5000]
        for k in keys_to_remove:
            del iv_cache[k]
    
    return iv

# 向量化处理批量数据
def calculate_volatility_surface_vectorized(price_data, volcanic_products, current_day=3):
    """计算波动率曲面数据（优化版本）"""
    strikes = extract_strike_prices(volcanic_products)
    
    # 找到标的资产价格 (VOLCANIC_ROCK)
    underlying_price = price_data[price_data['product'] == 'VOLCANIC_ROCK']
    if underlying_price.empty:
        raise Exception("找不到标的资产 VOLCANIC_ROCK 的价格数据")
    
    # 按时间戳分组获取最新价格
    latest_prices = price_data.groupby(['timestamp', 'product'])['mid_price'].last().reset_index()
    
    # 计算波动率数据
    volatility_data = []
    
    # 假设无风险利率
    r = 0.01
    
    # 预先提取所有需要的数据
    timestamps = latest_prices['timestamp'].unique()
    
    # 创建进度指示器
    total_steps = len(timestamps)
    print(f"开始计算波动率曲面，总共 {total_steps} 个时间点")
    
    # 为每个时间戳执行计算
    for idx, timestamp in enumerate(timestamps):
        # 显示进度
        if idx % 50 == 0:
            print(f"进度: {idx}/{total_steps} ({idx/total_steps*100:.1f}%)")
        
        # 获取当前时间戳的标的价格
        S_values = latest_prices[(latest_prices['timestamp'] == timestamp) & 
                              (latest_prices['product'] == 'VOLCANIC_ROCK')]['mid_price'].values
        
        if len(S_values) == 0:
            continue
        S = S_values[0]
        
        # 计算到期时间
        T = calculate_time_to_maturity(current_day)
        
        # 同一时间戳的所有期权同时处理
        for product in volcanic_products:
            if product == 'VOLCANIC_ROCK' or product not in strikes:
                continue
                
            K = strikes[product]
            
            # 获取期权价格
            option_prices = latest_prices[(latest_prices['timestamp'] == timestamp) & 
                                         (latest_prices['product'] == product)]['mid_price'].values
            
            if len(option_prices) == 0:
                continue
            option_price = option_prices[0]
            
            # 确定期权类型
            option_type = 'call' if K > S else 'put'
            
            # 计算m_t = log(K/S)/sqrt(T)
            m_t = math.log(K/S) / math.sqrt(T)
            
            # 计算隐含波动率 (使用缓存版本)
            try:
                iv = calculate_implied_volatility_cached(option_price, S, K, T, r, option_type)
                
                # 只保留有效值
                if iv is not None and 0.001 <= iv <= 5.0:
                    volatility_data.append({
                        'timestamp': timestamp,
                        'strike': K,
                        'implied_volatility': iv,
                        'product': product,
                        'underlying_price': S,
                        'option_price': option_price,
                        'm_t': m_t  # 添加m_t值
                    })
            except:
                pass
    
    print("波动率曲面计算完成！")
    return pd.DataFrame(volatility_data)

# 加载数据
def load_price_data():
    price = pd.DataFrame()
    for day in [0, 1, 2]:
        try:
            price_day = pd.read_csv(f'round-3/round-3-island-data-bottle/prices_round_3_day_{day}.csv', sep=';')
            price = pd.concat([price, price_day])
        except FileNotFoundError:
            print(f"警告: 数据文件 round-3/round-3-island-data-bottle/prices_round_3_day_{day}.csv 未找到")
    
    if price.empty:
        raise Exception("无法加载价格数据")
    
    price.fillna(0, inplace=True)
    price['timestamp'] = price['timestamp']/100 + price['day'] * 10000 + 10000
    price.set_index('timestamp', inplace=True)
    
    # 计算盘口中间价
    price['mid_price'] = (price['bid_price_1'] + price['ask_price_1']) / 2
    price['mid_depth_price'] = (price['bid_price_1'] * price['bid_volume_1'] + price['bid_price_2']*price['bid_volume_2'] 
                               + price['bid_price_3']*price['bid_volume_3'] + price['ask_price_1']*price['ask_volume_1'] 
                               + price['ask_price_2']*price['ask_volume_2'] + price['ask_price_3']*price['ask_volume_3']
                              ) / (price['bid_volume_1']+price['bid_volume_2']+price['bid_volume_3']+
                                   price['ask_volume_1']+price['ask_volume_2']+price['ask_volume_3'])
    
    # 筛选火山岩期权产品
    volcanic_products = [product for product in price['product'].unique() if 'VOLCANIC' in product]
    price = price[price['product'].isin(volcanic_products)].reset_index()
    
    return price, volcanic_products

# 提取行权价和计算隐含波动率
def extract_strike_prices(products):
    """从产品名称中提取行权价"""
    strikes = {}
    for product in products:
        if 'VOUCHER' in product:
            parts = product.split('_')
            if len(parts) > 2:
                strike = int(parts[-1])
                strikes[product] = strike
    return strikes

def calculate_time_to_maturity(current_day, expiry_day=7):
    """计算到期时间（以年为单位）"""
    return (expiry_day - current_day) / 365

# 保留原始函数作为备用
# 但默认调用优化版本
def calculate_volatility_surface(price_data, volcanic_products, current_day=3, use_optimized=True):
    """计算波动率曲面数据"""
    if use_optimized:
        return calculate_volatility_surface_vectorized(price_data, volcanic_products, current_day)
    
    # 以下是原始版本
    strikes = extract_strike_prices(volcanic_products)
    
    # 找到标的资产价格 (VOLCANIC_ROCK)
    underlying_price = price_data[price_data['product'] == 'VOLCANIC_ROCK']
    if underlying_price.empty:
        raise Exception("找不到标的资产 VOLCANIC_ROCK 的价格数据")
    
    # 按时间戳分组获取最新价格
    latest_prices = price_data.groupby(['timestamp', 'product'])['mid_price'].last().reset_index()
    
    # 计算波动率数据
    volatility_data = []
    
    # 假设无风险利率
    r = 0.01
    
    # 遍历时间戳
    for timestamp in latest_prices['timestamp'].unique():
        # 获取当前时间戳的标的价格
        S = latest_prices[(latest_prices['timestamp'] == timestamp) & 
                          (latest_prices['product'] == 'VOLCANIC_ROCK')]['mid_price'].values
        
        if len(S) == 0:
            continue
        S = S[0]
        
        # 计算到期时间
        T = calculate_time_to_maturity(current_day)
        
        # 遍历每个期权产品
        for product in volcanic_products:
            if product == 'VOLCANIC_ROCK' or product not in strikes:
                continue
                
            K = strikes[product]
            
            # 获取期权价格
            option_prices = latest_prices[(latest_prices['timestamp'] == timestamp) & 
                                         (latest_prices['product'] == product)]['mid_price'].values
            
            if len(option_prices) == 0:
                continue
            option_price = option_prices[0]
            
            # 确定期权类型
            option_type = 'call' if K > S else 'put'
            
            # 计算m_t = log(K/S)/sqrt(T)
            m_t = math.log(K/S) / math.sqrt(T)
            
            # 计算隐含波动率
            try:
                iv = calculate_implied_volatility(option_price, S, K, T, r, option_type)
                if iv is not None:
                    volatility_data.append({
                        'timestamp': timestamp,
                        'strike': K,
                        'implied_volatility': iv,
                        'product': product,
                        'underlying_price': S,
                        'option_price': option_price,
                        'm_t': m_t  # 添加m_t值
                    })
            except:
                pass
    
    return pd.DataFrame(volatility_data)

# 绘制波动率曲面
def plot_volatility_surface(vol_data):
    """绘制波动率曲面和微笑"""
    if vol_data.empty:
        print("没有有效的波动率数据用于绘图")
        return
    
    # 创建多个图表
    fig = plt.figure(figsize=(15, 10))
    
    # 3D波动率曲面
    ax1 = fig.add_subplot(221, projection='3d')
    
    timestamps = vol_data['timestamp'].unique()
    strikes = vol_data['strike'].unique()
    
    X, Y = np.meshgrid(strikes, timestamps)
    Z = np.zeros_like(X, dtype=float)
    
    for i, t in enumerate(timestamps):
        for j, k in enumerate(strikes):
            matching_data = vol_data[(vol_data['timestamp'] == t) & (vol_data['strike'] == k)]
            if not matching_data.empty:
                Z[i, j] = matching_data['implied_volatility'].values[0]
    
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    ax1.set_xlabel('行权价')
    ax1.set_ylabel('时间戳')
    ax1.set_zlabel('隐含波动率')
    ax1.set_title('波动率曲面')
    
    # 传统波动率微笑 (IV vs Strike)
    ax2 = fig.add_subplot(222)
    
    # 获取最新时间戳
    latest_timestamp = vol_data['timestamp'].max()
    latest_data = vol_data[vol_data['timestamp'] == latest_timestamp]
    
    if not latest_data.empty:
        ax2.scatter(latest_data['strike'], latest_data['implied_volatility'], marker='o')
        ax2.set_xlabel('行权价 (K)')
        ax2.set_ylabel('隐含波动率 (IV)')
        ax2.set_title(f'时间戳 {latest_timestamp} 的传统波动率微笑')
        
        # 添加拟合曲线
        try:
            # 简单的二次多项式拟合
            xdata = latest_data['strike'].values
            ydata = latest_data['implied_volatility'].values
            
            # 使用numpy的多项式拟合
            coeffs = np.polyfit(xdata, ydata, 2)
            poly = np.poly1d(coeffs)
            
            # 绘制拟合曲线
            x_fit = np.linspace(min(xdata), max(xdata), 100)
            y_fit = poly(x_fit)
            ax2.plot(x_fit, y_fit, 'r-', label='拟合曲线')
            ax2.legend()
        except:
            print("无法拟合传统波动率微笑曲线")
            
    # 添加新的波动率微笑图表 (IV vs m_t)
    ax3 = fig.add_subplot(223)
    
    if not latest_data.empty:
        ax3.scatter(latest_data['m_t'], latest_data['implied_volatility'], marker='o')
        ax3.set_xlabel('标准化价内值 (m_t)')
        ax3.set_ylabel('隐含波动率 (IV)')
        ax3.set_title(f'时间戳 {latest_timestamp} 的标准化波动率微笑')
        
        # 添加m_t的拟合曲线
        try:
            # 二次多项式拟合
            xdata = latest_data['m_t'].values
            ydata = latest_data['implied_volatility'].values
            
            # 使用numpy的多项式拟合
            coeffs = np.polyfit(xdata, ydata, 2)
            poly = np.poly1d(coeffs)
            
            # 绘制拟合曲线
            x_fit = np.linspace(min(xdata), max(xdata), 100)
            y_fit = poly(x_fit)
            ax3.plot(x_fit, y_fit, 'r-', label='拟合曲线')
            
            # 计算m_t=0时的IV值（基准隐含波动率）
            base_iv = poly(0)
            ax3.axvline(x=0, color='g', linestyle='--', alpha=0.7)
            ax3.scatter([0], [base_iv], color='g', s=100, marker='x')
            ax3.annotate(f'基准IV: {base_iv:.4f}', 
                         xy=(0, base_iv), 
                         xytext=(0.1, base_iv),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                         fontsize=10)
            
            ax3.legend()
        except Exception as e:
            print(f"无法拟合标准化波动率微笑曲线: {e}")
    
    # 添加基准隐含波动率随时间变化的图表
    ax4 = fig.add_subplot(224)
    
    # 计算每个时间戳的基准IV
    base_iv_data = []
    
    for timestamp in timestamps:
        ts_data = vol_data[vol_data['timestamp'] == timestamp]
        
        if len(ts_data) >= 3:  # 确保有足够的数据点进行拟合
            try:
                # 使用m_t进行拟合
                xdata = ts_data['m_t'].values
                ydata = ts_data['implied_volatility'].values
                
                # 二次多项式拟合
                coeffs = np.polyfit(xdata, ydata, 2)
                poly = np.poly1d(coeffs)
                
                # 计算m_t=0时的IV
                base_iv = poly(0)
                base_iv_data.append({
                    'timestamp': timestamp,
                    'base_iv': base_iv
                })
            except:
                pass
    
    if base_iv_data:
        base_iv_df = pd.DataFrame(base_iv_data)
        ax4.plot(base_iv_df['timestamp'], base_iv_df['base_iv'], 'b-')
        ax4.scatter(base_iv_df['timestamp'], base_iv_df['base_iv'], color='b')
        ax4.set_xlabel('时间戳')
        ax4.set_ylabel('基准隐含波动率')
        ax4.set_title('基准IV随时间的变化')
        
        # 保存基准IV数据
        base_iv_df.to_csv('base_iv_data.csv', index=False)
        print("基准IV数据已保存到 base_iv_data.csv")
    
    plt.tight_layout()
    plt.show()
    
    return base_iv_data

# 主函数
def main():
    try:
        start_time = pd.Timestamp.now()
        print(f"开始时间: {start_time}")
        
        # 加载数据
        price_data, volcanic_products = load_price_data()
        print(f"加载了 {len(price_data)} 行价格数据")
        print(f"发现火山岩产品: {volcanic_products}")
        
        # 计算波动率表面
        vol_data = calculate_volatility_surface(price_data, volcanic_products)
        print(f"计算了 {len(vol_data)} 个隐含波动率数据点")
        
        end_time = pd.Timestamp.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"计算完成时间: {end_time}")
        print(f"总执行时间: {execution_time:.2f} 秒")
        
        # 显示波动率数据
        if not vol_data.empty:
            print("\n隐含波动率数据样例:")
            print(vol_data.head())
            
            # 绘制波动率曲面和微笑
            base_iv_data = plot_volatility_surface(vol_data)
            
            # 保存波动率数据
            vol_data.to_csv('volatility_data.csv', index=False)
            print("波动率数据已保存到 volatility_data.csv")
            
            # 显示m_t数据
            print("\nm_t数据样例:")
            print(vol_data[['timestamp', 'product', 'strike', 'm_t', 'implied_volatility']].head())
            
        else:
            print("未能计算出有效的隐含波动率数据")
            
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 