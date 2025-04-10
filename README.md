# 配对交易策略(Pair Trading Strategy)

## 策略概述

这是一个基于统计套利的配对交易策略，主要关注两个产品之间的价格关系：KELP和SQUID_INK。策略假设这两个产品的价格差异（价差）会围绕一个均值波动，当价差显著偏离均值时，可以通过买入相对低估的产品，卖出相对高估的产品来获利。

## 策略工作原理

1. **数据收集和分析**：策略会持续收集KELP和SQUID_INK的中间价格，并计算两者价差的均值和标准差。

2. **信号生成**：使用z-score来衡量当前价差相对于历史价差的偏离程度。z-score = (当前价差 - 均值) / 标准差。

3. **交易执行**：
   - 当z-score > 阈值（默认1.0）：价差高于正常水平，卖出KELP，买入SQUID_INK
   - 当z-score < -阈值：价差低于正常水平，买入KELP，卖出SQUID_INK
   
4. **风险管理**：
   - 设置持仓上限（默认20）
   - 当持仓接近上限时，开始平仓
   - 限制单次交易量
   - 需要足够的历史数据才能开始交易

## 使用方法

```python
from algo_trade import Trader
from datamodel import TradingState

# 创建交易员实例
trader = Trader()

# 在交易循环中调用run方法
def trading_loop(state: TradingState):
    # 运行策略
    result, conversions, trader_data = trader.run(state)
    return result, conversions, trader_data
```

## 测试

提供了一个测试脚本`test_algo_trade.py`来验证策略的各个组件:

```
python test_algo_trade.py
```

## 优化方向

1. 动态调整z-score阈值
2. 实现自适应的持仓管理
3. 加入更复杂的风险管理机制
4. 考虑更多金融产品间的相关性
5. 结合其他技术指标进行交易决策 