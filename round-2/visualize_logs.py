#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import glob
import argparse
from datetime import datetime

def parse_log_file(log_file_path):
    """解析日志文件，提取关键交易数据"""
    data = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # 检查是否包含lambdaLog字段
                if 'lambdaLog' in entry and entry['lambdaLog'] and 'timestamp' in entry:
                    # 提取时间戳
                    timestamp = entry.get('timestamp', None)
                    
                    # 解析lambdaLog中的内容
                    log_content = entry['lambdaLog']
                    
                    # 简单解析，假设格式统一
                    if "价格:" in log_content and "信号:" in log_content:
                        try:
                            # 提取价格
                            price_start = log_content.find("价格:") + 4
                            price_end = log_content.find(",", price_start)
                            price = float(log_content[price_start:price_end].strip())
                            
                            # 提取信号
                            signal_start = log_content.find("信号:") + 4
                            signal_end = log_content.find(",", signal_start)
                            signal = float(log_content[signal_start:signal_end].strip())
                            
                            # 提取目标仓位
                            target_pos_start = log_content.find("目标仓位:") + 6
                            target_pos_end = log_content.find(",", target_pos_start)
                            target_position = int(log_content[target_pos_start:target_pos_end].strip())
                            
                            # 提取当前仓位
                            current_pos_start = log_content.find("当前仓位:") + 6
                            current_pos_end = log_content.find(",", current_pos_start)
                            current_position = int(log_content[current_pos_start:current_pos_end].strip())
                            
                            # 提取调整量
                            adjustment_start = log_content.find("调整:") + 4
                            adjustment_end = log_content.find(",", adjustment_start) if "," in log_content[adjustment_start:] else len(log_content)
                            adjustment = int(log_content[adjustment_start:adjustment_end].strip())
                            
                            # 尝试提取累计PnL（如果有）
                            cumulative_pnl = None
                            if "累计PnL:" in log_content:
                                pnl_start = log_content.find("累计PnL:") + 6
                                pnl_end = len(log_content)
                                try:
                                    cumulative_pnl = float(log_content[pnl_start:pnl_end].strip())
                                except:
                                    pass
                            
                            # 添加到数据列表
                            data_point = {
                                'timestamp': timestamp,
                                'price': price,
                                'signal': signal,
                                'target_position': target_position,
                                'current_position': current_position,
                                'adjustment': adjustment
                            }
                            
                            if cumulative_pnl is not None:
                                data_point['cumulative_pnl'] = cumulative_pnl
                                
                            data.append(data_point)
                        except (ValueError, IndexError) as e:
                            print(f"解析日志行时出错: {e}")
                            continue
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理日志时出错: {e}")
                continue
    
    print(f"从日志中提取了 {len(data)} 条交易记录")
    return pd.DataFrame(data)

def calculate_pnl(df):
    """计算策略的PnL (如果日志中没有提供)"""
    if len(df) < 2:
        return df
    
    # 如果日志已经包含累计PnL，直接使用
    if 'cumulative_pnl' in df.columns:
        print("使用日志中的累计PnL数据")
        return df
    
    # 否则，根据价格和仓位计算
    print("根据价格和仓位变化计算PnL")
    
    # 确保时间戳排序
    df = df.sort_values('timestamp')
    
    # 计算价格变化
    df['price_change'] = df['price'].diff()
    
    # 计算持仓价值变化（PnL）
    df['position_pnl'] = df['current_position'] * df['price_change']
    
    # 计算累计PnL
    df['cumulative_pnl'] = df['position_pnl'].cumsum()
    
    # 计算交易成本（假设每单位0.5的成本）
    df['trading_cost'] = abs(df['adjustment']) * 0.5
    
    # 计算净PnL
    df['net_pnl'] = df['cumulative_pnl'] - df['trading_cost'].cumsum()
    
    # 计算每日最大回撤
    df['high_watermark'] = df['net_pnl'].cummax()
    df['drawdown'] = df['high_watermark'] - df['net_pnl']
    
    return df

def calculate_statistics(df):
    """计算交易统计数据"""
    if len(df) < 2:
        return {}
    
    # 决定使用哪个列作为PnL
    pnl_column = 'cumulative_pnl' if 'cumulative_pnl' in df.columns else 'net_pnl' if 'net_pnl' in df.columns else None
    if not pnl_column:
        return {}
    
    # 获取最终PnL
    final_pnl = df[pnl_column].iloc[-1]
    
    # 计算日收益率变化
    if 'position_pnl' in df.columns:
        returns = df['position_pnl'].dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    else:
        # 使用累计PnL的差分作为收益
        returns = df[pnl_column].diff().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # 计算最大回撤
    if 'drawdown' in df.columns:
        max_drawdown = df['drawdown'].max()
    else:
        # 手动计算回撤
        rolling_max = df[pnl_column].cummax()
        drawdown = rolling_max - df[pnl_column]
        max_drawdown = drawdown.max()
    
    # 计算胜率
    df['trade'] = df['adjustment'] != 0
    
    if 'price_change' in df.columns:
        df['trade_result'] = np.sign(df['adjustment']) * df['price_change'].shift(-1)
        winning_trades = df[df['trade']]['trade_result'] > 0
    else:
        # 使用PnL变化判断交易结果
        df['pnl_change'] = df[pnl_column].diff().shift(-1)
        winning_trades = (df[df['trade']]['pnl_change'] > 0)
    
    total_trades = df['trade'].sum()
    win_rate = winning_trades.sum() / total_trades if total_trades > 0 else 0
    
    return {
        'final_pnl': final_pnl,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades
    }

def plot_trading_results(df, log_file, output_dir=None):
    """绘制交易结果图表"""
    if len(df) < 2:
        print(f"数据不足，无法生成图表: {log_file}")
        return
    
    # 确定使用哪个列作为PnL
    pnl_column = 'cumulative_pnl' if 'cumulative_pnl' in df.columns else 'net_pnl' if 'net_pnl' in df.columns else None
    
    # 创建一个包含4个子图的大图
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    
    # 第一个子图：价格和持仓
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['timestamp'], df['price'], label='价格', color='blue')
    ax1.set_xlabel('时间戳')
    ax1.set_ylabel('价格', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 添加一个次坐标轴，用于显示持仓
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(df['timestamp'], df['current_position'], color='lightgreen', alpha=0.3, label='持仓')
    ax1_twin.plot(df['timestamp'], df['current_position'], color='green', alpha=0.7)
    ax1_twin.set_ylabel('持仓', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1_twin.set_ylim(-55, 55)  # 设置Y轴限制
    
    # 添加移动平均线
    window_size = 20
    if len(df) > window_size:
        df['ma'] = df['price'].rolling(window=window_size).mean()
        # 添加标准差带
        df['std'] = df['price'].rolling(window=window_size).std()
        ax1.plot(df['timestamp'], df['ma'], label=f'{window_size}期移动平均', color='red', alpha=0.8)
        ax1.fill_between(df['timestamp'], df['ma'] - 2*df['std'], df['ma'] + 2*df['std'], 
                        color='red', alpha=0.1, label='±2标准差')
    
    # 第二个子图：信号强度
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df['timestamp'], df['signal'], label='交易信号', color='purple')
    ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='上阈值(1.5)')
    ax2.axhline(y=-1.5, color='green', linestyle='--', alpha=0.7, label='下阈值(-1.5)')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylabel('信号强度')
    ax2.legend(loc='upper right')
    
    # 第三个子图：PnL
    if pnl_column:
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(df['timestamp'], df[pnl_column], label='累计PnL', color='green')
        ax3.set_ylabel('盈亏(PnL)')
        ax3.legend(loc='upper left')
    
    # 第四个子图：交易和信号关系
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    # 绘制仓位调整
    ax4.bar(df['timestamp'], df['adjustment'], color='blue', alpha=0.6, label='仓位调整')
    ax4.set_ylabel('仓位调整')
    ax4.set_xlabel('时间戳')
    
    # 计算每次调整的正确率（如果可能）
    if 'price_change' in df.columns:
        # 计算每次调整后的价格变动
        df['correct_trade'] = np.sign(df['adjustment']) * np.sign(df['price_change'].shift(-1))
        df['trade_color'] = df['correct_trade'].apply(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray')
        
        # 在调整不为0的地方添加颜色标记
        for i, row in df[df['adjustment'] != 0].iterrows():
            ax4.scatter(row['timestamp'], row['adjustment'], 
                      color=row['trade_color'], s=50, zorder=5)
    
    ax4.legend(loc='upper right')
    
    # 添加标题
    stats = calculate_statistics(df)
    title = (f"SQUID_INK 均值回归策略回测结果\n"
             f"最终PnL: {stats.get('final_pnl', 0):.2f}, 夏普比率: {stats.get('sharpe_ratio', 0):.2f}\n"
             f"最大回撤: {stats.get('max_drawdown', 0):.2f}, 胜率: {stats.get('win_rate', 0)*100:.1f}%, "
             f"总交易次数: {stats.get('total_trades', 0)}")
    plt.suptitle(title, fontsize=15)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(log_file).replace('.log', '')
        output_file = os.path.join(output_dir, f"{base_name}_analysis.png")
        plt.savefig(output_file, dpi=150)
        print(f"分析图表已保存至: {output_file}")
    else:
        plt.show()
    
    plt.close()

def get_latest_log_file(backtests_dir):
    """获取最新的日志文件"""
    log_files = glob.glob(os.path.join(backtests_dir, '*.log'))
    if not log_files:
        return None
    
    return max(log_files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description='分析均值回归策略的交易日志')
    parser.add_argument('--log', type=str, help='日志文件路径')
    parser.add_argument('--output', type=str, default='analysis_results', help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示图表而不是保存')
    
    args = parser.parse_args()
    
    # 默认在round-2/backtests目录查找日志
    backtests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtests')
    
    if args.log:
        # 使用指定的日志文件
        log_file = args.log
        if not os.path.exists(log_file):
            print(f"指定的日志文件不存在: {log_file}")
            return
    else:
        # 使用最新的日志文件
        log_file = get_latest_log_file(backtests_dir)
        if not log_file:
            print(f"在 {backtests_dir} 目录下未找到日志文件")
            return
    
    print(f"分析日志文件: {log_file}")
    
    # 解析日志文件
    df = parse_log_file(log_file)
    
    if len(df) < 2:
        print(f"日志文件中未找到足够的交易数据: {log_file}")
        return
    
    # 计算PnL
    df = calculate_pnl(df)
    
    # 打印统计数据
    stats = calculate_statistics(df)
    print("\n===== 交易统计 =====")
    print(f"- 最终PnL: {stats.get('final_pnl', 0):.2f}")
    print(f"- 夏普比率: {stats.get('sharpe_ratio', 0):.2f}")
    print(f"- 最大回撤: {stats.get('max_drawdown', 0):.2f}")
    print(f"- 胜率: {stats.get('win_rate', 0)*100:.1f}%")
    print(f"- 总交易次数: {stats.get('total_trades', 0)}")
    
    # 绘制图表
    if args.show:
        plot_trading_results(df, log_file)
    else:
        plot_trading_results(df, log_file, args.output)

if __name__ == "__main__":
    main() 