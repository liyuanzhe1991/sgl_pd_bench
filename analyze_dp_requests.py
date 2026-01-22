#!/usr/bin/env python3
"""
分析 prefill.log 中每个 iter 的每个 dp_rank 请求数，使用堆叠柱状图展示
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def parse_log(log_path: str, min_tokens: int = 2):
    """解析日志文件，提取每个 iter 中每个 dp_rank 的请求数
    
    Args:
        log_path: 日志文件路径
        min_tokens: 最小 token 数，低于此值的请求会被过滤掉（默认 2，过滤 warmup/health check）
    """
    # 扩展正则表达式，捕获 #new-token 的值
    pattern = r'\[.*?DP(\d+)\s+TP\d+\s+EP\d+\]\s+Prefill batch \[(\d+)\],\s+#new-seq:\s+(\d+),\s+#new-token:\s+(\d+)'
    
    data = defaultdict(lambda: defaultdict(int))
    all_dp_ranks = set()
    all_iters = set()
    filtered_count = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                dp_rank = int(match.group(1))
                iter_num = int(match.group(2))
                new_seq = int(match.group(3))
                new_token = int(match.group(4))
                
                # 过滤掉 token 数小于 min_tokens 的请求（warmup/health check）
                if new_token < min_tokens:
                    filtered_count += 1
                    continue
                
                data[iter_num][dp_rank] = new_seq
                all_dp_ranks.add(dp_rank)
                all_iters.add(iter_num)
    
    if filtered_count > 0:
        print(f"已过滤 {filtered_count} 条 token 数 < {min_tokens} 的请求（warmup/health check）")
    
    return data, sorted(all_dp_ranks), sorted(all_iters)


def create_stacked_bar(data, dp_ranks, iters, output_path: str):
    """堆叠柱状图 - 每个 iter 一个柱子，颜色区分 rank，右侧显示总数和各rank汇总"""
    n_iters = len(iters)
    n_dp_ranks = len(dp_ranks)
    
    # 创建数据矩阵
    matrix = np.zeros((n_dp_ranks, n_iters))
    for i, iter_num in enumerate(iters):
        for j, dp_rank in enumerate(dp_ranks):
            matrix[j, i] = data[iter_num].get(dp_rank, 0)
    
    # 计算每个 iter 的总数
    iter_totals = matrix.sum(axis=0)
    # 计算每个 rank 的总数
    rank_totals = matrix.sum(axis=1)
    grand_total = matrix.sum()
    
    # 自适应图表宽度，右侧留出空间显示总数
    fig_width = max(18, n_iters * 0.15 + 4)
    fig = plt.figure(figsize=(fig_width, 9))
    
    # 使用 GridSpec 创建布局：左边大图，右边两个小表格
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 1], 
                          wspace=0.05, hspace=0.15)
    
    ax = fig.add_subplot(gs[:, 0])  # 左侧占满两行
    ax_iter_table = fig.add_subplot(gs[0, 1])  # 右上：iter 总数表
    ax_rank_table = fig.add_subplot(gs[1, 1])  # 右下：rank 总数表
    
    x = np.arange(n_iters)
    colors = plt.cm.tab10(np.linspace(0, 1, n_dp_ranks))
    
    # 堆叠柱状图
    bottom = np.zeros(n_iters)
    for j, dp_rank in enumerate(dp_ranks):
        ax.bar(x, matrix[j], bottom=bottom, label=f'DP{dp_rank}', 
               color=colors[j], width=0.85, edgecolor='white', linewidth=0.5)
        bottom += matrix[j]
    
    # X 轴标签
    if n_iters <= 60:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in iters], rotation=90, ha='center', fontsize=8)
    else:
        step = max(1, n_iters // 40)
        x_ticks = range(0, n_iters, step)
        ax.set_xticks(list(x_ticks))
        ax.set_xticklabels([str(iters[i]) for i in x_ticks], rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Requests (#new-seq)', fontsize=12)
    ax.set_title('Stacked Bar: Requests per Iteration by DP Rank', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.5, n_iters - 0.5)
    
    # ========== 右上表格：每个 iter 的总数 ==========
    ax_iter_table.axis('off')
    ax_iter_table.set_title('Total per Iter', fontsize=11, fontweight='bold', pad=5)
    
    iter_table_data = []
    for i, iter_num in enumerate(iters):
        iter_table_data.append([str(iter_num), str(int(iter_totals[i]))])
    
    # 如果 iter 太多，只显示部分
    max_rows = 18
    if len(iter_table_data) > max_rows:
        display_iter_data = iter_table_data[:8] + [['...', '...']] + iter_table_data[-8:]
    else:
        display_iter_data = iter_table_data
    
    table1 = ax_iter_table.table(
        cellText=display_iter_data,
        colLabels=['Iter', 'Total'],
        loc='upper center',
        cellLoc='center',
        colWidths=[0.45, 0.45]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1.1, 0.9)
    
    for (row, col), cell in table1.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#E6E6E6')
        cell.set_height(0.045)
    
    # ========== 右下表格：每个 rank 的总请求数 ==========
    ax_rank_table.axis('off')
    ax_rank_table.set_title('Total per DP Rank', fontsize=11, fontweight='bold', pad=5)
    
    rank_table_data = []
    for j, dp_rank in enumerate(dp_ranks):
        total = int(rank_totals[j])
        pct = rank_totals[j] / grand_total * 100 if grand_total > 0 else 0
        rank_table_data.append([f'DP{dp_rank}', str(total), f'{pct:.1f}%'])
    
    # 添加总计行
    rank_table_data.append(['Sum', str(int(grand_total)), '100%'])
    
    table2 = ax_rank_table.table(
        cellText=rank_table_data,
        colLabels=['Rank', 'Total', '%'],
        loc='upper center',
        cellLoc='center',
        colWidths=[0.35, 0.35, 0.30]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.1, 1.2)
    
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#E6E6E6')
        # 最后一行（Sum）加粗并设置背景色
        if row == len(rank_table_data):
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#D4EDDA')
        cell.set_height(0.08)
    
    # 底部添加汇总统计
    stats_text = f'Grand Total: {int(grand_total)} requests | {n_iters} iterations | {n_dp_ranks} DP ranks'
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=11, style='italic')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"堆叠柱状图已保存到: {output_path}")
    plt.close()


def print_summary(data, dp_ranks, iters):
    """打印汇总统计信息"""
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    print(f"总 iter 数: {len(iters)}")
    print(f"DP ranks: {dp_ranks}")
    print(f"Iter 范围: {min(iters)} - {max(iters)}" if iters else "无数据")
    
    total_per_dp = defaultdict(int)
    grand_total = 0
    for iter_num in iters:
        for dp_rank in dp_ranks:
            count = data[iter_num].get(dp_rank, 0)
            total_per_dp[dp_rank] += count
            grand_total += count
    
    print("\n各 DP Rank 总请求数:")
    for dp_rank in dp_ranks:
        pct = total_per_dp[dp_rank] / grand_total * 100 if grand_total > 0 else 0
        print(f"  DP{dp_rank}: {total_per_dp[dp_rank]:4d} ({pct:.1f}%)")
    print(f"\n总请求数: {grand_total}")
    print("=" * 80)


def main():
    log_path = "/mnt/launch/prefill.log"
    output_path = "/mnt/launch/dp_stacked_bar.png"
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    if not Path(log_path).exists():
        print(f"错误: 日志文件不存在: {log_path}")
        sys.exit(1)
    
    print(f"正在解析日志文件: {log_path}")
    
    data, dp_ranks, iters = parse_log(log_path)
    
    if not iters:
        print("未找到任何 Prefill batch 日志记录")
        sys.exit(1)
    
    print(f"找到 {len(iters)} 个 iterations, {len(dp_ranks)} 个 DP ranks")
    
    print_summary(data, dp_ranks, iters)
    
    create_stacked_bar(data, dp_ranks, iters, output_path)
    
    print("\n✅ 图表生成完成!")


if __name__ == "__main__":
    main()
