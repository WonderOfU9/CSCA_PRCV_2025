import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 100

# 正常显示中文
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 加载数据
df = pd.read_csv('/Merged_test_output.csv')

# 对 image_sum 进行归一化
df['normalized ink intensity'] = (df['image_sum'] - df['image_sum'].min()) / (df['image_sum'].max() - df['image_sum'].min())
df.rename(columns={'ratings': 'human rating'}, inplace=True)

# 获取 content 的唯一值，并按升序排序
unique_content = sorted(df['content'].unique())

# 确定子图的行数和列数（包含新增的一个子图）
num_plots = len(unique_content) + 1
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols

# 创建子图布局
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

# 确保 axes 是二维数组（即使只有一行或一列）
axes = axes.reshape(num_rows, num_cols)

# 使用更专业的配色方案，这里选用更柔和、适合学术的配色
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

# 设置全局字体为更正式的学术字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 定义标签对应关系
label_mapping = {
    1: 'other',
    2: 'object',
    3: 'animal',
    4: 'plant',
    5: 'human'
}

# 遍历每个 content 值并绘制散点图和计算 SRCC
for i, content_value in enumerate(unique_content):
    row = i // num_cols
    col = i % num_cols
    subset = df[df['content'] == content_value]
    # 设置较小的点大小（s 参数）并使用不同颜色，同时调整点的透明度以减少视觉重叠
    scatter = sns.scatterplot(x='normalized ink intensity', y='human rating', data=subset, ax=axes[row, col], s=10, color=colors[i], alpha=0.6)
    axes[row, col].set_title(f'Content = {label_mapping[content_value]}', fontweight='bold')
    axes[row, col].set_xlabel('Normalized Ink Intensity', fontsize=10)
    axes[row, col].set_ylabel('Human Rating', fontsize=10)
    # 添加网格线，调整网格线颜色和样式使其更柔和
    axes[row, col].grid(True, linestyle='--', alpha=0.7, color='gray')
    # 设置横轴范围为 0 到 1
    axes[row, col].set_xlim(0, 1)
    # 设置纵轴范围为 0 到 1
    axes[row, col].set_ylim(0, 1)
    # 计算 SRCC
    srcc, p_value = spearmanr(subset['normalized ink intensity'], subset['human rating'])
    print(f"Content 为 {content_value} 时，归一化后的 image_sum 与 ratings 的斯皮尔曼等级相关系数（SRCC）为: {srcc:.2f}，p 值为: {p_value:.2e}")

# 新增的子图：不限制 content，绘制归一化后的 image_sum 和 rating 的散点图
row = (len(unique_content)) // num_cols
col = (len(unique_content)) % num_cols
# 设置较小的点大小（s 参数）并使用不同颜色，同时调整点的透明度以减少视觉重叠
scatter_all = sns.scatterplot(x='normalized ink intensity', y='human rating', data=df, ax=axes[row, col], s=10, color=colors[-1], alpha=0.6)
axes[row, col].set_title('Combined Content', fontweight='bold')
axes[row, col].set_xlabel('Normalized Ink Intensity', fontsize=10)
axes[row, col].set_ylabel('Human Rating', fontsize=10)
# 添加网格线，调整网格线颜色和样式使其更柔和
axes[row, col].grid(True, linestyle='--', alpha=0.7, color='gray')
# 设置横轴范围为 0 到 1
axes[row, col].set_xlim(0, 1)
# 设置纵轴范围为 0 到 1
axes[row, col].set_ylim(0, 1)

# 计算不限制 content 时归一化后的 image_sum 和 rating 的 SRCC
srcc_all, p_value_all = spearmanr(df['normalized ink intensity'], df['human rating'])
print(f"不限制 content 时，归一化后的 image_sum 与 ratings 的斯皮尔曼等级相关系数（SRCC）为: {srcc_all:.2f}，p 值为: {p_value_all:.2e}")

# 如果有多余的子图，隐藏它们
for i in range(num_plots, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].axis('off')

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# 显示图形
plt.show()