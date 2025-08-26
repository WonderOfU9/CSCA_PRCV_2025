import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 4
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.3


df = pd.read_csv('Merged_test_output.csv')


df['image_sum_bin'] = pd.qcut(df['image_sum'], q=5)

label_mapping = {
    1: 'other',
    2: 'object',
    3: 'animal',
    4: 'plant',
    5: 'human'
}


df['content'] = df['content'].map(label_mapping)


grouped = df.groupby(['image_sum_bin', 'content'])['ratings'].mean().unstack()


desired_order = ['other', 'object', 'animal', 'plant', 'human']
grouped = grouped.reindex(desired_order, axis=1)


colors = ['#1f77b4', '#ff9f40', '#2ca02c', '#e74c3c', '#9b59b6']
line_styles = ['-', '--', '-.', ':', '-']

markers = ['o', 's', '^', 'D', 'v']


fig, ax = plt.subplots(figsize=(6, 4))


for i, col in enumerate(grouped.columns):
    ax.plot(range(len(grouped)), grouped[col],
            color=colors[i],
            linestyle=line_styles[i],
            marker=markers[i],
            label=col,
            linewidth=1.5,
            markersize=4,
            markerfacecolor=colors[i],
            markeredgewidth=0.8)


ax.set_xlabel('Ink Density Bins')
ax.set_ylabel('Average Human Rating')
ax.set_ylim(0, 1)

ax.set_xticks(range(len(grouped)))
ax.set_xticklabels([f"Bin {i + 1}" for i in range(len(grouped))], rotation=0)


ax.grid(True, linestyle='--', alpha=0.3)

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles=handles,
    labels=[label.capitalize() for label in labels],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(grouped.columns),
    frameon=False,
    handlelength=3,
    handleheight=0.5,
    columnspacing=1.5,
    labelspacing=0.3,
    handletextpad=0.5,
    title="Content",
    title_fontsize=9,
    fontsize=8
)

plt.tight_layout()
plt.show()
