import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd


plt.rcParams['figure.dpi'] = 100


plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']


df = pd.read_csv('/Merged_test_output.csv')


df['normalized ink intensity'] = (df['image_sum'] - df['image_sum'].min()) / (df['image_sum'].max() - df['image_sum'].min())
df.rename(columns={'ratings': 'human rating'}, inplace=True)


unique_content = sorted(df['content'].unique())


num_plots = len(unique_content) + 1
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols


fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))


axes = axes.reshape(num_rows, num_cols)


colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


label_mapping = {
    1: 'other',
    2: 'object',
    3: 'animal',
    4: 'plant',
    5: 'human'
}


for i, content_value in enumerate(unique_content):
    row = i // num_cols
    col = i % num_cols
    subset = df[df['content'] == content_value]

    scatter = sns.scatterplot(x='normalized ink intensity', y='human rating', data=subset, ax=axes[row, col], s=10, color=colors[i], alpha=0.6)
    axes[row, col].set_title(f'Content = {label_mapping[content_value]}', fontweight='bold')
    axes[row, col].set_xlabel('Normalized Ink Intensity', fontsize=10)
    axes[row, col].set_ylabel('Human Rating', fontsize=10)

    axes[row, col].grid(True, linestyle='--', alpha=0.7, color='gray')

    axes[row, col].set_xlim(0, 1)

    axes[row, col].set_ylim(0, 1)

    srcc, p_value = spearmanr(subset['normalized ink intensity'], subset['human rating'])
    print(f"When content is {content_value} ,the SRCC between normalized image_sum and ratings is: {srcc:.2f}, p-value is : {p_value:.2e}")


row = (len(unique_content)) // num_cols
col = (len(unique_content)) % num_cols

scatter_all = sns.scatterplot(x='normalized ink intensity', y='human rating', data=df, ax=axes[row, col], s=10, color=colors[-1], alpha=0.6)
axes[row, col].set_title('Combined Content', fontweight='bold')
axes[row, col].set_xlabel('Normalized Ink Intensity', fontsize=10)
axes[row, col].set_ylabel('Human Rating', fontsize=10)

axes[row, col].grid(True, linestyle='--', alpha=0.7, color='gray')

axes[row, col].set_xlim(0, 1)

axes[row, col].set_ylim(0, 1)


srcc_all, p_value_all = spearmanr(df['normalized ink intensity'], df['human rating'])
print(f"The SRCC between Normalized image_sum and ratings is : {srcc_all:.2f},p-value is : {p_value_all:.2e}")

for i in range(num_plots, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].axis('off')


plt.subplots_adjust(wspace=0.3, hspace=0.4)


plt.show()