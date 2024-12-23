import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb

df = pd.read_csv('summary_stats.csv')

tetneg_m1_m8 = df[(df['cells'] == 'CD8') & (df['mouse'].isin(['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']))]
tetpos_m1_m8 = df[(df['cells'] == 'TETpos') & (df['mouse'].isin(['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']))]
tetpos_m1_m14 = df[df['cells'] == 'TETpos']

tetneg_m1_m8 = tetneg_m1_m8.copy()
tetpos_m1_m8 = tetpos_m1_m8.copy()
tetpos_m1_m14 = tetpos_m1_m14.copy()

tetneg_m1_m8['category'] = 'TETneg-M1-M8'
tetpos_m1_m8['category'] = 'TETpos-M1-M8'
tetpos_m1_m14['category'] = 'TETpos-M1-M14'


combined_df = pd.concat([tetneg_m1_m8, tetpos_m1_m8, tetpos_m1_m14])


title_fontsize = 14
axis_title_fontsize = 14
axis_label_fontsize = 12
title_pad = 20
axis_title_pad = 15

flierprops = dict(marker='x', markerfacecolor='black', markeredgecolor='black', markersize=7, alpha=0.7)

def format_ticks(value, pos):
    if value >= 1000:
        return f'{int(round(value, -2)):,}'  
    elif value >= 100:
        return f'{int(value)}'  
    else:
        return f'{round(value, 2)}'  

def label_outliers(data, var, category_col='category', label_col='mouse'):
    for category in data[category_col].unique():
        subset = data[data[category_col] == category]
        q1 = subset[var].quantile(0.25)
        q3 = subset[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = subset[(subset[var] < lower_bound) | (subset[var] > upper_bound)]
        x_position = data[category_col].unique().tolist().index(category) 
        for _, outlier in outliers.iterrows():
            plt.text(
                x_position + 0.04, 
                outlier[var] + (iqr * 0.02),
                outlier[label_col],
                color='black', ha='left', va='bottom', fontsize=10
            )


dull_palette = ["#4c72b0", "#dd8452", "#55a868"]

sns.set_palette(dull_palette)

def lighten_color(color, amount=0.5):
    white = np.array([1, 1, 1])
    color = np.array(to_rgb(color))
    return (1 - amount) * color + amount * white

lighter_palette = [lighten_color(color, 0.3) for color in dull_palette] 

sns.set_palette(lighter_palette)

# Plot 1: Unique CDR3 Count
plt.figure(figsize=(8, 6), dpi=300)
sns.boxplot(data=combined_df, x='category', y='uniqueCDR3', showfliers=True, flierprops=flierprops, palette=lighter_palette)
plt.ylabel('Unique CDR3 Count', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.xlabel('Datasets', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.ylim(0, 9000)
y_ticks = np.arange(0, 9001, 1000)
plt.yticks(y_ticks)
plt.xticks(fontsize=axis_label_fontsize)
plt.yticks(fontsize=axis_label_fontsize)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
label_outliers(combined_df, 'uniqueCDR3') 
plt.show()

# Plot 2: Total CDR3 Count
plt.figure(figsize=(8, 6), dpi=300)
sns.boxplot(data=combined_df, x='category', y='total_CDR3', showfliers=True, flierprops=flierprops, palette=lighter_palette)
plt.ylabel('Total CDR3 Count', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.xlabel('Datasets', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.ylim(0, 70000)
y_ticks = np.arange(0, 70001, 10000)
plt.yticks(y_ticks)
plt.xticks(fontsize=axis_label_fontsize)
plt.yticks(fontsize=axis_label_fontsize)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
label_outliers(combined_df, 'total_CDR3') 
plt.show()

# Plot 3: Richness
plt.figure(figsize=(8, 6), dpi=300)
sns.boxplot(data=combined_df, x='category', y='richness', showfliers=True, flierprops=flierprops, palette=lighter_palette)
plt.ylabel('Richness', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.xlabel('Datasets', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.ylim(0, 0.5)
y_ticks = np.arange(0, 0.51, 0.1)
plt.yticks(y_ticks)
plt.xticks(fontsize=axis_label_fontsize)
plt.yticks(fontsize=axis_label_fontsize)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
label_outliers(combined_df, 'richness') 
plt.show()

# Plot 4: Diversity 
plt.figure(figsize=(8, 6), dpi=300)
sns.boxplot(data=combined_df, x='category', y='diversity', showfliers=True, flierprops=flierprops, palette=lighter_palette)
plt.ylabel("Pielou's evenness index", fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.xlabel('Datasets', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.ylim(0.3, 1.0)
y_ticks = np.arange(0.3, 1.1, 0.1)
plt.yticks(y_ticks)
plt.xticks(fontsize=axis_label_fontsize)
plt.yticks(fontsize=axis_label_fontsize)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
label_outliers(combined_df, 'diversity')
plt.show()

# Plot 5: Simpson's Diversity Index 
plt.figure(figsize=(8, 6), dpi=300)
sns.boxplot(data=combined_df, x='category', y='simpsons', showfliers=True, flierprops=flierprops, palette=lighter_palette)
plt.ylabel("Simpson's diversity index", fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.xlabel('Datasets', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.ylim(-0.005, 0.45) 
y_ticks = np.arange(0, 0.46, 0.05)
plt.yticks(y_ticks)
plt.xticks(fontsize=axis_label_fontsize)
plt.yticks(fontsize=axis_label_fontsize)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
label_outliers(combined_df, 'simpsons')
plt.show()

# Plot 6: DE50 
plt.figure(figsize=(8, 6), dpi=300)
sns.boxplot(data=combined_df, x='category', y='DE50', showfliers=True, flierprops=flierprops, palette=lighter_palette)
plt.ylabel('DE50', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.xlabel('Datasets', fontsize=axis_title_fontsize, labelpad=axis_title_pad)
plt.ylim(0, 26)
y_ticks = np.arange(0, 26.1, 2)
plt.yticks(y_ticks)
plt.xticks(fontsize=axis_label_fontsize)
plt.yticks(fontsize=axis_label_fontsize)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
label_outliers(combined_df, 'DE50') 
plt.show()