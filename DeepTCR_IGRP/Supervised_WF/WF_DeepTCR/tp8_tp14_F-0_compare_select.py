# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:40:29 2024

@author: Mitch
"""

import seaborn as sns
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon, pearsonr, shapiro

# ========== COMPARISON OF TP8 AND TP14 STATISTICS USING F-0.5 ===========================

# Define the file paths for the two groups
tp8_metrics_folder = './tp8_metrics/'
tp14_metrics_folder = './tp14_metrics/'

# Thresholds for the CSV files
thresholds = [50, 60, 70, 80, 90]

# Function to load CSV files into a dictionary for a given folder
def load_metrics(folder, thresholds, group):
    metrics_dict = {}
    for threshold in thresholds:
        file_path = os.path.join(folder, f'tp{group}_metrics_{threshold}.csv')
        metrics_dict[threshold] = pd.read_csv(file_path)
    return metrics_dict

def process_metrics(folder, thresholds, group):
    processed_dataframes = {}  # To store processed dataframes by threshold
    
    for threshold in thresholds:
        # Load the CSV file
        file_path = os.path.join(folder, f'tp{group}_metrics_{threshold}.csv')
        df = pd.read_csv(file_path)
        
        # Rename any columns containing '(Kas_Gearty_no_cut)' or with inconsistent names
        df = df.rename(columns=lambda x: x.replace('(Kas_Gearty_no_cut)', '').strip())
        
        # Change any column named 'f1_score' to 'F1 score'
        if 'f1_score' in df.columns:
            df = df.rename(columns={'f1_score': 'F1 score'})
        
        # Ensure that the precision column is labeled 'precision (PPV)'
        if 'precision' in df.columns and 'precision (PPV)' not in df.columns:
            df = df.rename(columns={'precision': 'precision (PPV)'})
        
        # Handle recall column: either 'recall' or 'recall (TPR)'
        recall_col = None
        if 'recall' in df.columns:
            recall_col = 'recall'
        elif 'recall (TPR)' in df.columns:
            recall_col = 'recall (TPR)'
        else:
            print(f"Warning: No 'recall' or 'recall (TPR)' column found in {file_path}, skipping F0.5 calculation.")
            continue
        
        # Calculate the F-0.5 score if precision and recall columns exist
        if 'precision (PPV)' in df.columns and recall_col:
            df['F0.5_score'] = (1 + 0.5**2) * (df['precision (PPV)'] * df[recall_col]) / ((0.5**2 * df['precision (PPV)']) + df[recall_col])
            print(f"F0.5 score added to {file_path}")
        
        # Calculate the F-0.75 score
        if 'precision (PPV)' in df.columns and recall_col:
            df['F0.75_score'] = (1 + 0.75**2) * (df['precision (PPV)'] * df[recall_col]) / ((0.75**2 * df['precision (PPV)']) + df[recall_col])
            print(f"F0.75 score added to {file_path}")
        
        # Ensure the F0.5_score and F0.75_score are placed correctly between 'F1 score' and 'b_accuracy'
        if 'F1 score' in df.columns and 'b_accuracy' in df.columns:
            columns = df.columns.tolist()
            f1_index = columns.index('F1 score') 
            
            # Remove F0.5_score and F0.75_score if they already exist, then insert them in the correct position
            if 'F0.5_score' in columns:
                columns.remove('F0.5_score')
            if 'F0.75_score' in columns:
                columns.remove('F0.75_score')
            
            columns.insert(f1_index + 1, 'F0.5_score')  
            columns.insert(f1_index + 2, 'F0.75_score')  
            
            # Apply the new column order
            df = df[columns]
        
        # Save the modified DataFrame back to the CSV with the same name
        df.to_csv(file_path, index=False)
        print(f'Updated {file_path} saved.')
        
        # Store the processed DataFrame in the dictionary
        processed_dataframes[threshold] = df
    
    return processed_dataframes

# Process tp8 metrics and store DataFrames in a variable
tp8_processed = process_metrics(tp8_metrics_folder, thresholds, '8')

# Process tp14 metrics and store DataFrames in a variable
tp14_processed = process_metrics(tp14_metrics_folder, thresholds, '14')

# Function to calculate mean F-0.5 score for each model across thresholds
def calculate_mean_f05(processed_dataframes):
    # Concatenate all thresholds' dataframes into a single dataframe
    consolidated_df = pd.concat(processed_dataframes.values(), ignore_index=True)
    # Calculate the mean F-0.5 score grouped by 'model_name'
    return consolidated_df.groupby('model_name')[['F0.5_score']].mean()

# Use the processed dataframes from process_metrics function
tp8_mean_f05 = calculate_mean_f05(tp8_processed)
tp14_mean_f05 = calculate_mean_f05(tp14_processed)

# Function to calculate mean F-0.75 score for each model across thresholds
def calculate_mean_f075(processed_dataframes):
    # Concatenate all thresholds' dataframes into a single dataframe
    consolidated_df_75 = pd.concat(processed_dataframes.values(), ignore_index=True)
    # Calculate the mean F-0.5 score grouped by 'model_name'
    return consolidated_df_75.groupby('model_name')[['F0.75_score']].mean()

# Use the processed dataframes from process_metrics function
tp8_mean_f075 = calculate_mean_f075(tp8_processed)


# Function to strip unwanted suffixes (_dist and _add_dist) from 'model_name'
def remove_suffix(df, suffix):
    df.index = df.index.str.replace(suffix, '', regex=False)
    return df

# Strip suffixes from 'model_name'
tp8_mean_f05 = remove_suffix(tp8_mean_f05, '_dist')
tp14_mean_f05 = remove_suffix(tp14_mean_f05, '_add_dist')

# Rename the F0.5_score columns to include _tp8 and _tp14
tp8_mean_f05.rename(columns={'F0.5_score': 'F0.5_score_tp8'}, inplace=True)
tp14_mean_f05.rename(columns={'F0.5_score': 'F0.5_score_tp14'}, inplace=True)

# Rename the F0.75_score columns to include _tp8 and _tp14

# Merge the two dataframes on 'model_name'
combined_mean_f05 = pd.merge(tp8_mean_f05, tp14_mean_f05, left_index=True, right_index=True)




# Now the 'combined_mean_f05' dataframe contains both F0.5_score_tp8 and F0.5_score_tp14
print(combined_mean_f05.head())

# =================== STATISTICAL TESTS TP8 vS TP14 PAIRED ======================

# ===================== F.050 =====================
# Normality check using Shapiro-Wilk Test
stat_tp8, p_tp8 = shapiro(tp8_mean_f05['F0.5_score_tp8'])
stat_tp14, p_tp14 = shapiro(tp14_mean_f05['F0.5_score_tp14'])

print(f"Shapiro-Wilk Test for tp8 F-0.5: stat={stat_tp8}, p-value={p_tp8}")
print(f"Shapiro-Wilk Test for tp14 F-0.5: stat={stat_tp14}, p-value={p_tp14}")

# Depending on normality, choose between paired t-test or Wilcoxon signed-rank test, extract data prior
f05_score_tp8 = tp8_mean_f05['F0.5_score_tp8']
f05_score_tp14 = tp14_mean_f05['F0.5_score_tp14']

if p_tp8 > 0.05 and p_tp14 > 0.05:
    # Perform paired t-test if normally distributed
    t_stat_f05, p_value_f05 = stats.ttest_rel(f05_score_tp8, f05_score_tp14)
    print(f"Paired t-test for F-0.5 Score: t-statistic = {t_stat_f05}, p-value = {p_value_f05}")
else:
    # Use Wilcoxon signed-rank test for non-normal data
    w_stat_f05, p_value_wilcoxon_f05 = wilcoxon(f05_score_tp8, f05_score_tp14)
    print(f"Wilcoxon Signed-Rank Test for F-0.5 Score: stat = {w_stat_f05}, p-value = {p_value_wilcoxon_f05}")

# Pearson correlation analysis between F-0.5 scores for tp8 and tp14
corr_f05, _ = pearsonr(f05_score_tp8, f05_score_tp14)
print(f"Pearson correlation for F-0.5 Score between tp8 and tp14: {corr_f05}")

# Reset the index to turn the model_name into a column
combined_mean_f05 = combined_mean_f05.reset_index()

# Now you can assign the model type based on the model_name column
combined_mean_f05['model_type'] = combined_mean_f05['model_name'].apply(
    lambda x: 'Monte Carlo' if x.endswith('_Mc') else 'K-fold' if x.endswith('_Kf') else 'Unknown'
)



# ============ BOXPLOTS FOR DISTRIBUTION =================

# ================== F0.50 =========================================
# Separate tp8 and tp14 F0.5 scores from combined_mean_f05
df_tp8 = combined_mean_f05[['F0.5_score_tp8', 'model_type']].rename(columns={'F0.5_score_tp8': 'F0.5_score'})
df_tp8['Group'] = 'tp8'

df_tp14 = combined_mean_f05[['F0.5_score_tp14', 'model_type']].rename(columns={'F0.5_score_tp14': 'F0.5_score'})
df_tp14['Group'] = 'tp14'

# Combine tp8 and tp14 data back into one dataframe
f05_combined_df = pd.concat([df_tp8, df_tp14], ignore_index=True)

# Create 'Group_numeric' for the combined dataframe
f05_combined_df['Group_numeric'] = f05_combined_df['Group'].map({'tp8': 0, 'tp14': 1})

# Identify outliers using IQR method for tp8
Q1_tp8 = f05_combined_df[f05_combined_df['Group'] == 'tp8']['F0.5_score'].quantile(0.25)
Q3_tp8 = f05_combined_df[f05_combined_df['Group'] == 'tp8']['F0.5_score'].quantile(0.75)
IQR_tp8 = Q3_tp8 - Q1_tp8

# Identify outliers using IQR method for tp14
Q1_tp14 = f05_combined_df[f05_combined_df['Group'] == 'tp14']['F0.5_score'].quantile(0.25)
Q3_tp14 = f05_combined_df[f05_combined_df['Group'] == 'tp14']['F0.5_score'].quantile(0.75)
IQR_tp14 = Q3_tp14 - Q1_tp14

# Filter non-outliers for tp8 and tp14 separately
non_outliers_tp8 = f05_combined_df[(f05_combined_df['Group'] == 'tp8') & 
                                   (f05_combined_df['F0.5_score'] >= (Q1_tp8 - 1.5 * IQR_tp8)) & 
                                   (f05_combined_df['F0.5_score'] <= (Q3_tp8 + 1.5 * IQR_tp8))]

non_outliers_tp14 = f05_combined_df[(f05_combined_df['Group'] == 'tp14') & 
                                    (f05_combined_df['F0.5_score'] >= (Q1_tp14 - 1.5 * IQR_tp14)) & 
                                    (f05_combined_df['F0.5_score'] <= (Q3_tp14 + 1.5 * IQR_tp14))]

# Combine non-outliers
non_outliers_df = pd.concat([non_outliers_tp8, non_outliers_tp14])

# Filter outliers for K-fold and Monte Carlo separately for tp8 and tp14
# Outliers for K-fold in tp8
outliers_kfold_tp8 = f05_combined_df[(f05_combined_df['model_type'] == 'K-fold') & 
                                     (f05_combined_df['Group'] == 'tp8') & 
                                     ((f05_combined_df['F0.5_score'] < (Q1_tp8 - 1.5 * IQR_tp8)) |
                                      (f05_combined_df['F0.5_score'] > (Q3_tp8 + 1.5 * IQR_tp8)))]

# Outliers for K-fold in tp14
outliers_kfold_tp14 = f05_combined_df[(f05_combined_df['model_type'] == 'K-fold') & 
                                      (f05_combined_df['Group'] == 'tp14') & 
                                      ((f05_combined_df['F0.5_score'] < (Q1_tp14 - 1.5 * IQR_tp14)) |
                                       (f05_combined_df['F0.5_score'] > (Q3_tp14 + 1.5 * IQR_tp14)))]

# Outliers for Monte Carlo in tp8
outliers_mc_tp8 = f05_combined_df[(f05_combined_df['model_type'] == 'Monte Carlo') & 
                                  (f05_combined_df['Group'] == 'tp8') & 
                                  ((f05_combined_df['F0.5_score'] < (Q1_tp8 - 1.5 * IQR_tp8)) |
                                   (f05_combined_df['F0.5_score'] > (Q3_tp8 + 1.5 * IQR_tp8)))]

# Outliers for Monte Carlo in tp14
outliers_mc_tp14 = f05_combined_df[(f05_combined_df['model_type'] == 'Monte Carlo') & 
                                   (f05_combined_df['Group'] == 'tp14') & 
                                   ((f05_combined_df['F0.5_score'] < (Q1_tp14 - 1.5 * IQR_tp14)) |
                                    (f05_combined_df['F0.5_score'] > (Q3_tp14 + 1.5 * IQR_tp14)))]



# Plot the boxplot without the default outliers
plt.figure(figsize=(10, 7), dpi=600)
sns.boxplot(x='Group_numeric', y='F0.5_score', data=f05_combined_df, 
            palette=['lightblue', 'lightgray'], showfliers=False)

# Add jittered scatter plot for non-outliers, colored by model type
sns.stripplot(x='Group_numeric', y='F0.5_score', data=non_outliers_df, 
              hue='model_type', jitter=0.2, size=3.5, palette={'K-fold': 'red', 'Monte Carlo': 'blue'},
              marker='o', linewidth=0.4, edgecolor='black', alpha=0.6)

# Define jitter strength for the outliers
jitter_strength = 0.15

# Jitter the outliers for K-fold in tp8
x_jitter_kfold_tp8 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_kfold_tp8))
plt.scatter(outliers_kfold_tp8['Group_numeric'] + x_jitter_kfold_tp8, 
            outliers_kfold_tp8['F0.5_score'], color='red', s=35, marker='x', label='K-fold outliers (tp8)')

# Jitter the outliers for K-fold in tp14
x_jitter_kfold_tp14 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_kfold_tp14))
plt.scatter(outliers_kfold_tp14['Group_numeric'] + x_jitter_kfold_tp14, 
            outliers_kfold_tp14['F0.5_score'], color='red', s=35, marker='x', label='K-fold outliers (tp14)')

# Jitter the outliers for Monte Carlo in tp8
x_jitter_mc_tp8 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_mc_tp8))
plt.scatter(outliers_mc_tp8['Group_numeric'] + x_jitter_mc_tp8, 
            outliers_mc_tp8['F0.5_score'], color='blue', s=35, marker='x', label='Monte Carlo outliers (tp8)')

# Jitter the outliers for Monte Carlo in tp14
x_jitter_mc_tp14 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_mc_tp14))
plt.scatter(outliers_mc_tp14['Group_numeric'] + x_jitter_mc_tp14, 
            outliers_mc_tp14['F0.5_score'], color='blue', s=35, marker='x', label='Monte Carlo outliers (tp14)')

# Add means for each group
mean_tp8 = f05_combined_df[f05_combined_df['Group'] == 'tp8']['F0.5_score'].mean()
mean_tp14 = f05_combined_df[f05_combined_df['Group'] == 'tp14']['F0.5_score'].mean()

plt.scatter([0], [mean_tp8], color='limegreen', s=100, marker='+', edgecolor='black', linewidths=1.5, label='Mean (tp8)', zorder=10)
plt.scatter([1], [mean_tp14], color='limegreen', s=100, marker='+', edgecolor='black', linewidths=1.5, label='Mean (tp14)', zorder=10)

# Set Y-axis limits manually for better scaling (adjust as needed based on your data range)
plt.ylim(0.60, 0.85)

# Customize X-axis to show 'tp8' and 'tp14'
plt.xticks([0, 1], ['tp8', 'tp14'])

# Add light horizontal gridlines
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# Customize plot labels, title, and legend placement
plt.ylabel('Threshold Averaged F-0.5 Score', fontsize=13, labelpad=10)
plt.xlabel('Model Training Dataset', fontsize=13)
plt.title('Threshold Averaged F-0.5 Scores for tp8 and tp14 Models', fontsize=15, pad=10)
plt.tight_layout(1.0)
plt.subplots_adjust(top=0.85)
plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12) 

# Manually create legend entries for K-fold and Monte Carlo with transparency and black edgecolor
kfold_handle = plt.Line2D([0], [0], marker='o', color='red', label='K-fold', markersize=8, 
                          linestyle='None', markerfacecolor='red', markeredgewidth=0.8, 
                          markeredgecolor='black', alpha=0.6)
mc_handle = plt.Line2D([0], [0], marker='o', color='blue', label='Monte Carlo', markersize=8, 
                       linestyle='None', markerfacecolor='blue', markeredgewidth=0.8, 
                       markeredgecolor='black', alpha=0.6)

# Create outlier and mean handles for the legend
outliers_kfold_handle = plt.Line2D([0], [0], marker='x', color='red', label='K-fold outliers', markersize=8, 
                                   linestyle='None')
outliers_mc_handle = plt.Line2D([0], [0], marker='x', color='blue', label='Monte Carlo outliers', markersize=8, 
                                linestyle='None')

# Add means for each group with consistent size and marker style
mean_tp8_handle = plt.scatter([0], [mean_tp8], color='limegreen', s=100, marker='+', 
                              edgecolor='black', linewidths=1.5, label='Mean (tp8)', zorder=10)

mean_tp14_handle = plt.scatter([1], [mean_tp14], color='limegreen', s=100, marker='+', 
                               edgecolor='black', linewidths=1.5, label='Mean (tp14)', zorder=10)

# Customize legend placement (outside the plot to the right)
legend = plt.legend(handles=[kfold_handle, mc_handle, outliers_kfold_handle, outliers_mc_handle, 
                    mean_tp8_handle, mean_tp14_handle], 
           loc='center left', bbox_to_anchor=(1, 0.5), title='Legend', fontsize=10)

# Darken the legend border
legend.get_frame().set_edgecolor('black')  
legend.get_frame().set_linewidth(0.5)  

# Display the plot with adjusted layout
plt.tight_layout()
plt.show()


# ==================== SCATTERPLOT FOR PEARSON CORRELATION =====================

# Separate the data into two dataframes based on the 'Group' column
df_tp8 = f05_combined_df[f05_combined_df['Group'] == 'tp8'][['model_type', 'F0.5_score']]
df_tp14 = f05_combined_df[f05_combined_df['Group'] == 'tp14'][['model_type', 'F0.5_score']]

# Reset indices for tp8 and tp14 to align the rows
df_tp8.reset_index(drop=True, inplace=True)
df_tp14.reset_index(drop=True, inplace=True)

# Rename columns for clarity
df_tp8.rename(columns={'F0.5_score': 'F0.5_score_tp8'}, inplace=True)
df_tp14.rename(columns={'F0.5_score': 'F0.5_score_tp14'}, inplace=True)

# Concatenate the two dataframes column-wise (axis=1)
merged_df = pd.concat([df_tp8, df_tp14['F0.5_score_tp14']], axis=1)

# Adjust the color palette: switch K-fold to red and Monte Carlo to blue
color_palette = {'K-fold': 'red', 'Monte Carlo': 'blue'}

# Plot the scatter plot with colors based on model_type
plt.figure(figsize=(10, 7), dpi=600)
scatter_plot = sns.scatterplot(x='F0.5_score_tp8', y='F0.5_score_tp14', hue='model_type', data=merged_df,
                               palette=color_palette, s=35, edgecolor='black', alpha=0.6, zorder=3)

# Add a regression line (without zorder)
sns.regplot(x='F0.5_score_tp8', y='F0.5_score_tp14', data=merged_df, scatter=False, 
            color='red', line_kws={'linewidth': 1.5}, ci=95)

# Calculate and display Pearson's correlation coefficient
pearson_r, _ = pearsonr(merged_df['F0.5_score_tp8'], merged_df['F0.5_score_tp14'])
plt.text(0.05, 0.95, f'Pearson r = {pearson_r:.2f}', fontsize=12, color='red', ha='left', va='top', transform=plt.gca().transAxes)

# Adjust X-axis range to fit all points
plt.xlim(0.55, 0.85)
plt.ylim(0.55, 0.85)
plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12) 

# Add labels and title
plt.title('Threshold Averaged F-0.5 Scores of tp8 and tp14 Paired Models', fontsize=15, pad=10)
plt.xlabel('Threshold Averaged F-0.5 Score (tp8)', fontsize=13, labelpad=15)
plt.ylabel('Threshold Averaged F-0.5 Score (tp14)', fontsize=13, labelpad=15)

# Add grid lines with a lower zorder to place them behind the dots
plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)

# Manually adjust legend to match the transparency, size, and edge color
handles, labels = scatter_plot.get_legend_handles_labels()
for handle in handles:
    handle.set_alpha(0.6)  
    handle.set_edgecolor('black') 
    handle.set_linewidth(0.5)  

# Customize legend placement (bottom right)
legend = plt.legend(handles=handles, labels=labels, loc='lower right', title='Cross-validation')
legend.get_frame().set_edgecolor('black')  
legend.get_frame().set_linewidth(0.5) 

# Show the plot
plt.tight_layout()
plt.show()

# ================== Bland-Altman Plot with F-0.5 Score ====================

# Bland-Altman Plot for F-0.5 Score
plt.figure(figsize=(6, 5), dpi=600)
mean_f05 = (combined_mean_f05['F0.5_score_tp8'] + combined_mean_f05['F0.5_score_tp14']) / 2
diff_f05 = combined_mean_f05['F0.5_score_tp8'] - combined_mean_f05['F0.5_score_tp14']

# Color the dots by model type (Monte Carlo or K-fold)
colors = {'Monte Carlo': 'blue', 'K-fold': 'red'}
for model_type in ['Monte Carlo', 'K-fold']:
    subset = combined_mean_f05[combined_mean_f05['model_type'] == model_type]
    plt.scatter(subset['F0.5_score_tp8'], subset['F0.5_score_tp14'] - subset['F0.5_score_tp8'], 
                color=colors[model_type], label=model_type, s=10, alpha=0.6, edgecolors='black', linewidth=0.4)

# Add horizontal lines for the mean difference and limits
plt.axhline(np.mean(diff_f05), alpha=0.5, color='gray', linestyle='--', label='Mean Difference')
plt.axhline(np.mean(diff_f05) + 1.96 * np.std(diff_f05), alpha=0.5, color='limegreen', linestyle=':', label='Upper Limit')
plt.axhline(np.mean(diff_f05) - 1.96 * np.std(diff_f05), alpha=0.5, color='red', linestyle=':', label='Lower Limit')

# Labels and title
plt.xlabel('Average of Mean F-0.5 for tp8 tp14 Model Pairs', fontsize=8)
plt.ylabel('Difference in Mean F-0.5 Score of Model Pairs (tp8 - tp14)', fontsize=8)
plt.title('Mean F-0.5 Difference Plot for tp8 tp14 Model Pairs', fontsize=9)

# Ensure even X and Y axis limits
plt.xlim([0.5, 0.9])  
plt.ylim([-0.15, 0.15])  

# Legend
legend = plt.legend(title="Legend", title_fontsize=7, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, 0.5))
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.5)

plt.tick_params(axis='x', labelsize=6) 
plt.tick_params(axis='y', labelsize=6)  

plt.tight_layout(pad=2.0)
plt.show()

# Save results with descriptions, equations, and interpretations
test_descriptions = [
    "Paired t-test: Compares the means of two paired datasets (tp8 vs tp14) to assess whether there is a significant difference.",
    "Wilcoxon Signed-Rank Test: A non-parametric test comparing the medians of two paired datasets without assuming normal distribution.",
    "Pearson Correlation: Measures the linear relationship between tp8 and tp14 F-0.5 scores.",
    "Shapiro-Wilk Test: Tests for the normality of the data distribution."
]

test_equations = [
    "t = (mean difference) / (standard error of the difference)",
    "W = sum of the signed ranks of the differences between tp8 and tp14",
    "r = covariance(tp8, tp14) / (std(tp8) * std(tp14))",
    "W = (observed statistic for normality)"
]

interpretation_guidelines = [
    "If the p-value is less than 0.05, the difference is statistically significant.",
    "If the p-value is less than 0.05, the difference is statistically significant.",
    "Correlation ranges from -1 (perfect negative) to 1 (perfect positive).",
    "If the p-value is less than 0.05, the data deviates significantly from a normal distribution."
]

# Prepare the results data
results_data = [
    ['Paired t-test (F-0.5)' if p_tp8 > 0.05 and p_tp14 > 0.05 else 'Wilcoxon signed-rank test (F-0.5)',
     t_stat_f05 if p_tp8 > 0.05 and p_tp14 > 0.05 else w_stat_f05, 
     p_value_f05 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f05, 
     'No' if (p_value_f05 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f05) >= 0.05 else 'Yes'],
    ['Pearson Correlation (F-0.5)', corr_f05, 'N/A', 'N/A'],
    ['Shapiro-Wilk tp8', stat_tp8, p_tp8, 'No' if p_tp8 >= 0.05 else 'Yes'],
    ['Shapiro-Wilk tp14', stat_tp14, p_tp14, 'No' if p_tp14 >= 0.05 else 'Yes']
]

# Interpretations for the results
interpretations = [
    f"{'t-stat' if p_tp8 > 0.05 and p_tp14 > 0.05 else 'w-stat'} = {t_stat_f05 if p_tp8 > 0.05 and p_tp14 > 0.05 else w_stat_f05}, "
    f"p-value = {p_value_f05 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f05}, "
    f"Interpretation: {'No significant difference' if (p_value_f05 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f05) >= 0.05 else 'Significant difference'}",
    f"Pearson r = {corr_f05}, Interpretation: {'Strong positive correlation' if corr_f05 > 0.7 else 'Weak or moderate correlation'}",
    f"Shapiro-Wilk stat tp8 = {stat_tp8}, p-value = {p_tp8}, Interpretation: {'Normally distributed' if p_tp8 > 0.05 else 'Not normally distributed'}",
    f"Shapiro-Wilk stat tp14 = {stat_tp14}, p-value = {p_tp14}, Interpretation: {'Normally distributed' if p_tp14 > 0.05 else 'Not normally distributed'}"
]

# Prepare the full DataFrame
results_df = pd.DataFrame({
    'Test Name': [row[0] for row in results_data],
    'Statistic': [row[1] for row in results_data],
    'p-value': [row[2] for row in results_data],
    'Significant (alpha = 0.05)': [row[3] for row in results_data],
    'Test Description': test_descriptions,
    'Equation': test_equations,
    'Interpretation': interpretations,
    'Interpretation Guidelines': interpretation_guidelines
})

# Save the DataFrame to a CSV file
output_file_path = './tp8_tp14_f05_full_statistical_analysis.csv'
results_df.to_csv(output_file_path, index=False)

print(f"Results saved to {output_file_path}")







# ===================== F0.75 =====================

tp14_mean_f075 = calculate_mean_f075(tp14_processed)
tp8_mean_f075 = remove_suffix(tp8_mean_f075, '_dist')
tp14_mean_f075 = remove_suffix(tp14_mean_f075, '_add_dist')
tp8_mean_f075.rename(columns={'F0.75_score': 'F0.75_score_tp8'}, inplace=True)
tp14_mean_f075.rename(columns={'F0.75_score': 'F0.75_score_tp14'}, inplace=True)
combined_mean_f075 = pd.merge(tp8_mean_f075, tp14_mean_f075, left_index=True, right_index=True)



# Normality check using Shapiro-Wilk Test
stat_tp8, p_tp8 = shapiro(tp8_mean_f075['F0.75_score_tp8'])
stat_tp14, p_tp14 = shapiro(tp14_mean_f075['F0.75_score_tp14'])

print(f"Shapiro-Wilk Test for tp8 F-0.75: stat={stat_tp8}, p-value={p_tp8}")
print(f"Shapiro-Wilk Test for tp14 F-0.75: stat={stat_tp14}, p-value={p_tp14}")

# Depending on normality, choose between paired t-test or Wilcoxon signed-rank test, extract data prior
f075_score_tp8 = tp8_mean_f075['F0.75_score_tp8']
f075_score_tp14 = tp14_mean_f075['F0.75_score_tp14']

if p_tp8 > 0.05 and p_tp14 > 0.05:
    # Perform paired t-test if normally distributed
    t_stat_f075, p_value_f075 = stats.ttest_rel(f075_score_tp8, f075_score_tp14)
    print(f"Paired t-test for F-0.75 Score: t-statistic = {t_stat_f075}, p-value = {p_value_f075}")
else:
    # Use Wilcoxon signed-rank test for non-normal data
    w_stat_f075, p_value_wilcoxon_f075 = wilcoxon(f075_score_tp8, f075_score_tp14)
    print(f"Wilcoxon Signed-Rank Test for F-0.75 Score: stat = {w_stat_f075}, p-value = {p_value_wilcoxon_f075}")

# Pearson correlation analysis between F-0.5 scores for tp8 and tp14
corr_f075, _ = pearsonr(f075_score_tp8, f075_score_tp14)
print(f"Pearson correlation for F-0.75 Score between tp8 and tp14: {corr_f075}")

# Reset the index to turn the model_name into a column
combined_mean_f075 = combined_mean_f075.reset_index()

# Now you can assign the model type based on the model_name column
combined_mean_f075['model_type'] = combined_mean_f075['model_name'].apply(
    lambda x: 'Monte Carlo' if x.endswith('_Mc') else 'K-fold' if x.endswith('_Kf') else 'Unknown'
)


# ================== F0.75 =========================================
# Separate tp8 and tp14 F0.75 scores from combined_mean_f075
df_tp8 = combined_mean_f075[['F0.75_score_tp8', 'model_type']].rename(columns={'F0.75_score_tp8': 'F0.75_score'})
df_tp8['Group'] = 'tp8'

df_tp14 = combined_mean_f075[['F0.75_score_tp14', 'model_type']].rename(columns={'F0.75_score_tp14': 'F0.75_score'})
df_tp14['Group'] = 'tp14'

# Combine tp8 and tp14 data back into one dataframe
f075_combined_df = pd.concat([df_tp8, df_tp14], ignore_index=True)

# Create 'Group_numeric' for the combined dataframe
f075_combined_df['Group_numeric'] = f075_combined_df['Group'].map({'tp8': 0, 'tp14': 1})

# Identify outliers using IQR method for tp8
Q1_tp8 = f075_combined_df[f075_combined_df['Group'] == 'tp8']['F0.75_score'].quantile(0.25)
Q3_tp8 = f075_combined_df[f075_combined_df['Group'] == 'tp8']['F0.75_score'].quantile(0.75)
IQR_tp8 = Q3_tp8 - Q1_tp8

# Identify outliers using IQR method for tp14
Q1_tp14 = f075_combined_df[f075_combined_df['Group'] == 'tp14']['F0.75_score'].quantile(0.25)
Q3_tp14 = f075_combined_df[f075_combined_df['Group'] == 'tp14']['F0.75_score'].quantile(0.75)
IQR_tp14 = Q3_tp14 - Q1_tp14

# Filter non-outliers for tp8 and tp14 separately
non_outliers_tp8 = f075_combined_df[(f075_combined_df['Group'] == 'tp8') & 
                                    (f075_combined_df['F0.75_score'] >= (Q1_tp8 - 1.5 * IQR_tp8)) & 
                                    (f075_combined_df['F0.75_score'] <= (Q3_tp8 + 1.5 * IQR_tp8))]

non_outliers_tp14 = f075_combined_df[(f075_combined_df['Group'] == 'tp14') & 
                                     (f075_combined_df['F0.75_score'] >= (Q1_tp14 - 1.5 * IQR_tp14)) & 
                                     (f075_combined_df['F0.75_score'] <= (Q3_tp14 + 1.5 * IQR_tp14))]

# Combine non-outliers after filtering them separately
non_outliers_df_75 = pd.concat([non_outliers_tp8, non_outliers_tp14])


# Outliers for K-fold in tp8
outliers_kfold_tp8 = f075_combined_df[(f075_combined_df['model_type'] == 'K-fold') & 
                                      (f075_combined_df['Group'] == 'tp8') & 
                                      ((f075_combined_df['F0.75_score'] < (Q1_tp8 - 1.5 * IQR_tp8)) |
                                       (f075_combined_df['F0.75_score'] > (Q3_tp8 + 1.5 * IQR_tp8)))]

# Outliers for K-fold in tp14
outliers_kfold_tp14 = f075_combined_df[(f075_combined_df['model_type'] == 'K-fold') & 
                                       (f075_combined_df['Group'] == 'tp14') & 
                                       ((f075_combined_df['F0.75_score'] < (Q1_tp14 - 1.5 * IQR_tp14)) |
                                        (f075_combined_df['F0.75_score'] > (Q3_tp14 + 1.5 * IQR_tp14)))]

# Outliers for Monte Carlo in tp8
outliers_mc_tp8 = f075_combined_df[(f075_combined_df['model_type'] == 'Monte Carlo') & 
                                   (f075_combined_df['Group'] == 'tp8') & 
                                   ((f075_combined_df['F0.75_score'] < (Q1_tp8 - 1.5 * IQR_tp8)) |
                                    (f075_combined_df['F0.75_score'] > (Q3_tp8 + 1.5 * IQR_tp8)))]

# Outliers for Monte Carlo in tp14
outliers_mc_tp14 = f075_combined_df[(f075_combined_df['model_type'] == 'Monte Carlo') & 
                                    (f075_combined_df['Group'] == 'tp14') & 
                                    ((f075_combined_df['F0.75_score'] < (Q1_tp14 - 1.5 * IQR_tp14)) |
                                     (f075_combined_df['F0.75_score'] > (Q3_tp14 + 1.5 * IQR_tp14)))]


# Plot the boxplot without the default outliers
plt.figure(figsize=(10, 7), dpi=600)
sns.boxplot(x='Group_numeric', y='F0.75_score', data=f075_combined_df, 
            palette=['thistle', 'khaki'], showfliers=False)

# Add jittered scatter plot for non-outliers, colored by model type
sns.stripplot(x='Group_numeric', y='F0.75_score', data=non_outliers_df_75, 
              hue='model_type', jitter=0.2, size=4.5, palette={'K-fold': 'red', 'Monte Carlo': 'blue'},
              marker='o', linewidth=0.4, edgecolor='black', alpha=0.6)

# Jitter the outliers for K-fold and Monte Carlo
jitter_strength = 0.15  

# Jitter the outliers for K-fold in tp8
x_jitter_kfold_tp8 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_kfold_tp8))
plt.scatter(outliers_kfold_tp8['Group_numeric'] + x_jitter_kfold_tp8, 
            outliers_kfold_tp8['F0.75_score'], color='red', s=35, marker='x', label='K-fold outliers (tp8)')

# Jitter the outliers for K-fold in tp14
x_jitter_kfold_tp14 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_kfold_tp14))
plt.scatter(outliers_kfold_tp14['Group_numeric'] + x_jitter_kfold_tp14, 
            outliers_kfold_tp14['F0.75_score'], color='red', s=35, marker='x', label='K-fold outliers (tp14)')

# Jitter the outliers for Monte Carlo in tp8
x_jitter_mc_tp8 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_mc_tp8))
plt.scatter(outliers_mc_tp8['Group_numeric'] + x_jitter_mc_tp8, 
            outliers_mc_tp8['F0.75_score'], color='blue', s=35, marker='x', label='Monte Carlo outliers (tp8)')

# Jitter the outliers for Monte Carlo in tp14
x_jitter_mc_tp14 = np.random.uniform(-jitter_strength, jitter_strength, size=len(outliers_mc_tp14))
plt.scatter(outliers_mc_tp14['Group_numeric'] + x_jitter_mc_tp14, 
            outliers_mc_tp14['F0.75_score'], color='blue', s=35, marker='x', label='Monte Carlo outliers (tp14)')

# Add means for each group
mean_tp8 = f075_combined_df[f075_combined_df['Group'] == 'tp8']['F0.75_score'].mean()
mean_tp14 = f075_combined_df[f075_combined_df['Group'] == 'tp14']['F0.75_score'].mean()

plt.scatter([0], [mean_tp8], color='limegreen', s=100, marker='+', edgecolor='black', linewidths=1.5, label='Mean (tp8)', zorder=10)
plt.scatter([1], [mean_tp14], color='limegreen', s=100, marker='+', edgecolor='black', linewidths=1.5, label='Mean (tp14)', zorder=10)

# Set Y-axis limits manually for better scaling (adjust as needed based on your data range)
plt.ylim(0.60, 0.80)

# Customize X-axis to show 'tp8' and 'tp14'
plt.xticks([0, 1], ['tp8', 'tp14'])

# Add light horizontal gridlines
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# Customize plot labels, title, and legend placement
plt.ylabel('Threshold Averaged F-0.75 Score', fontsize=13, labelpad=15)
plt.xlabel('Model Training Dataset', fontsize=13)
plt.title('Threshold Averaged F-0.75 Scores for tp8 and tp14 Models', fontsize=15, pad=10)
plt.tight_layout(1.0)
plt.subplots_adjust(top=0.85)

# Manually create legend entries for K-fold and Monte Carlo with transparency and black edgecolor
kfold_handle = plt.Line2D([0], [0], marker='o', color='red', label='K-fold', markersize=8, 
                          linestyle='None', markerfacecolor='red', markeredgewidth=0.8, 
                          markeredgecolor='black', alpha=0.6)
mc_handle = plt.Line2D([0], [0], marker='o', color='blue', label='Monte Carlo', markersize=8, 
                       linestyle='None', markerfacecolor='blue', markeredgewidth=0.8, 
                       markeredgecolor='black', alpha=0.6)

# Create outlier and mean handles for the legend
outliers_kfold_handle = plt.Line2D([0], [0], marker='x', color='red', label='K-fold outliers', markersize=8, 
                                   linestyle='None')
outliers_mc_handle = plt.Line2D([0], [0], marker='x', color='blue', label='Monte Carlo outliers', markersize=8, 
                                linestyle='None')

# Add means for each group with consistent size and marker style
mean_tp8_handle = plt.scatter([0], [mean_tp8], color='limegreen', s=100, marker='+', 
                              edgecolor='black', linewidths=1.5, label='Mean (tp8)', zorder=10)

mean_tp14_handle = plt.scatter([1], [mean_tp14], color='limegreen', s=100, marker='+', 
                               edgecolor='black', linewidths=1.5, label='Mean (tp14)', zorder=10)

# Customize legend placement (outside the plot to the right)
legend = plt.legend(handles=[kfold_handle, mc_handle, outliers_kfold_handle, outliers_mc_handle, 
                             mean_tp8_handle, mean_tp14_handle], 
                    loc='center left', bbox_to_anchor=(1, 0.5), title='Legend', fontsize=10)

# Darken the legend border
legend.get_frame().set_edgecolor('black')  
legend.get_frame().set_linewidth(0.5)  

# Set tick label size
plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12)  

# Display the plot with adjusted layout
plt.tight_layout()
plt.show()


# ==================== SCATTERPLOT FOR PEARSON CORRELATION =====================

# Separate the data into two dataframes based on the 'Group' column
df_tp8 = f075_combined_df[f075_combined_df['Group'] == 'tp8'][['model_type', 'F0.75_score']]
df_tp14 = f075_combined_df[f075_combined_df['Group'] == 'tp14'][['model_type', 'F0.75_score']]

# Reset indices for tp8 and tp14 to align the rows
df_tp8.reset_index(drop=True, inplace=True)
df_tp14.reset_index(drop=True, inplace=True)

# Rename columns for clarity
df_tp8.rename(columns={'F0.75_score': 'F0.75_score_tp8'}, inplace=True)
df_tp14.rename(columns={'F0.75_score': 'F0.75_score_tp14'}, inplace=True)

# Concatenate the two dataframes column-wise (axis=1)
merged_df = pd.concat([df_tp8, df_tp14['F0.75_score_tp14']], axis=1)

# Adjust the color palette: switch K-fold to red and Monte Carlo to blue
color_palette = {'K-fold': 'red', 'Monte Carlo': 'blue'}

# Plot the scatter plot with colors based on model_type
plt.figure(figsize=(10, 7), dpi=600)
scatter_plot = sns.scatterplot(x='F0.75_score_tp8', y='F0.75_score_tp14', hue='model_type', data=merged_df,
                               palette=color_palette, s=35, edgecolor='black', alpha=0.6, zorder=3)

# Add a regression line (without zorder)
sns.regplot(x='F0.75_score_tp8', y='F0.75_score_tp14', data=merged_df, scatter=False, 
            color='red', line_kws={'linewidth': 1.5}, ci=95)

# Calculate and display Pearson's correlation coefficient
pearson_r, _ = pearsonr(merged_df['F0.75_score_tp8'], merged_df['F0.75_score_tp14'])
plt.text(0.05, 0.95, f'Pearson r = {pearson_r:.2f}', fontsize=12, color='red', ha='left', va='top', transform=plt.gca().transAxes)

# Adjust X-axis range to fit all points
plt.xlim(0.55, 0.80)
plt.ylim(0.55, 0.80)
plt.tick_params(axis='x', labelsize=12) 
plt.tick_params(axis='y', labelsize=12) 

# Add labels and title
plt.title('Threshold Averaged F-0.75 Scores of tp8 and tp14 Paired Models', fontsize=15, pad=10)
plt.xlabel('Threshold Averaged F-0.75 Score (tp8)', fontsize=13, labelpad=10)
plt.ylabel('Threshold Averaged F-0.75 Score (tp14)', fontsize=13, labelpad=15)

# Add grid lines with a lower zorder to place them behind the dots
plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)

# Manually adjust legend to match the transparency, size, and edge color
handles, labels = scatter_plot.get_legend_handles_labels()
for handle in handles:
    handle.set_alpha(0.6) 
    handle.set_edgecolor('black')  
    handle.set_linewidth(0.5)  

# Customize legend placement (bottom right)
legend = plt.legend(handles=handles, labels=labels, loc='lower right', title='Cross-validation')
legend.get_frame().set_edgecolor('black') 
legend.get_frame().set_linewidth(0.5) 

# Show the plot
plt.tight_layout()
plt.show()

# ================== Bland-Altman Plot with F-0.75 Score ====================
# Bland-Altman Plot for F-0.75 Score
plt.figure(figsize=(6, 5), dpi=600)
mean_f075 = (combined_mean_f075['F0.75_score_tp8'] + combined_mean_f075['F0.75_score_tp14']) / 2
diff_f075 = combined_mean_f075['F0.75_score_tp8'] - combined_mean_f075['F0.75_score_tp14']

# Color the dots by model type (Monte Carlo or K-fold)
colors = {'Monte Carlo': 'blue', 'K-fold': 'red'}
for model_type in ['Monte Carlo', 'K-fold']:
    subset = combined_mean_f075[combined_mean_f075['model_type'] == model_type]
    plt.scatter(subset['F0.75_score_tp8'], subset['F0.75_score_tp14'] - subset['F0.75_score_tp8'], 
                color=colors[model_type], label=model_type, s=10, alpha=0.6, edgecolors='black', linewidth=0.4)

# Add horizontal lines for the mean difference and limits
plt.axhline(np.mean(diff_f075), alpha=0.5, color='gray', linestyle='--', label='Mean Difference')
plt.axhline(np.mean(diff_f075) + 1.96 * np.std(diff_f075), alpha=0.5, color='limegreen', linestyle=':', label='Upper Limit')
plt.axhline(np.mean(diff_f075) - 1.96 * np.std(diff_f075), alpha=0.5, color='red', linestyle=':', label='Lower Limit')

# Labels and title
plt.xlabel('Average of Mean F-0.75 for tp8 tp14 Model Pairs', fontsize=8, labelpad=5)
plt.ylabel('Difference in Mean F-0.75 Score of Model Pairs (tp8 - tp14)', fontsize=8)
plt.title('Mean F-0.75 Difference Plot for tp8 tp14 Model Pairs', fontsize=10)

# Ensure even X and Y axis limits
plt.xlim([0.4, 0.8])  
plt.ylim([-0.15, 0.15])  

# Legend
legend = plt.legend(title="Legend", title_fontsize=7, prop={'size': 7}, loc='center left', bbox_to_anchor=(1, 0.5))
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.5)

plt.tick_params(axis='x', labelsize=6) 
plt.tick_params(axis='y', labelsize=6)  

plt.tight_layout(pad=2.0)
plt.show()

# Save results with descriptions, equations, and interpretations
test_descriptions = [
    "Paired t-test: Compares the means of two paired datasets (tp8 vs tp14) to assess whether there is a significant difference.",
    "Wilcoxon Signed-Rank Test: A non-parametric test comparing the medians of two paired datasets without assuming normal distribution.",
    "Pearson Correlation: Measures the linear relationship between tp8 and tp14 F-0.75 scores.",
    "Shapiro-Wilk Test: Tests for the normality of the data distribution."
]

test_equations = [
    "t = (mean difference) / (standard error of the difference)",
    "W = sum of the signed ranks of the differences between tp8 and tp14",
    "r = covariance(tp8, tp14) / (std(tp8) * std(tp14))",
    "W = (observed statistic for normality)"
]

interpretation_guidelines = [
    "If the p-value is less than 0.05, the difference is statistically significant.",
    "If the p-value is less than 0.05, the difference is statistically significant.",
    "Correlation ranges from -1 (perfect negative) to 1 (perfect positive).",
    "If the p-value is less than 0.05, the data deviates significantly from a normal distribution."
]

# Prepare the results data
results_data = [
    ['Paired t-test (F-0.75)' if p_tp8 > 0.05 and p_tp14 > 0.05 else 'Wilcoxon signed-rank test (F-0.75)',
     t_stat_f075 if p_tp8 > 0.05 and p_tp14 > 0.05 else w_stat_f075, 
     p_value_f075 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f075, 
     'No' if (p_value_f075 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f075) >= 0.05 else 'Yes'],
    ['Pearson Correlation (F-0.75)', corr_f075, 'N/A', 'N/A'],
    ['Shapiro-Wilk tp8', stat_tp8, p_tp8, 'No' if p_tp8 >= 0.05 else 'Yes'],
    ['Shapiro-Wilk tp14', stat_tp14, p_tp14, 'No' if p_tp14 >= 0.05 else 'Yes']
]

# Interpretations for the results
interpretations = [
    f"{'t-stat' if p_tp8 > 0.05 and p_tp14 > 0.05 else 'w-stat'} = {t_stat_f075 if p_tp8 > 0.05 and p_tp14 > 0.05 else w_stat_f075}, "
    f"p-value = {p_value_f075 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f075}, "
    f"Interpretation: {'No significant difference' if (p_value_f075 if p_tp8 > 0.05 and p_tp14 > 0.05 else p_value_wilcoxon_f075) >= 0.05 else 'Significant difference'}",
    f"Pearson r = {corr_f075}, Interpretation: {'Strong positive correlation' if corr_f075 > 0.7 else 'Weak or moderate correlation'}",
    f"Shapiro-Wilk stat tp8 = {stat_tp8}, p-value = {p_tp8}, Interpretation: {'Normally distributed' if p_tp8 > 0.05 else 'Not normally distributed'}",
    f"Shapiro-Wilk stat tp14 = {stat_tp14}, p-value = {p_tp14}, Interpretation: {'Normally distributed' if p_tp14 > 0.05 else 'Not normally distributed'}"
]

# Prepare the full DataFrame
results_df = pd.DataFrame({
    'Test Name': [row[0] for row in results_data],
    'Statistic': [row[1] for row in results_data],
    'p-value': [row[2] for row in results_data],
    'Significant (alpha = 0.05)': [row[3] for row in results_data],
    'Test Description': test_descriptions,
    'Equation': test_equations,
    'Interpretation': interpretations,
    'Interpretation Guidelines': interpretation_guidelines
})

# Save the DataFrame to a CSV file
output_file_path = './tp8_tp14_f075_full_statistical_analysis.csv'
results_df.to_csv(output_file_path, index=False)

print(f"Results saved to {output_file_path}")




