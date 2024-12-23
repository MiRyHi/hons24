# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:40:29 2024

@author: Mitch
"""

import pandas as pd
import os

# Define the file paths for the two groups
tp8_metrics_folder = './tp8_metrics/'
tp14_metrics_folder = './tp14_metrics/'

# Thresholds for the CSV files
thresholds = [50, 60, 70, 80, 90]

# load metrics csv files

# Function to load CSV files into a dictionary for a given folder
def load_metrics(folder, thresholds, group):
    metrics_dict = {}
    for threshold in thresholds:
        file_path = os.path.join(folder, f'tp{group}_metrics_{threshold}.csv')  # Fixing the file name construction
        metrics_dict[threshold] = pd.read_csv(file_path)
    return metrics_dict

# Load the CSV files for both groups
tp8_metrics = load_metrics(tp8_metrics_folder, thresholds, '8')
tp14_metrics = load_metrics(tp14_metrics_folder, thresholds, '14')

# Check if files are loaded correctly by printing the keys (thresholds)
print(tp8_metrics.keys())
print(tp14_metrics.keys())

# Inspect one of the threshold dataframes, for example from tp8_metrics for threshold 50
print(tp8_metrics[50].head())  # First few rows of the 50% threshold data


# Print column names to verify consistency between CSVs
print(tp8_metrics[50].columns)
print(tp14_metrics[50].columns)




# consolidate relevant columns 

# Function to extract relevant performance metrics for each threshold
def extract_metrics(metrics_dict, thresholds, metrics):
    all_data = []
    for threshold in thresholds:
        df = metrics_dict[threshold]
        # Extract model_name and the relevant metrics (e.g., precision, F1 score) along with the threshold
        selected_columns = ['model_name'] + [f'{metric} (Kas_Gearty_no_cut)' for metric in metrics]
        selected_data = df[selected_columns].copy()
        selected_data['threshold'] = threshold  # Add a column for the threshold
        all_data.append(selected_data)
    
    # Concatenate data for all thresholds into one dataframe
    return pd.concat(all_data, ignore_index=True)

# Define the performance metrics we want to extract (precision, F1 score)
metrics_to_extract = ['precision', 'f1_score']

# Extract metrics for tp8 and tp14
tp14_consolidated = extract_metrics(tp14_metrics, thresholds, metrics_to_extract)

# Function to extract relevant performance metrics with flexible column name handling #modified fix for column name in t50 that I manually changed
def extract_metrics_flexible(metrics_dict, thresholds, precision_col_names, f1_col_names):
    all_data = []
    for threshold in thresholds:
        df = metrics_dict[threshold]
        
        # Check for correct column names
        precision_col = next((col for col in precision_col_names if col in df.columns), None)
        f1_col = next((col for col in f1_col_names if col in df.columns), None)
        
        if precision_col is None or f1_col is None:
            raise KeyError(f"Could not find precision or F1 score columns in threshold {threshold}")
        
        # Extract model_name and the relevant metrics along with the threshold
        selected_columns = ['model_name', precision_col, f1_col]
        selected_data = df[selected_columns].copy()
        selected_data['threshold'] = threshold  # Add a column for the threshold
        selected_data.columns = ['model_name', 'precision', 'f1_score', 'threshold']  # Standardize column names
        all_data.append(selected_data)
    
    # Concatenate data for all thresholds into one dataframe
    return pd.concat(all_data, ignore_index=True)

# Define the possible column names for precision and F1 score
precision_columns_tp8 = ['precision (Kas_Gearty_no_cut)', 'precision (PPV)']
f1_columns_tp8 = ['f1_score (Kas_Gearty_no_cut)', 'F1 score']

# Extract metrics for tp8 with flexible column name handling
tp8_consolidated_flexible = extract_metrics_flexible(tp8_metrics, thresholds, precision_columns_tp8, f1_columns_tp8)

# Verify if the extraction worked
print(tp8_consolidated_flexible.head())

# View the consolidated data for tp14
print(tp14_consolidated.head())

# Function to calculate mean precision and F1 score for each model across thresholds in tp14
def calculate_mean_metrics(consolidated_df):
    # Group by model_name and calculate mean for precision and f1 score
    mean_metrics = consolidated_df.groupby('model_name')[['precision (Kas_Gearty_no_cut)', 'f1_score (Kas_Gearty_no_cut)']].mean()
    # No need to sort here if sorting will happen after merging
    return mean_metrics

# Calculate mean metrics for tp8 and tp14
tp14_mean_metrics = calculate_mean_metrics(tp14_consolidated)

# Function to calculate mean precision and F1 score for each model across thresholds in tp8 with flexible column names
def calculate_mean_metrics_flexible(consolidated_df):
    # Group by model_name and calculate mean for precision and f1 score
    mean_metrics = consolidated_df.groupby('model_name')[['precision', 'f1_score']].mean()
    # No need to sort here if sorting will happen after merging
    return mean_metrics

# Calculate mean metrics for tp8
tp8_mean_metrics_flexible = calculate_mean_metrics_flexible(tp8_consolidated_flexible)

# Verify
print(tp8_mean_metrics_flexible.head())
print(tp14_mean_metrics.head())

# Rename model_name in tp8 dataframes
tp8_consolidated_flexible['model_name'] = tp8_consolidated_flexible['model_name'].str.replace('_dist', '_tp8')

# Rename model_name in tp14 dataframes
tp14_consolidated['model_name'] = tp14_consolidated['model_name'].str.replace('_add_dist', '_tp14')

# Verify the changes
print(tp8_consolidated_flexible.head())
print(tp14_consolidated.head())

# Create a new column with the core model name by removing the suffixes
tp8_mean_metrics_flexible['core_model_name'] = tp8_mean_metrics_flexible.index.str.replace('_tp8', '')
tp14_mean_metrics['core_model_name'] = tp14_mean_metrics.index.str.replace('_tp14', '')

# Rename columns in tp8 to add _tp8 suffix before the merge
tp8_mean_metrics_flexible = tp8_mean_metrics_flexible.rename(columns={'precision': 'precision_tp8', 'f1_score': 'f1_score_tp8'})

# Rename columns in tp14 to add _tp14 suffix before the merge
tp14_mean_metrics = tp14_mean_metrics.rename(columns={'precision (Kas_Gearty_no_cut)': 'precision_tp14', 
                                                      'f1_score (Kas_Gearty_no_cut)': 'f1_score_tp14'})

# Reset index to ensure model names are part of the dataframe and prepare for merging
tp8_mean_metrics_flexible.reset_index(drop=True, inplace=True)
tp14_mean_metrics.reset_index(drop=True, inplace=True)

# Rename core_model_name in tp8 dataframes (fix)
tp8_mean_metrics_flexible['core_model_name'] = tp8_mean_metrics_flexible['core_model_name'].str.replace('_dist', '')

# Rename core_model_name in tp14 dataframes (fix)
tp14_mean_metrics['core_model_name'] = tp14_mean_metrics['core_model_name'].str.replace('_add_dist', '')

# Merge based on the core model name
combined_mean_metrics = pd.merge(tp8_mean_metrics_flexible, 
                                 tp14_mean_metrics, 
                                 on='core_model_name')

# View the merged dataframe with suffixes in place
print(combined_mean_metrics.head())

import matplotlib.pyplot as plt

# Add a new column to indicate whether the model was trained with Mc or Kf
combined_mean_metrics['model_type'] = combined_mean_metrics['core_model_name'].apply(lambda x: 'Mc' if 'Mc' in x else 'Kf')

# plotting -----------------------------------------------------------------

# Define colors for Mc and Kf model dots on plot
colors = {'Mc': 'blue', 'Kf': 'red'}

# Plot precision comparison with different colors for Mc and Kf
plt.figure(figsize=(5, 5), dpi=600)
for model_type in ['Mc', 'Kf']:
    subset = combined_mean_metrics[combined_mean_metrics['model_type'] == model_type]
    plt.scatter(subset['precision_tp8'], subset['precision_tp14'], 
                color=colors[model_type], label=model_type, s=30, alpha=0.5, edgecolors='black', linewidth=0.4)

# Adjust the size of the axis number text (ticks)
plt.tick_params(axis='both', which='major', labelsize=8)

plt.xlabel('Mean Precision (tp8 Models 1-96)')
plt.ylabel('Mean Precision (tp14 Models 1-96)')
plt.title('Mean Precision Comparison (all thresholds): tp8 vs tp14 Training Pairs')
legend = plt.legend()
legend.get_frame().set_edgecolor('black')  
legend.get_frame().set_linewidth(0.7)  
plt.axline((0, 0), slope=1, color='gray', linestyle=':', alpha=0.7, linewidth=1.0)
plt.grid(True, linestyle='--', alpha=0.45)
plt.show()

# Plot F1 score comparison with different colors for Mc and Kf
plt.figure(figsize=(5, 5), dpi=600)
for model_type in ['Mc', 'Kf']:
    subset = combined_mean_metrics[combined_mean_metrics['model_type'] == model_type]
    plt.scatter(subset['f1_score_tp8'], subset['f1_score_tp14'], 
                color=colors[model_type], label=model_type, s=30, alpha=0.5, edgecolors='black', linewidth=0.4)

# Adjust the size of the axis number text (ticks)
plt.tick_params(axis='both', which='major', labelsize=8)

plt.xlabel('Mean F1 Score (tp8 Models 1-96)')
plt.ylabel('Mean F1 Score (tp14 Models 1-96)')
plt.title('Mean F1 Score Comparison (all thresholds): tp8 vs tp14 Training Pairs')
legend = plt.legend()
legend.get_frame().set_edgecolor('black') 
legend.get_frame().set_linewidth(0.7) 
plt.axline((0, 0), slope=1, color='gray', linestyle=':', alpha=0.7, linewidth=1.0)
plt.grid(True, linestyle='--', alpha=0.45)
plt.show()


# plot the same but change axis scale to ZOOM -----------------------------------------

plt.figure(figsize=(5, 5), dpi=600)
for model_type in ['Mc', 'Kf']:
    subset = combined_mean_metrics[combined_mean_metrics['model_type'] == model_type]
    plt.scatter(subset['precision_tp8'], subset['precision_tp14'], 
                color=colors[model_type], label=model_type, s=30, alpha=0.5, edgecolors='black', linewidth=0.4)

# Set X and Y limits for precision between 0.6 and 1.0
plt.xlim(0.7, 1.0)
plt.ylim(0.7, 1.0)

# Adjust the size of the axis number text (ticks)
plt.tick_params(axis='both', which='major', labelsize=8)

plt.xlabel('Mean Precision (tp8 Models 1-96)')
plt.ylabel('Mean Precision (tp14 Models 1-96)')
plt.title('Mean Precision Comparison (all thresholds): tp8 vs tp14 Training Pairs')
legend = plt.legend()
legend.get_frame().set_edgecolor('black')  
legend.get_frame().set_linewidth(0.5)  
plt.axline((0, 0), slope=1, color='gray', linestyle=':', alpha=0.7, linewidth=1.0)
plt.grid(True, linestyle='--', alpha=0.45)
plt.show()

# Plot F1 score comparison with different colors for Mc and Kf
plt.figure(figsize=(5, 5), dpi=600)
for model_type in ['Mc', 'Kf']:
    subset = combined_mean_metrics[combined_mean_metrics['model_type'] == model_type]
    plt.scatter(subset['f1_score_tp8'], subset['f1_score_tp14'], 
                color=colors[model_type], label=model_type, s=30, alpha=0.5, edgecolors='black', linewidth=0.4)

# Set X and Y limits for F1 score between 0.3 and 0.8
plt.xlim(0.3, 0.8)
plt.ylim(0.3, 0.8)

# Adjust the size of the axis number text (ticks)
plt.tick_params(axis='both', which='major', labelsize=8)

plt.xlabel('Mean F1 Score (tp8 Models 1-96)')
plt.ylabel('Mean F1 Score (tp14 Models 1-96)')
plt.title('Mean F1 Score Comparison (all thresholds): tp8 vs tp14 Training Pairs')
legend = plt.legend()
legend.get_frame().set_edgecolor('black') 
legend.get_frame().set_linewidth(0.5) 
plt.axline((0, 0), slope=1, color='gray', linestyle=':', alpha=0.7, linewidth=1.0)
plt.grid(True, linestyle='--', alpha=0.45)
plt.show()