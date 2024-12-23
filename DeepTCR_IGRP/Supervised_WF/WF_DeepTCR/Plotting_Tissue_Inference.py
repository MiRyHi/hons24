# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:25:09 2024

@author: Mitch
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from matplotlib import cm
import numpy as np

# Define file paths for tp8 and tp14 CSV files
tp14_files = [
    "tp14_metrics/tp14_metrics_50.csv", 
    "tp14_metrics/tp14_metrics_60.csv", 
    "tp14_metrics/tp14_metrics_70.csv", 
    "tp14_metrics/tp14_metrics_80.csv", 
    "tp14_metrics/tp14_metrics_90.csv"
]

tp8_files = [
    "tp8_metrics/tp8_metrics_50.csv", 
    "tp8_metrics/tp8_metrics_60.csv", 
    "tp8_metrics/tp8_metrics_70.csv", 
    "tp8_metrics/tp8_metrics_80.csv", 
    "tp8_metrics/tp8_metrics_90.csv"
]

# Function to read all CSVs and return a list of DataFrames
def load_csv_files(file_paths):
    return [pd.read_csv(file) for file in file_paths]

# Load tp14 and tp8 CSV files
tp14_dataframes = load_csv_files(tp14_files)
tp8_dataframes = load_csv_files(tp8_files)

# Function to clean each dataframe
def clean_dataframe(df):
    # Retain column 0
    first_column_name = df.columns[0]
    first_column = df[first_column_name]

    # Drop columns from index 1 to 45
    df_cleaned = df.drop(df.columns[1:46], axis=1)

    # Remove columns with "_rk" suffix in their names
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.endswith('_rk')]

    # Add back the first column if it's not already in the dataframe
    if first_column_name not in df_cleaned.columns:
        df_cleaned.insert(0, first_column_name, first_column)

    return df_cleaned

# Apply the cleaning function to tp14 and tp8 dataframes
tp14_cleaned = [clean_dataframe(df) for df in tp14_dataframes]
tp8_cleaned = [clean_dataframe(df) for df in tp8_dataframes]

# Example of checking the cleaned data
tp14_cleaned[0].head()

# Function to overwrite the original CSV files
def save_cleaned_data(dataframes, original_files):
    for df, file_path in zip(dataframes, original_files):
        # Save the cleaned dataframe back to the original file path
        df.to_csv(file_path, index=False)
        print(f"Overwritten cleaned data to: {file_path}")

# Overwrite the original tp14 and tp8 CSV files with the cleaned data
save_cleaned_data(tp14_cleaned, tp14_files)
save_cleaned_data(tp8_cleaned, tp8_files)


# Function to load CSV files and combine the relevant columns
def load_and_combine(files):
    dfs = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Replace underscores with spaces in column names
    combined_df.columns = combined_df.columns.str.replace('_', ' ')
    
    return combined_df[['model name', 'F1 score', 'F0.5 score', 'F0.75 score']]  # Ensure names match after the replacement

# Load tp8 and tp14 dataframes
tp14_combined = load_and_combine(tp14_files)
tp8_combined = load_and_combine(tp8_files)

# Function to calculate the mean F1, F0.5, and F0.75 scores for each model
def calculate_mean_scores(df):
    mean_df = df.groupby('model name').mean().reset_index()
    mean_df.columns = ['model name', 'F1 mean', 'F0.5 mean', 'F0.75 mean']
    return mean_df

# Calculate the mean scores for tp8 and tp14 models
tp14_mean_scores = calculate_mean_scores(tp14_combined)
tp8_mean_scores = calculate_mean_scores(tp8_combined)

# Function to sort by the F-metric and get top 10 models
def get_top_models(mean_scores_df, f_metric, cleaned_dataframes):
    # Sort by the specified F-metric column (highest to lowest)
    sorted_df = mean_scores_df.sort_values(by=f_metric, ascending=False).head(10)
    
    # Create a new dataframe to store the model names and F-scores
    result_df = pd.DataFrame()
    result_df['model_name'] = sorted_df['model name'].values
    
    # For each top model, extract the F1/F0.75/F0.5 score across all thresholds
    for i, model in enumerate(result_df['model_name']):
        f_scores = []
        for df in cleaned_dataframes:
            # Filter to find the row for the current model
            row = df[df['model name'] == model]
            if not row.empty:
                f_scores.append(row[f_metric.replace(' mean', ' score')].values[0])
            else:
                f_scores.append(None)  # In case a row is not found for a model in a threshold dataframe
        
        # Add the F-scores for this model as new columns (one per threshold)
        result_df.loc[i, 'Threshold 50'] = f_scores[0]
        result_df.loc[i, 'Threshold 60'] = f_scores[1]
        result_df.loc[i, 'Threshold 70'] = f_scores[2]
        result_df.loc[i, 'Threshold 80'] = f_scores[3]
        result_df.loc[i, 'Threshold 90'] = f_scores[4]
    
    return result_df

# Process for tp8 models
tp8_f1_df = get_top_models(tp8_mean_scores, 'F1 mean', tp8_cleaned)
tp8_f075_df = get_top_models(tp8_mean_scores, 'F0.75 mean', tp8_cleaned)
tp8_f05_df = get_top_models(tp8_mean_scores, 'F0.5 mean', tp8_cleaned)

# Process for tp14 models
tp14_f1_df = get_top_models(tp14_mean_scores, 'F1 mean', tp14_cleaned)
tp14_f075_df = get_top_models(tp14_mean_scores, 'F0.75 mean', tp14_cleaned)
tp14_f05_df = get_top_models(tp14_mean_scores, 'F0.5 mean', tp14_cleaned)

# Example to check the F1 results for tp8
print(tp8_f1_df.head())

# Function to plot models from tp8 and tp14 together
def plot_models(df_tp8, df_tp14, metric_name):
    thresholds = ['Threshold 50', 'Threshold 60', 'Threshold 70', 'Threshold 80', 'Threshold 90']
    
    plt.figure(figsize=(10, 6))
    
    # Plot tp8 models with one color (e.g., blue)
    for i, row in df_tp8.iterrows():
        plt.plot(thresholds, row[thresholds], label=row['model_name'] + ' (tp8)', color='blue', marker='o', linestyle='-', alpha=0.7)
    
    # Plot tp14 models with another color (e.g., red)
    for i, row in df_tp14.iterrows():
        plt.plot(thresholds, row[thresholds], label=row['model_name'] + ' (tp14)', color='red', marker='o', linestyle='--', alpha=0.7)
    
    # Add title, labels, and legend
    plt.title(f'{metric_name} Scores Across Thresholds for tp8 and tp14 Models')
    plt.xlabel('Thresholds')
    plt.ylabel(f'{metric_name} Score')
    plt.legend(loc='best', fontsize='small')
    
    # Display the plot
    plt.grid(True)
    plt.show()

# Plot for F1 mean models
plot_models(tp8_f1_df, tp14_f1_df, 'F1')

# Plot for F0.75 mean models
plot_models(tp8_f075_df, tp14_f075_df, 'F0.75')

# Plot for F0.5 mean models
plot_models(tp8_f05_df, tp14_f05_df, 'F0.5')

# Combine the tp8 and tp14 dataframes, adding a column to indicate the group
tp8_f1_df['Group'] = 'tp8'
tp14_f1_df['Group'] = 'tp14'
combined_f1_df = pd.concat([tp8_f1_df, tp14_f1_df])

# Melt the dataframe to get it in a long format suitable for seaborn
combined_f1_df_melted = combined_f1_df.melt(id_vars=['model_name', 'Group'], 
                                            value_vars=['Threshold 50', 'Threshold 60', 'Threshold 70', 'Threshold 80', 'Threshold 90'],
                                            var_name='Threshold', value_name='F1 Score')

# Function to plot each dataframe separately
def plot_line_graphs(df, metric_name, group_name):
    thresholds = ['Threshold 50', 'Threshold 60', 'Threshold 70', 'Threshold 80', 'Threshold 90']
    
    plt.figure(figsize=(10, 6))
    
    # Plot each model in the dataframe
    for i, row in df.iterrows():
        plt.plot(thresholds, row[thresholds], label=row['model_name'], marker='o', linestyle='-', alpha=0.7)
    
    # Add title, labels, and legend
    plt.title(f'Top 10 {group_name} Models by Averaged Threshold {metric_name} Score', fontsize=14)
    plt.xlabel('Thresholds', fontsize=12)
    plt.ylabel(f'{metric_name} Score', fontsize=12)
    plt.legend(loc='best', fontsize='small')
    
    # Display the plot
    plt.grid(True)
    plt.show()

# Plot each dataframe separately
# F1 score plots
plot_line_graphs(tp8_f1_df, 'F1', 'tp8')
plot_line_graphs(tp14_f1_df, 'F1', 'tp14')

# F0.75 score plots
plot_line_graphs(tp8_f075_df, 'F0.75', 'tp8')
plot_line_graphs(tp14_f075_df, 'F0.75', 'tp14')

# F0.5 score plots
plot_line_graphs(tp8_f05_df, 'F0.5', 'tp8')
plot_line_graphs(tp14_f05_df, 'F0.5', 'tp14')

# Function to plot accuracy, precision, and recall for each group of models using viridis color scheme
def plot_metric(models_df, cleaned_dfs, metric_column, metric_name, group_name, f_score):
    thresholds_numeric = [0, 1, 2, 3, 4]  # Numeric values corresponding to thresholds
    threshold_labels = ['T50', 'T60', 'T70', 'T80', 'T90']  # Custom labels for x-axis

    plt.figure(figsize=(10, 6))
    
    # Generate colors from the viridis colormap
    colors = cm.turbo(np.linspace(0, 1, len(models_df)))
    
    # Loop over each top model and plot the corresponding metric
    for i, row in models_df.iterrows():
        model_name = row['model_name']  # From top_models_df
        y_values = []
        
        # Loop over each dataframe corresponding to thresholds 50, 60, 70, etc.
        for j, df in enumerate(cleaned_dfs):
            # Check for the correct column name, either 'recall' or 'recall (TPR)'
            if metric_column not in df.columns:
                if 'recall (TPR)' in df.columns:
                    metric_column = 'recall (TPR)'
                elif 'recall' in df.columns:
                    metric_column = 'recall'
                else:
                    print(f"'{metric_column}' column missing in dataframe for threshold {threshold_labels[j]} for model '{model_name}', skipping.")
                    y_values.append(np.nan)  # If no column exists, append NaN and skip
                    continue

            # Filter the cleaned dataframes for the same model
            if model_name in df['model_name'].values:
                model_data = df[df['model_name'] == model_name]
                y_values.append(model_data[metric_column].values[0])  # Collect the metric value for this threshold
            else:
                y_values.append(np.nan)  # If no data, append NaN

        # Plot only if there is some valid data
        if any(~np.isnan(y_values)):
            plt.plot(thresholds_numeric, y_values, label=model_name, marker='o', linestyle='-', color=colors[i], alpha=0.7)

    # Add title, labels, and customize font sizes
    plt.title(f'{metric_name} Scores Across Thresholds for {group_name} (Top 10 models by {f_score})', fontsize=16)
    plt.xlabel('Thresholds', fontsize=14)
    plt.ylabel(f'{metric_name} Score', fontsize=14)
    
    # Customize tick font sizes and set custom labels for thresholds
    plt.xticks(ticks=thresholds_numeric, labels=threshold_labels, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Customize legend font size
    plt.legend(loc='best', fontsize='small')  # Legend fontsize should also be 'fontsize'
    
    # Display the plot
    plt.grid(True)
    plt.show()

# Function to filter models and plot accuracy, precision, and recall
def extract_and_plot_top_models(top_models_df, cleaned_dfs, metric_column, metric_name, group_name, f_score):
    # Plot the metric for the top models
    plot_metric(top_models_df, cleaned_dfs, metric_column, metric_name, group_name, f_score)

# Example usage:
# For tp8 and tp14, plot accuracy, precision, and recall based on F1, F0.75, F0.5 scores.

# Accuracy plot for tp8 and tp14 models (based on F1 score)
extract_and_plot_top_models(tp8_f1_df, tp8_cleaned, 'accuracy', 'Accuracy', 'tp8', 'F1 score')
extract_and_plot_top_models(tp8_f1_df, tp8_cleaned, 'precision (PPV)', 'Precision', 'tp8', 'F1 score')
extract_and_plot_top_models(tp8_f1_df, tp8_cleaned, 'recall', 'Recall', 'tp8', 'F1 score')
extract_and_plot_top_models(tp14_f1_df, tp14_cleaned, 'accuracy', 'Accuracy', 'tp14', 'F1 score')
extract_and_plot_top_models(tp14_f1_df, tp14_cleaned, 'precision (PPV)', 'Precision', 'tp14', 'F1 score')
extract_and_plot_top_models(tp14_f1_df, tp14_cleaned, 'recall', 'Recall', 'tp14', 'F1 score')

# Precision plot for tp8 and tp14 models (based on F0.75 score)
extract_and_plot_top_models(tp8_f075_df, tp8_cleaned, 'accuracy', 'Accuracy', 'tp8', 'F0.75 score')
extract_and_plot_top_models(tp8_f075_df, tp8_cleaned, 'precision (PPV)', 'Precision', 'tp8', 'F0.75 score')
extract_and_plot_top_models(tp8_f075_df, tp8_cleaned, 'recall', 'Recall', 'tp8', 'F0.75 score')
extract_and_plot_top_models(tp14_f075_df, tp14_cleaned, 'accuracy', 'Accuracy', 'tp14', 'F0.75 score')
extract_and_plot_top_models(tp14_f075_df, tp14_cleaned, 'precision (PPV)', 'Precision', 'tp14', 'F0.75 score')
extract_and_plot_top_models(tp14_f075_df, tp14_cleaned, 'recall', 'Recall', 'tp14', 'F0.75 score')

# Recall plot for tp8 and tp14 models (based on F0.5 score)
extract_and_plot_top_models(tp8_f05_df, tp8_cleaned, 'accuracy', 'Accuracy', 'tp8', 'F0.5 score')
extract_and_plot_top_models(tp8_f05_df, tp8_cleaned, 'precision (PPV)', 'Precision','tp8', 'F0.5 score')
extract_and_plot_top_models(tp8_f05_df, tp8_cleaned, 'recall', 'Recall', 'tp8', 'F0.5 score')
extract_and_plot_top_models(tp14_f05_df, tp14_cleaned, 'accuracy', 'Accuracy', 'tp14', 'F0.5 score')
extract_and_plot_top_models(tp14_f05_df, tp14_cleaned, 'precision (PPV)', 'Precision', 'tp14', 'F0.5 score')
extract_and_plot_top_models(tp14_f05_df, tp14_cleaned, 'recall', 'Recall', 'tp14', 'F0.5 score')


# Function to extract the relevant data for each model and threshold
# Function to extract the relevant data for each model and threshold
def extract_threshold_data(cleaned_dfs, models_series, group_name):
    tissue_types = ['bld igg', 'bld pd1', 'bld spt', 'pln igg', 'pln pd1', 'pln spt', 'panc igg', 'panc pd1', 'panc spt']
    thresholds = ['t50', 't60', 't70', 't80', 't90']
    
    # Initialize a dictionary to store data for each threshold
    threshold_data = {threshold: pd.DataFrame(index=tissue_types) for threshold in thresholds}
    
    # Loop over each model in the models_series (Series, not DataFrame)
    for model_name in models_series:
        print(f"Processing model: {model_name}")
        
        # Loop over each dataframe corresponding to thresholds (e.g., threshold 50 is first dataframe, etc.)
        for i, df in enumerate(cleaned_dfs):
            # Determine which threshold we're working with
            threshold_label = thresholds[i]
            
            # Collect the 9 tissue-type data columns for this model at this threshold
            columns = [f'{threshold_label} {tissue}' for tissue in tissue_types]
            print(f"Looking for columns: {columns} in {threshold_label}")
            
            # Check if columns exist in the dataframe
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns in {threshold_label}: {missing_columns}")
                continue
            
            if model_name in df['model_name'].values:
                print(f"Model {model_name} found in dataframe for {threshold_label}")
                model_data = df[df['model_name'] == model_name]
                
                # Extract the values for the current threshold and model
                values = model_data[columns].values.flatten()  # Get the 9 values
                threshold_data[threshold_label][model_name] = values  # Add to the respective threshold dataframe
            else:
                print(f"Model {model_name} not found in dataframe for {threshold_label}")
    
    return threshold_data

# Function to create a heatmap for each threshold with column-wise normalization and original value annotations
def create_heatmaps(threshold_data, group_name):
    for threshold, data in threshold_data.items():
        plt.figure(figsize=(24, 10))  # Adjust size to fit all models
        
        # Save the original data for annotation
        original_data = data.copy()
        
        # Normalize the data per column (across each model)
        normalized_data = data.div(data.max(axis=0), axis=1)
        
        # Create the heatmap using normalized data for coloring
        ax = sns.heatmap(normalized_data, annot=original_data, fmt='.2f', cmap='plasma', cbar=True)
        
        # Adjust the color bar to reflect the range of original values
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Adjust these ticks based on normalization scale
        
        # Set tick labels, formatted to two decimal places
        cbar.set_ticklabels([f'{original_data.min().min():.2f}', 
                             f'{original_data.mean().min():.2f}', 
                             f'{original_data.mean().mean():.2f}', 
                             f'{original_data.max().min():.2f}', 
                             f'{original_data.max().max():.2f}'])  # Format to 2 decimal places
        
        plt.title(f'Heatmap for {group_name} at {threshold}', fontsize=16)
        plt.xlabel('Models', fontsize=14)
        
        # Rotate Y-axis labels (tissue/type) to be horizontal
        plt.yticks(rotation=0, fontsize=12)
        plt.xticks(rotation=90, fontsize=12)  # Keep the X-axis labels rotated vertically for readability
        
        plt.show()

# Combine the selected top models from tp8 and tp14 by concatenating their model names
tp8_selected_models = pd.concat([tp8_f1_df['model_name'], tp8_f075_df['model_name'], tp8_f05_df['model_name']]).drop_duplicates()
tp14_selected_models = pd.concat([tp14_f1_df['model_name'], tp14_f075_df['model_name'], tp14_f05_df['model_name']]).drop_duplicates()

# Extract data for the selected models from tp8_cleaned and tp14_cleaned
tp8_combined_data = extract_threshold_data(tp8_cleaned, tp8_selected_models, 'tp8')
tp14_combined_data = extract_threshold_data(tp14_cleaned, tp14_selected_models, 'tp14')

# Define the desired order of rows (tissue types) for the heatmaps
custom_row_order = ['panc spt', 'bld spt', 'pln spt', 'panc igg', 'bld igg', 'pln igg', 'panc pd1', 'bld pd1', 'pln pd1']

# Combine the tp8 and tp14 models into one dataframe for each threshold
combined_data = {}
for threshold in tp8_combined_data.keys():
    # Combine tp8 and tp14 columns into one dataframe for each threshold
    combined_df = pd.concat([tp8_combined_data[threshold], tp14_combined_data[threshold]], axis=1)
    
    # Reorder rows according to custom_row_order
    combined_data[threshold] = combined_df.loc[custom_row_order]

# Create heatmaps with column-wise normalization, plasma color, and original value annotations
create_heatmaps(combined_data, 'All Models')

# Function to create heatmaps for subsets of the data (based on tissue or prefix)
def create_subset_heatmaps(threshold_data, group_name, subset_labels):
    for threshold, data in threshold_data.items():
        plt.figure(figsize=(24, 5))  # Adjust size for fewer rows
        
        # Filter the data to only include the rows for the selected subset
        subset_data = data.loc[subset_labels]
        
        # Save the original data for annotation
        original_data = subset_data.copy()
        
        # Normalize the data per column (across each model)
        normalized_data = subset_data.div(subset_data.max(axis=0), axis=1)
        
        # Create the heatmap using normalized data for coloring
        ax = sns.heatmap(normalized_data, annot=original_data, fmt='.2f', cmap='plasma', cbar=True)
        
        # Adjust the color bar to reflect the range of original values
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Adjust these ticks based on normalization scale
        cbar.set_ticklabels([f'{original_data.min().min():.2f}', 
                             f'{original_data.mean().min():.2f}', 
                             f'{original_data.mean().mean():.2f}', 
                             f'{original_data.max().min():.2f}', 
                             f'{original_data.max().max():.2f}'])  # Format to 2 decimal places
        
        plt.title(f'Heatmap for {group_name} at {threshold}', fontsize=16)
        plt.xlabel('Models', fontsize=14)
        
        # Rotate Y-axis labels (tissue/type) to be horizontal
        plt.yticks(rotation=0, fontsize=12)
        plt.xticks(rotation=90, fontsize=12)  # Keep the X-axis labels rotated vertically for readability
        
        plt.show()

# Define subsets for each heatmap group

# Subsets based on tissue types
bld_datasets = ['bld spt', 'bld igg', 'bld pd1']
pln_datasets = ['pln spt', 'pln igg', 'pln pd1']
panc_datasets = ['panc spt', 'panc igg', 'panc pd1']

# Subsets based on dataset prefixes (igg, pd1, spt)
igg_datasets = ['pln igg', 'bld igg', 'panc igg']
pd1_datasets = ['pln pd1', 'bld pd1', 'panc pd1']
spt_datasets = ['pln spt', 'bld spt', 'panc spt']

# Create heatmaps for the different tissue types
create_subset_heatmaps(combined_data, 'Blood Datasets', bld_datasets)
create_subset_heatmaps(combined_data, 'PLN Datasets', pln_datasets)
create_subset_heatmaps(combined_data, 'Panc Datasets', panc_datasets)

# Create heatmaps for the different dataset prefixes
create_subset_heatmaps(combined_data, 'IGG Datasets', igg_datasets)
create_subset_heatmaps(combined_data, 'PD1 Datasets', pd1_datasets)
create_subset_heatmaps(combined_data, 'SPT Datasets', spt_datasets)


