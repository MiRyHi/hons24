# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:49:28 2024

@author: Mitch
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

# ==================== IMPORT DATA FOR TP8 ==============================

# Define the directory for the tp8 metrics CSV files within the current working directory
tp8_metrics_dir = os.path.join(os.getcwd(), "tp8_metrics")
tp8_files = [
    "tp8_metrics_50.csv", 
    "tp8_metrics_60.csv", 
    "tp8_metrics_70.csv", 
    "tp8_metrics_80.csv", 
    "tp8_metrics_90.csv"
]

# Load the CSV files and store them in a dictionary without modifying model names
tp8_data = {}
for file in tp8_files:
    threshold = file.split("_")[-1].split(".")[0]  # Extract threshold from filename
    df = pd.read_csv(os.path.join(tp8_metrics_dir, file))
    tp8_data[threshold] = df  # Store dataframe in dictionary under the threshold key

# Define thresholds
thresholds = ["50", "60", "70", "80", "90"]


# ================== CREATE F1, PRECISION, RECALL, AND F0.5 MATRICES ======================

# List of all models (assuming consistent model names across files)
models = list(tp8_data["50"]["model name"])

# Initialize DataFrames to store F1, Precision, Recall, F0.5, and Accuracy scores for each model across thresholds
f1_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
precision_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
recall_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
f0_5_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
accuracy_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
accuracy_scores_matrix = pd.DataFrame(index=models, columns=thresholds)

# Populate matrices for tp8
for threshold in thresholds:
    df = tp8_data[threshold]
    for model in models:
        f1_score = df.loc[df["model name"] == model, "F1 score"].values[0]
        precision_score = df.loc[df["model name"] == model, "precision"].values[0]
        recall_score = df.loc[df["model name"] == model, "recall"].values[0]
        f0_5_score = df.loc[df["model name"] == model, "F0.5 score"].values[0]
        accuracy_score = df.loc[df["model name"] == model, "accuracy"].values[0]  # Assumes column name "accuracy"

        f1_scores_matrix.loc[model, threshold] = f1_score
        precision_scores_matrix.loc[model, threshold] = precision_score
        recall_scores_matrix.loc[model, threshold] = recall_score
        f0_5_scores_matrix.loc[model, threshold] = f0_5_score
        accuracy_scores_matrix.loc[model, threshold] = accuracy_score  # Populate accuracy matrix

# Ensure all values are numeric
f1_scores_matrix = f1_scores_matrix.apply(pd.to_numeric, errors='coerce')
precision_scores_matrix = precision_scores_matrix.apply(pd.to_numeric, errors='coerce')
recall_scores_matrix = recall_scores_matrix.apply(pd.to_numeric, errors='coerce')
f0_5_scores_matrix = f0_5_scores_matrix.apply(pd.to_numeric, errors='coerce')
accuracy_scores_matrix = accuracy_scores_matrix.apply(pd.to_numeric, errors='coerce')

# ======================== SELECT TOP 10 MODELS PER THRESHOLD FOR F1 AND F0.5 =========

# Dictionary to store top 10 models for each threshold based on F1 score
top_models_by_f1_threshold = {}
for threshold, df in tp8_data.items():
    top_10_f1 = df.sort_values(by="F1 score", ascending=False).head(10)
    top_models_by_f1_threshold[threshold] = top_10_f1

# Dictionary to store top 10 models for each threshold based on F0.5 score
top_models_by_f0_5_threshold = {}
for threshold, df in tp8_data.items():
    top_10_f0_5 = df.sort_values(by="F0.5 score", ascending=False).head(10)
    top_models_by_f0_5_threshold[threshold] = top_10_f0_5

# Define a color palette for consistent coloring
color_palette_group = sns.color_palette("turbo", 10)

# ==================== PLOT FUNCTION FOR F1, F0.5, PRECISION, AND RECALL ============================

# Ensure the data in all matrices are correctly indexed by string thresholds
f1_scores_matrix.columns = thresholds
precision_scores_matrix.columns = thresholds
recall_scores_matrix.columns = thresholds
f0_5_scores_matrix.columns = thresholds
accuracy_scores_matrix.columns = thresholds

def plot_scores_with_highlighted_group(group, threshold, scores_matrix, selection_criterion, score_type="F1 Score"):
    plt.figure(figsize=(10, 7), dpi=900)

    # Convert thresholds to integers for plotting
    numeric_thresholds = [int(t) for t in thresholds]

    # Plot all models in faint gray for context
    for model in scores_matrix.index:
        plt.plot(numeric_thresholds, scores_matrix.loc[model, thresholds], color="gray", alpha=0.5, linewidth=0.8)

    # Highlight the top 10 models for the given threshold
    custom_legend_handles = []
    for i, model in enumerate(group["model name"]):
        plt.plot(numeric_thresholds, scores_matrix.loc[model, thresholds], color=color_palette_group[i], linewidth=1.2)
        for j, threshold_val in enumerate(numeric_thresholds):
            alpha_value = 1.0 if threshold_val == int(threshold) else 0.0
            plt.scatter(threshold_val, scores_matrix.loc[model, str(threshold_val)], 
                        color=color_palette_group[i], alpha=alpha_value, zorder=3)
        
        # Create a custom legend entry
        custom_legend_handles.append(Line2D([0], [0], color=color_palette_group[i], lw=1.5, marker='o', 
                                            markerfacecolor=color_palette_group[i], label=model))

    # Title and labels
    plt.title(f"Tp8 Models Selected by {selection_criterion} at Threshold {threshold} - {score_type}", 
              fontsize=20, pad=20, loc='center', x=0.65)
    plt.xlabel("Threshold", fontsize=17, labelpad=15)
    plt.ylabel(score_type, fontsize=17, labelpad=15)
    plt.xticks(numeric_thresholds, fontsize=14)  
    plt.yticks(fontsize=14)
    
    # Set X and Y limits for consistency
    plt.xlim(49.5, 90.5)
    plt.ylim(0.0, 1.0)  # Adjust Y-axis limit based on data range

    # Add legend
    plt.legend(handles=custom_legend_handles, 
               title=f"Top 10 Models (t{threshold})", 
               bbox_to_anchor=(1.01, 1.015), loc="upper left", 
               fontsize=12, title_fontsize=14, frameon=True, 
               edgecolor="gray", framealpha=0.9, 
               borderpad=1, labelspacing=0.6)

    plt.grid(True, which='both', axis='both', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Generate plots for top 10 models selected by F1 scores across all metrics
for threshold, top_10_df in top_models_by_f1_threshold.items():
    plot_scores_with_highlighted_group(top_10_df, threshold, f1_scores_matrix, "F1 Score", "F1 Score")
    plot_scores_with_highlighted_group(top_10_df, threshold, precision_scores_matrix, "F1 Score", "Precision")
    plot_scores_with_highlighted_group(top_10_df, threshold, recall_scores_matrix, "F1 Score", "Recall")
    plot_scores_with_highlighted_group(top_10_df, threshold, accuracy_scores_matrix, "F1 Score", "Accuracy")  # Added accuracy

# Generate plots for top 10 models selected by F0.5 scores across all metrics for tp8
for threshold, top_10_df in top_models_by_f0_5_threshold.items():
    plot_scores_with_highlighted_group(top_10_df, threshold, f0_5_scores_matrix, "F0.5 Score", "F0.5 Score")
    plot_scores_with_highlighted_group(top_10_df, threshold, precision_scores_matrix, "F0.5 Score", "Precision")
    plot_scores_with_highlighted_group(top_10_df, threshold, recall_scores_matrix, "F0.5 Score", "Recall")
    plot_scores_with_highlighted_group(top_10_df, threshold, accuracy_scores_matrix, "F0.5 Score", "Accuracy")  # Added accuracy







# ==================== IMPORT DATA FOR TP14 ==============================

# Define the directory for the tp14 metrics CSV files within the current working directory
tp14_metrics_dir = os.path.join(os.getcwd(), "tp14_metrics")
tp14_files = [
    "tp14_metrics_50.csv", 
    "tp14_metrics_60.csv", 
    "tp14_metrics_70.csv", 
    "tp14_metrics_80.csv", 
    "tp14_metrics_90.csv"
]

# Load the CSV files and store them in a dictionary without modifying model names
tp14_data = {}
for file in tp14_files:
    threshold = file.split("_")[-1].split(".")[0]  # Extract threshold from filename
    df = pd.read_csv(os.path.join(tp14_metrics_dir, file))
    tp14_data[threshold] = df  # Store dataframe in dictionary under the threshold key


# ================== CREATE F1, PRECISION, RECALL, AND F0.5 MATRICES FOR TP14 ======================

# List of all models (assuming consistent model names across files)
models_tp14 = list(tp14_data["50"]["model name"])

# Initialize DataFrames to store F1, Precision, Recall, and F0.5 scores for each model across thresholds for tp14
f1_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
precision_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
recall_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
f0_5_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
accuracy_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)

# Populate matrices for tp14
for threshold in thresholds:
    df = tp14_data[threshold]
    for model in models_tp14:
        f1_score = df.loc[df["model name"] == model, "F1 score"].values[0]
        precision_score = df.loc[df["model name"] == model, "precision"].values[0]
        recall_score = df.loc[df["model name"] == model, "recall"].values[0]
        f0_5_score = df.loc[df["model name"] == model, "F0.5 score"].values[0]
        accuracy_score = df.loc[df["model name"] == model, "accuracy"].values[0]  # Assumes column name "accuracy"

        f1_scores_matrix_tp14.loc[model, threshold] = f1_score
        precision_scores_matrix_tp14.loc[model, threshold] = precision_score
        recall_scores_matrix_tp14.loc[model, threshold] = recall_score
        f0_5_scores_matrix_tp14.loc[model, threshold] = f0_5_score
        accuracy_scores_matrix_tp14.loc[model, threshold] = accuracy_score
        
# Ensure all values are numeric
f1_scores_matrix_tp14 = f1_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
precision_scores_matrix_tp14 = precision_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
recall_scores_matrix_tp14 = recall_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
f0_5_scores_matrix_tp14 = f0_5_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
accuracy_scores_matrix_tp14 = accuracy_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')


# ======================== SELECT TOP 10 MODELS PER THRESHOLD FOR TP14 F1 AND F0.5 =========

# Dictionary to store top 10 models for each threshold based on F1 score for tp14
top_models_by_f1_threshold_tp14 = {}
for threshold, df in tp14_data.items():
    top_10_f1 = df.sort_values(by="F1 score", ascending=False).head(10)
    top_models_by_f1_threshold_tp14[threshold] = top_10_f1

# Dictionary to store top 10 models for each threshold based on F0.5 score for tp14
top_models_by_f0_5_threshold_tp14 = {}
for threshold, df in tp14_data.items():
    top_10_f0_5 = df.sort_values(by="F0.5 score", ascending=False).head(10)
    top_models_by_f0_5_threshold_tp14[threshold] = top_10_f0_5

# ==================== PLOT FUNCTION FOR TP14 F1, F0.5, PRECISION, AND RECALL ============================

# Ensure the data in all matrices are correctly indexed by string thresholds for tp14
f1_scores_matrix_tp14.columns = thresholds
precision_scores_matrix_tp14.columns = thresholds
recall_scores_matrix_tp14.columns = thresholds
f0_5_scores_matrix_tp14.columns = thresholds
accuracy_scores_matrix_tp14.columns = thresholds

def plot_scores_with_highlighted_group_tp14(group, threshold, scores_matrix, selection_criterion, score_type="F1 Score"):
    plt.figure(figsize=(10, 7), dpi=900)

    # Convert thresholds to integers for plotting
    numeric_thresholds = [int(t) for t in thresholds]

    # Plot all models in faint gray for context
    for model in scores_matrix.index:
        plt.plot(numeric_thresholds, scores_matrix.loc[model, thresholds], color="gray", alpha=0.5, linewidth=0.8)

    # Highlight the top 10 models for the given threshold
    custom_legend_handles = []
    for i, model in enumerate(group["model name"]):
        plt.plot(numeric_thresholds, scores_matrix.loc[model, thresholds], color=color_palette_group[i], linewidth=1.2)
        for j, threshold_val in enumerate(numeric_thresholds):
            alpha_value = 1.0 if threshold_val == int(threshold) else 0.0
            plt.scatter(threshold_val, scores_matrix.loc[model, str(threshold_val)], 
                        color=color_palette_group[i], alpha=alpha_value, zorder=3)
        
        # Create a custom legend entry
        custom_legend_handles.append(Line2D([0], [0], color=color_palette_group[i], lw=1.5, marker='o', 
                                            markerfacecolor=color_palette_group[i], label=model))

    # Title and labels
    plt.title(f"Tp14 Models Selected by {selection_criterion} at Threshold {threshold} - {score_type}", 
              fontsize=20, pad=20, loc='center', x=0.65)
    plt.xlabel("Threshold", fontsize=17, labelpad=15)
    plt.ylabel(score_type, fontsize=17, labelpad=15)
    plt.xticks(numeric_thresholds, fontsize=14)  
    plt.yticks(fontsize=14)
    
    # Set X and Y limits for consistency
    plt.xlim(49.5, 90.5)
    plt.ylim(0.0, 1.0)  # Adjust Y-axis limit based on data range

    # Add legend
    plt.legend(handles=custom_legend_handles, 
               title=f"Top 10 Models (t{threshold})", 
               bbox_to_anchor=(1.01, 1.015), loc="upper left", 
               fontsize=12, title_fontsize=14, frameon=True, 
               edgecolor="gray", framealpha=0.9, 
               borderpad=1, labelspacing=0.6)

    plt.grid(True, which='both', axis='both', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Generate plots for top 10 models selected by F1 scores across all metrics for tp14
for threshold, top_10_df in top_models_by_f1_threshold_tp14.items():
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, f1_scores_matrix_tp14, "F1 Score", "F1 Score")
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, precision_scores_matrix_tp14, "F1 Score", "Precision")
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, recall_scores_matrix_tp14, "F1 Score", "Recall")
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, accuracy_scores_matrix_tp14, "F1 Score", "Accuracy")  # Added accuracy

# Generate plots for top 10 models selected by F0.5 scores across all metrics for tp14
for threshold, top_10_df in top_models_by_f0_5_threshold_tp14.items():
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, f0_5_scores_matrix_tp14, "F0.5 Score", "F0.5 Score")
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, precision_scores_matrix_tp14, "F0.5 Score", "Precision")
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, recall_scores_matrix_tp14, "F0.5 Score", "Recall")
    plot_scores_with_highlighted_group_tp14(top_10_df, threshold, accuracy_scores_matrix_tp14, "F0.5 Score", "Accuracy")  # Added accuracy



# =========================== PLOT HEATMAPS ====================================

def plot_experimental_heatmap(top_models_by_threshold, threshold_key, selection_criterion, experiment_type, row_normalize=True):
    # Access the DataFrame for the current threshold and selection criterion
    df = top_models_by_threshold[threshold_key]
    
    # Identify and clean the columns with experimental data by removing the prefix
    column_patterns = ["pln igg", "bld igg", "panc igg", "pln spt", "bld spt", "panc spt"]
    columns_to_plot = [f"t{threshold_key} {pattern}" for pattern in column_patterns]
    clean_column_labels = column_patterns  # Cleaned labels without threshold prefixes

    # Select only the relevant columns for the heatmap
    heatmap_data = df[columns_to_plot].copy()
    heatmap_data.columns = clean_column_labels  # Rename columns for cleaner labels

    # Apply row-wise normalization if row_normalize is set to True
    if row_normalize:
        heatmap_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
    
    # Set up the figure
    plt.figure(figsize=(10, 8), dpi=600)
    
    # Plot the heatmap
    ax = sns.heatmap(
        heatmap_data,
        cmap="Spectral",
        annot=df[columns_to_plot],  # Show actual values
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Fractional Value'}
    )
    
    # Adjust color bar to display correctly based on normalization
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 0.5, 1])  # Set ticks at min, mid, max positions
    colorbar.set_ticklabels(["Min", "Mid", "Max"])  # Indicate general levels

    # Set full model names as y-tick labels
    ax.set_yticklabels(df["model name"], rotation=0, ha='right')
    
    # Title and labels
    plt.title(f"{experiment_type.upper()} Models Selected by {selection_criterion} Score at Threshold {threshold_key}", fontsize=18, pad=20)
    plt.xlabel("Experimental Data Sources", fontsize=15, labelpad=15)
    plt.ylabel("Model Name", fontsize=15, labelpad=15)
    
    # Show plot
    plt.tight_layout()
    plt.show()

# ================== PLOT HEATMAPS ======================

# -------------- ROW NORMALISE ON --------------------
# Call the function for each threshold individually
for threshold_key in thresholds:
    # Pass only the data for the current threshold, not the entire dictionary
    plot_experimental_heatmap({threshold_key: top_models_by_f1_threshold[threshold_key]}, threshold_key, selection_criterion="F1", experiment_type="tp8", row_normalize=True)

# Loop through tp8 models selected by F0.5 score for each threshold
for threshold_key in thresholds:
    plot_experimental_heatmap({threshold_key: top_models_by_f0_5_threshold[threshold_key]}, threshold_key, selection_criterion="F0.5", experiment_type="tp8", row_normalize=True)

# Loop through tp14 models selected by F1 score for each threshold
for threshold_key in thresholds:
    plot_experimental_heatmap({threshold_key: top_models_by_f1_threshold_tp14[threshold_key]}, threshold_key, selection_criterion="F1", experiment_type="tp14", row_normalize=True)

# Loop through tp14 models selected by F0.5 score for each threshold
for threshold_key in thresholds:
    plot_experimental_heatmap({threshold_key: top_models_by_f0_5_threshold_tp14[threshold_key]}, threshold_key, selection_criterion="F0.5", experiment_type="tp14", row_normalize=True)


# ------------- ROW NORMALISE OFF -------------------
# Call the function for each threshold individually
for threshold_key in thresholds:
    # Pass only the data for the current threshold, not the entire dictionary
    plot_experimental_heatmap({threshold_key: top_models_by_f1_threshold[threshold_key]}, threshold_key, selection_criterion="F1", experiment_type="tp8", row_normalize=False)

# Loop through tp8 models selected by F0.5 score for each threshold
for threshold_key in thresholds:
    plot_experimental_heatmap({threshold_key: top_models_by_f0_5_threshold[threshold_key]}, threshold_key, selection_criterion="F0.5", experiment_type="tp8", row_normalize=False)

# Loop through tp14 models selected by F1 score for each threshold
for threshold_key in thresholds:
    plot_experimental_heatmap({threshold_key: top_models_by_f1_threshold_tp14[threshold_key]}, threshold_key, selection_criterion="F1", experiment_type="tp14", row_normalize=False)

# Loop through tp14 models selected by F0.5 score for each threshold
for threshold_key in thresholds:
    plot_experimental_heatmap({threshold_key: top_models_by_f0_5_threshold_tp14[threshold_key]}, threshold_key, selection_criterion="F0.5", experiment_type="tp14", row_normalize=False)




























# ====================== EXTRACT EXPERIMENTAL DATA =======================

# Dictionary to store extracted experimental data for each threshold
experimental_data_by_threshold = {}

# Define the column name patterns for the 9 experimental values
column_patterns = [
    "bld igg", "bld pd1", "bld spt",
    "pln igg", "pln pd1", "pln spt",
    "panc igg", "panc pd1", "panc spt"
]











# Loop through each threshold and extract data for top 10 models
for threshold, top_10_df in top_models_by_threshold.items():
    # Initialize a list to store experimental data for this threshold's top 10 models
    experimental_data = []

    # Access the relevant dataframe from tp8_data
    df = tp8_data[threshold]

    # Loop through each model in the top 10
    for model_name in top_10_df["model name"]:
        # Extract the 9 experimental values for this model
        model_data = {"model_name": model_name}
        for pattern in column_patterns:
            column_name = f"t{threshold} {pattern}"  # Construct the full column name based on threshold and pattern
            if column_name in df.columns:
                model_data[column_name] = df.loc[df["model name"] == model_name, column_name].values[0]
            else:
                model_data[column_name] = None  # If column is missing, store None

        # Append this model's data to the list
        experimental_data.append(model_data)

    # Store this threshold's experimental data in the main dictionary
    experimental_data_by_threshold[threshold] = pd.DataFrame(experimental_data)

# Check the structure of the extracted data for one threshold, e.g., threshold 50
print(experimental_data_by_threshold["50"].head())

# ==================== PLOT HEATMAPS FOR EXPERIMENTAL DATA ===================

# Set up parameters for the heatmaps
plt.figure(figsize=(12, 10))  # Set figure size to accommodate all heatmaps

# Define the new column order to match the requested arrangement
column_order = [
    "t50 panc spt", "t50 bld spt", "t50 pln spt",
    "t50 panc igg", "t50 bld igg", "t50 pln igg"
]


# Toggle for normalizing data within each column
normalize_data = True  # Set to True to enable normalization, False to disable

# Select a perceptually uniform colormap 
chosen_cmap = "turbo"  # Change this to any preferred colormap like "plasma", "viridis", etc.

# Loop through each threshold to plot its heatmap
for threshold in thresholds:
    data = experimental_data_by_threshold[threshold]
    
    # Adjust column names for the current threshold in the specified order
    ordered_columns = [col.replace("50", threshold) for col in column_order]
    heatmap_data = data.set_index("model_name")[ordered_columns].astype(float)
    
    # Transpose the data so that models are columns and experimental sources are rows
    heatmap_data_transposed = heatmap_data.T

    # Apply normalization if enabled
    if normalize_data:
        heatmap_data_display = heatmap_data_transposed.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    else:
        heatmap_data_display = heatmap_data_transposed

    # Set up the figure for each threshold
    plt.figure(figsize=(12, 8), dpi=600)
    
    # Plot with the chosen perceptually uniform colormap
    ax = sns.heatmap(
        heatmap_data_display,
        cmap=chosen_cmap,  # Use chosen colormap here
        annot=heatmap_data_transposed,  # Always show actual values regardless of normalization
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={}  # Leave empty to avoid an automatic color bar title
    )
    
    # Adjust color bar settings
    colorbar = ax.collections[0].colorbar
    colorbar.ax.set_aspect(30)  # Adjust color bar height
    
    # Set min and max for color bar based on raw values
    min_val, max_val = heatmap_data_transposed.min().min(), heatmap_data_transposed.max().max()
    colorbar.set_ticks([0, 0.5, 1])  # Position at start, middle, and end
    colorbar.set_ticklabels([f"{min_val:.3f}", f"{(min_val + max_val) / 2:.3f}", f"{max_val:.3f}"])  # Display actual values

    # Rotate row labels to horizontal
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    # Title and labels
    plt.title(f"Experimental Data Heatmap for Models Selected at Threshold {threshold}", fontsize=18, pad=20)
    plt.xlabel("Model Name", fontsize=15, labelpad=15)
    plt.ylabel("Experimental Data Sources", fontsize=15, labelpad=15)

    # Show plot
    plt.tight_layout()
    plt.show()