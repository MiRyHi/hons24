
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

# Define the directory for the tp14 metrics CSV files within the current working directory
tp14_metrics_dir = os.path.join(os.getcwd(), "tp14_metrics")
tp14_files = [
    "tp14_metrics_50.csv", 
    "tp14_metrics_60.csv", 
    "tp14_metrics_70.csv", 
    "tp14_metrics_80.csv", 
    "tp14_metrics_90.csv"
]


# Define thresholds
thresholds = ["50", "60", "70", "80", "90"]

# Load the CSV files and store them in a dictionary without modifying model names
tp8_data = {}
for file in tp8_files:
    threshold = file.split("_")[-1].split(".")[0]  # Extract threshold from filename
    df = pd.read_csv(os.path.join(tp8_metrics_dir, file))
    tp8_data[threshold] = df  # Store dataframe in dictionary under the threshold key

# Load the CSV files and store them in a dictionary without modifying model names
tp14_data = {}
for file in tp14_files:
    threshold = file.split("_")[-1].split(".")[0]  
    df = pd.read_csv(os.path.join(tp14_metrics_dir, file))
    tp14_data[threshold] = df  
    

# ================== CREATE F0.75, PRECISION, RECALL, F0.5, AND ACCURACY MATRICES ======================

# List of all models (assuming consistent model names across files)
models = list(tp8_data["50"]["model name"])

# List of all models (assuming consistent model names across files)
models_tp14 = list(tp14_data["50"]["model name"])

# Initialize DataFrames to store F0.75, Precision, Recall, F0.5, and Accuracy scores for each model across thresholds
f0_75_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
precision_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
recall_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
f0_5_scores_matrix = pd.DataFrame(index=models, columns=thresholds)
accuracy_scores_matrix = pd.DataFrame(index=models, columns=thresholds)

# Initialize DataFrames to store F0.75, Precision, Recall, F0.5, and Accuracy scores for each model across thresholds for tp14
f0_75_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
precision_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
recall_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
f0_5_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)
accuracy_scores_matrix_tp14 = pd.DataFrame(index=models_tp14, columns=thresholds)

# Populate matrices for tp8
for threshold in thresholds:
    df = tp8_data[threshold]
    for model in models:
        f0_75_score = df.loc[df["model name"] == model, "F0.75 score"].values[0]
        precision_score = df.loc[df["model name"] == model, "precision"].values[0]
        recall_score = df.loc[df["model name"] == model, "recall"].values[0]
        f0_5_score = df.loc[df["model name"] == model, "F0.5 score"].values[0]
        accuracy_score = df.loc[df["model name"] == model, "accuracy"].values[0]

        f0_75_scores_matrix.loc[model, threshold] = f0_75_score
        precision_scores_matrix.loc[model, threshold] = precision_score
        recall_scores_matrix.loc[model, threshold] = recall_score
        f0_5_scores_matrix.loc[model, threshold] = f0_5_score
        accuracy_scores_matrix.loc[model, threshold] = accuracy_score

# Ensure all values are numeric
f0_75_scores_matrix = f0_75_scores_matrix.apply(pd.to_numeric, errors='coerce')
precision_scores_matrix = precision_scores_matrix.apply(pd.to_numeric, errors='coerce')
recall_scores_matrix = recall_scores_matrix.apply(pd.to_numeric, errors='coerce')
f0_5_scores_matrix = f0_5_scores_matrix.apply(pd.to_numeric, errors='coerce')
accuracy_scores_matrix = accuracy_scores_matrix.apply(pd.to_numeric, errors='coerce')

# Populate matrices for tp14
for threshold in thresholds:
    df = tp14_data[threshold]
    for model in models_tp14:
        f0_75_score = df.loc[df["model name"] == model, "F0.75 score"].values[0]
        precision_score = df.loc[df["model name"] == model, "precision"].values[0]
        recall_score = df.loc[df["model name"] == model, "recall"].values[0]
        f0_5_score = df.loc[df["model name"] == model, "F0.5 score"].values[0]
        accuracy_score = df.loc[df["model name"] == model, "accuracy"].values[0]

        f0_75_scores_matrix_tp14.loc[model, threshold] = f0_75_score
        precision_scores_matrix_tp14.loc[model, threshold] = precision_score
        recall_scores_matrix_tp14.loc[model, threshold] = recall_score
        f0_5_scores_matrix_tp14.loc[model, threshold] = f0_5_score
        accuracy_scores_matrix_tp14.loc[model, threshold] = accuracy_score
        
# Ensure all values are numeric
f0_75_scores_matrix_tp14 = f0_75_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
precision_scores_matrix_tp14 = precision_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
recall_scores_matrix_tp14 = recall_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
f0_5_scores_matrix_tp14 = f0_5_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')
accuracy_scores_matrix_tp14 = accuracy_scores_matrix_tp14.apply(pd.to_numeric, errors='coerce')

# ======================== SELECT TOP 10 MODELS PER THRESHOLD FOR F0.75 AND F0.5 =========

# Dictionary to store top 10 models for each threshold based on F0.75 score
top_models_by_f0_75_threshold = {}
for threshold, df in tp8_data.items():
    top_10_f0_75 = df.sort_values(by="F0.75 score", ascending=False).head(10)
    top_models_by_f0_75_threshold[threshold] = top_10_f0_75

# Dictionary to store top 10 models for each threshold based on F0.5 score
top_models_by_f0_5_threshold = {}
for threshold, df in tp8_data.items():
    top_10_f0_5 = df.sort_values(by="F0.5 score", ascending=False).head(10)
    top_models_by_f0_5_threshold[threshold] = top_10_f0_5

# Define a color palette for consistent coloring
color_palette_group = sns.color_palette("tab10", 10)

# ==================== PLOT FUNCTION FOR F0.75, F0.5, PRECISION, RECALL, AND ACCURACY ============================

# Ensure the data in all matrices are correctly indexed by string thresholds
f0_75_scores_matrix.columns = thresholds
precision_scores_matrix.columns = thresholds
recall_scores_matrix.columns = thresholds
f0_5_scores_matrix.columns = thresholds
accuracy_scores_matrix.columns = thresholds

def plot_scores_with_highlighted_group(group, threshold, scores_matrix, selection_criterion, experiment_type, score_type="F0.75 Score"):
    plt.figure(figsize=(10, 6), dpi=600)

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
    plt.title(f"{score_type} of Top 10 tp8 Models by {selection_criterion} (T{threshold}) Across Thresholds", 
              fontsize=19, pad=20, loc='center', x=0.65)
    plt.xlabel("Threshold (T)", fontsize=15, labelpad=10)
    plt.ylabel(score_type, fontsize=15, labelpad=10)
    plt.xticks(numeric_thresholds, fontsize=12)  
    plt.yticks(fontsize=12)
    
    # Set X and Y limits for consistency
    plt.xlim(49.5, 90.5)
    plt.ylim(0.0, 1.0)  # Adjust Y-axis limit based on data range

    # Add legend
    plt.legend(handles=custom_legend_handles, 
               title=f"Top 10 Models (T{threshold})", 
               bbox_to_anchor=(1.005, 1.02), loc="upper left", 
               fontsize=12, title_fontsize=14, frameon=True, 
               edgecolor="gray", framealpha=0.9, 
               borderpad=1, labelspacing=0.6)

    plt.grid(True, which='both', axis='both', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Generate plots for top 10 models selected by F0.75 scores across all metrics
for threshold, top_10_df in top_models_by_f0_75_threshold.items():
    plot_scores_with_highlighted_group(top_10_df, threshold, f0_75_scores_matrix, "F0.75 Score", "F0.75 Score")
    plot_scores_with_highlighted_group(top_10_df, threshold, precision_scores_matrix, "F0.75 Score", "Precision")
    plot_scores_with_highlighted_group(top_10_df, threshold, recall_scores_matrix, "F0.75 Score", "Recall")
    plot_scores_with_highlighted_group(top_10_df, threshold, accuracy_scores_matrix, "F0.75 Score", "Accuracy")

# Generate plots for top 10 models selected by F0.5 scores across all metrics for tp8
for threshold, top_10_df in top_models_by_f0_5_threshold.items():
    plot_scores_with_highlighted_group(top_10_df, threshold, f0_5_scores_matrix, "F0.5 Score", "F0.5 Score")
    plot_scores_with_highlighted_group(top_10_df, threshold, precision_scores_matrix, "F0.5 Score", "Precision")
    plot_scores_with_highlighted_group(top_10_df, threshold, recall_scores_matrix, "F0.5 Score", "Recall")
    plot_scores_with_highlighted_group(top_10_df, threshold, accuracy_scores_matrix, "F0.5 Score", "Accuracy")
  
# ==================== PLOT FUNCTION FOR TP14 F0.75, F0.5, PRECISION, RECALL, AND ACCURACY ============================

def plot_scores_with_highlighted_group(group, threshold, scores_matrix, selection_criterion, experiment_type, score_type="F0.75 Score"):
    plt.figure(figsize=(10, 6), dpi=600)

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
    plt.title(f"{score_type} of Top 10 tp14 Models by {selection_criterion} (T{threshold}) Across Thresholds", 
              fontsize=19, pad=20, loc='center', x=0.65)
    plt.xlabel("Threshold (T)", fontsize=15, labelpad=10)
    plt.ylabel(score_type, fontsize=15, labelpad=10)
    plt.xticks(numeric_thresholds, fontsize=12)  
    plt.yticks(fontsize=12)
    
    # Set X and Y limits for consistency
    plt.xlim(49.5, 90.5)
    plt.ylim(0.0, 1.0)  # Adjust Y-axis limit based on data range

    # Add legend
    plt.legend(handles=custom_legend_handles, 
               title=f"Top 10 Models (T{threshold})", 
               bbox_to_anchor=(1.005, 1.02), loc="upper left", 
               fontsize=12, title_fontsize=14, frameon=True, 
               edgecolor="gray", framealpha=0.9, 
               borderpad=1, labelspacing=0.6)

    plt.grid(True, which='both', axis='both', alpha=0.6)
    plt.tight_layout()
    plt.show()
  
# Dictionary to store top 10 models for each threshold based on F0.75 score for tp14
top_models_by_f0_75_threshold_tp14 = {}
for threshold, df in tp14_data.items():
    top_10_f0_75 = df.sort_values(by="F0.75 score", ascending=False).head(10)
    top_models_by_f0_75_threshold_tp14[threshold] = top_10_f0_75

# Dictionary to store top 10 models for each threshold based on F0.5 score for tp14
top_models_by_f0_5_threshold_tp14 = {}
for threshold, df in tp14_data.items():
    top_10_f0_5 = df.sort_values(by="F0.5 score", ascending=False).head(10)
    top_models_by_f0_5_threshold_tp14[threshold] = top_10_f0_5

for threshold, top_10_df in top_models_by_f0_75_threshold_tp14.items():
    plot_scores_with_highlighted_group(top_10_df, threshold, f0_75_scores_matrix_tp14, "F0.75", "tp14", "F0.75 Score")
    plot_scores_with_highlighted_group(top_10_df, threshold, precision_scores_matrix_tp14, "F0.75", "tp14", "Precision")
    plot_scores_with_highlighted_group(top_10_df, threshold, recall_scores_matrix_tp14, "F0.75", "tp14", "Recall")
    plot_scores_with_highlighted_group(top_10_df, threshold, accuracy_scores_matrix_tp14, "F0.75", "tp14", "Accuracy")

for threshold, top_10_df in top_models_by_f0_5_threshold_tp14.items():
    plot_scores_with_highlighted_group(top_10_df, threshold, f0_5_scores_matrix_tp14, "F0.5", "tp14", "F0.5 Score")
    plot_scores_with_highlighted_group(top_10_df, threshold, precision_scores_matrix_tp14, "F0.5", "tp14", "Precision")
    plot_scores_with_highlighted_group(top_10_df, threshold, recall_scores_matrix_tp14, "F0.5", "tp14", "Recall")
    plot_scores_with_highlighted_group(top_10_df, threshold, accuracy_scores_matrix_tp14, "F0.5", "tp14", "Accuracy")

def calculate_mean_scores(data, metric):
    models = list(data["50"]["model name"])
    scores_matrix = pd.DataFrame(index=models, columns=thresholds)
    
    for threshold in thresholds:
        df = data[threshold]
        for model in models:
            score = df.loc[df["model name"] == model, metric].values[0]
            scores_matrix.loc[model, threshold] = score
    
    # Convert to numeric and add a mean column
    scores_matrix = scores_matrix.apply(pd.to_numeric, errors='coerce')
    scores_matrix["Mean"] = scores_matrix.mean(axis=1)
        
    return scores_matrix

# =========================== PLOT HEATMAPS ====================================

def plot_experimental_heatmap(top_models_by_threshold, threshold_key, selection_criterion, experiment_type, row_normalize=False):
    df = top_models_by_threshold[threshold_key]
    
    # Column patterns with underscores and lowercase format
    column_patterns = ["pln_igg", "bld_igg", "panc_igg", "pln_spt", "bld_spt", "panc_spt"]
    columns_to_plot = [f"t{threshold_key}_{pattern}" for pattern in column_patterns]
    
    available_columns = [col for col in columns_to_plot if col in df.columns]
    if not available_columns:
        print(f"No experimental data columns found for threshold {threshold_key}. Available columns: {df.columns}")
        return
    
    heatmap_data = df[available_columns].copy()
    heatmap_data.columns = [col.split('_', 1)[1] for col in available_columns]

    # Determine global min and max for color scaling
    vmin, vmax = heatmap_data.min().min(), heatmap_data.max().max()

    plt.figure(figsize=(10, 8), dpi=600)
    ax = sns.heatmap(
        heatmap_data, cmap="Spectral",
        annot=heatmap_data,  # Show non-normalized values
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Absolute Value'},
        vmin=vmin, vmax=vmax  # Use global min and max for consistent scaling
    )
    
    # Set full model names as y-tick labels
    ax.set_yticklabels(df["model name"], rotation=0, ha='right')
    plt.title(f"{experiment_type.lower()} Models Selected by {selection_criterion} Score at T{threshold_key}", fontsize=18, pad=20)
    plt.xlabel("Experimental Data Sources", fontsize=15, labelpad=15)
    plt.ylabel("Model Name", fontsize=15, labelpad=15)
    plt.tight_layout()
    plt.show()


# Calculate mean F0.75 and F0.5 for tp8
f0_75_scores_matrix_tp8 = calculate_mean_scores(tp8_data, "F0.75 score")
f0_5_scores_matrix_tp8 = calculate_mean_scores(tp8_data, "F0.5 score")

# Calculate mean F0.75 and F0.5 for tp14
f0_75_scores_matrix_tp14 = calculate_mean_scores(tp14_data, "F0.75 score")
f0_5_scores_matrix_tp14 = calculate_mean_scores(tp14_data, "F0.5 score")

# ================= Select Top 10 Models =================

def select_top_10_models_by_mean(scores_matrix):
    return scores_matrix.sort_values(by="Mean", ascending=False).head(10)

# Top 10 models by mean F0.75 and F0.5 for tp8
top_10_models_by_f0_75_mean_tp8 = select_top_10_models_by_mean(f0_75_scores_matrix_tp8)
top_10_models_by_f0_5_mean_tp8 = select_top_10_models_by_mean(f0_5_scores_matrix_tp8)

# Top 10 models by mean F0.75 and F0.5 for tp14
top_10_models_by_f0_75_mean_tp14 = select_top_10_models_by_mean(f0_75_scores_matrix_tp14)
top_10_models_by_f0_5_mean_tp14 = select_top_10_models_by_mean(f0_5_scores_matrix_tp14)

# Ensure precision and recall matrices have all five thresholds as columns
thresholds = ["50", "60", "70", "80", "90"]
precision_scores_matrix = precision_scores_matrix[thresholds]
recall_scores_matrix = recall_scores_matrix[thresholds]
precision_scores_matrix_tp14 = precision_scores_matrix_tp14[thresholds]
recall_scores_matrix_tp14 = recall_scores_matrix_tp14[thresholds]

# ================== PRECISION-RECALL LINE PLOTS FOR MEAN-SELECTED MODELS ======================

# Consistent function to plot scores with highlighted models
def plot_scores_with_highlighted_group(group, threshold, scores_matrix, selection_criterion, score_type="F0.75 Score"):
    plt.figure(figsize=(10, 6), dpi=600)

    # Convert thresholds to integers for plotting
    numeric_thresholds = [int(t) for t in thresholds]

    # Plot all models in faint gray for context
    for model in scores_matrix.index:
        plt.plot(numeric_thresholds, scores_matrix.loc[model, thresholds], color="gray", alpha=0.5, linewidth=0.8)

    # Highlight the selected models
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
    plt.title(f"Models Selected by {selection_criterion} at T{threshold} - {score_type}", 
              fontsize=19, pad=20, loc='center', x=0.65)
    plt.xlabel("Threshold", fontsize=15, labelpad=10)
    plt.ylabel(score_type, fontsize=15, labelpad=10)
    plt.xticks(numeric_thresholds, fontsize=12)  
    plt.yticks(fontsize=12)

    # Add legend
    plt.legend(handles=custom_legend_handles, 
               title=f"Top 10 Models (T{threshold})", 
               bbox_to_anchor=(1.005, 1.02), loc="upper left", 
               fontsize=12, title_fontsize=14, frameon=True, 
               edgecolor="gray", framealpha=0.9, 
               borderpad=1, labelspacing=0.6)

    plt.grid(True, which='both', axis='both', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ================== GENERATE PRECISION-RECALL PLOTS ======================
def plot_precision_recall_curves(models, precision_matrix, recall_matrix, experiment_type, selection_metric):
    """
    Plots precision-recall lines for the selected top models across thresholds.

    Parameters:
    - models: DataFrame of selected top models
    - precision_matrix: DataFrame containing precision values for each model across thresholds
    - recall_matrix: DataFrame containing recall values for each model across thresholds
    - experiment_type: "tp8" or "tp14" indicating the experiment type
    - selection_metric: "F0.5" or "F0.75" indicating the selection criterion
    """
    plt.figure(figsize=(10, 6), dpi=600)
    color_palette = sns.color_palette("tab10", len(models))

    # Plot precision-recall curves for each selected model
    for idx, model in enumerate(models.index):
        recall_values = recall_matrix.loc[model, thresholds]
        precision_values = precision_matrix.loc[model, thresholds]
        plt.plot(recall_values, precision_values, label=model, color=color_palette[idx], marker="o", markersize=5, linewidth=1.5)

    # Customize plot
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"Precision vs Recall for Top 10 {experiment_type.lower()} Models Selected by Mean {selection_metric}", fontsize=20, pad=20, loc='center', x=0.65)
    plt.legend(bbox_to_anchor=(1.005, 1.02), loc="upper left", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot for tp8 models selected by mean F0.5
plot_precision_recall_curves(top_10_models_by_f0_5_mean_tp8, precision_scores_matrix, recall_scores_matrix, "tp8", "F0.5")

# Plot for tp8 models selected by mean F0.75
plot_precision_recall_curves(top_10_models_by_f0_75_mean_tp8, precision_scores_matrix, recall_scores_matrix, "tp8", "F0.75")

# Plot for tp14 models selected by mean F0.5
plot_precision_recall_curves(top_10_models_by_f0_5_mean_tp14, precision_scores_matrix_tp14, recall_scores_matrix_tp14, "tp14", "F0.5")

# Plot for tp14 models selected by mean F0.75
plot_precision_recall_curves(top_10_models_by_f0_75_mean_tp14, precision_scores_matrix_tp14, recall_scores_matrix_tp14, "tp14", "F0.75")
    







# ================= Plot Function =================

def plot_scores_with_highlighted_group_mean(mean_selected_models, scores_matrix, thresholds, score_type, selection_metric, experiment_type):
    plt.figure(figsize=(10, 6), dpi=600)
    numeric_thresholds = [int(t) for t in thresholds]
    color_palette_group = sns.color_palette("tab10", 10)

    # Plot all models in faint gray
    for model in scores_matrix.index:
        plt.plot(numeric_thresholds, scores_matrix.loc[model, thresholds], color="gray", alpha=0.5, linewidth=0.8)

    # Highlight the selected models in color
    custom_legend_handles = []
    for i, model in enumerate(mean_selected_models.index):
        plt.plot(numeric_thresholds, scores_matrix.loc[model, thresholds], color=color_palette_group[i], linewidth=1.5)
        for j, threshold in enumerate(numeric_thresholds):
            plt.scatter(threshold, scores_matrix.loc[model, str(threshold)], color=color_palette_group[i], zorder=3)
        
        # Add to legend
        custom_legend_handles.append(Line2D([0], [0], color=color_palette_group[i], lw=1.5, marker='o', 
                                            markerfacecolor=color_palette_group[i], label=model))

    # Adjust title to reflect lowercase "tp8" or "tp14"
    plt.title(f"Top 10 {experiment_type} Models by Mean {selection_metric} - {score_type}", fontsize=18, pad=20)
    plt.xlabel("Threshold", fontsize=15)
    plt.ylabel(score_type, fontsize=15)
    plt.legend(handles=custom_legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================= Generate Line Plots for Top Models =================

# Generate plots for tp8 based on mean F0.75 and F0.5 selections
for metric, mean_models, scores_matrix in [
    ("F0.75", top_10_models_by_f0_75_mean_tp8, f0_75_scores_matrix_tp8),
    ("Precision", top_10_models_by_f0_75_mean_tp8, precision_scores_matrix),
    ("Accuracy", top_10_models_by_f0_75_mean_tp8, accuracy_scores_matrix),
    ("Recall", top_10_models_by_f0_75_mean_tp8, recall_scores_matrix)
]:
    plot_scores_with_highlighted_group_mean(mean_models, scores_matrix, thresholds, metric, "F0.75", "tp8")

for metric, mean_models, scores_matrix in [
    ("F0.5", top_10_models_by_f0_5_mean_tp8, f0_5_scores_matrix_tp8),
    ("Precision", top_10_models_by_f0_5_mean_tp8, precision_scores_matrix),
    ("Accuracy", top_10_models_by_f0_5_mean_tp8, accuracy_scores_matrix),
    ("Recall", top_10_models_by_f0_5_mean_tp8, recall_scores_matrix)
]:
    plot_scores_with_highlighted_group_mean(mean_models, scores_matrix, thresholds, metric, "F0.5", "tp8")

# Repeat similar process for tp14 models based on mean F0.75 and F0.5
for metric, mean_models, scores_matrix in [
    ("F0.75", top_10_models_by_f0_75_mean_tp14, f0_75_scores_matrix_tp14),
    ("Precision", top_10_models_by_f0_75_mean_tp14, precision_scores_matrix_tp14),
    ("Accuracy", top_10_models_by_f0_75_mean_tp14, accuracy_scores_matrix_tp14),
    ("Recall", top_10_models_by_f0_75_mean_tp14, recall_scores_matrix_tp14)
]:
    plot_scores_with_highlighted_group_mean(mean_models, scores_matrix, thresholds, metric, "F0.75", "tp14")

for metric, mean_models, scores_matrix in [
    ("F0.5", top_10_models_by_f0_5_mean_tp14, f0_5_scores_matrix_tp14),
    ("Precision", top_10_models_by_f0_5_mean_tp14, precision_scores_matrix_tp14),
    ("Accuracy", top_10_models_by_f0_5_mean_tp14, accuracy_scores_matrix_tp14),
    ("Recall", top_10_models_by_f0_5_mean_tp14, recall_scores_matrix_tp14)
]:
    plot_scores_with_highlighted_group_mean(mean_models, scores_matrix, thresholds, metric, "F0.5", "tp14")
    
# ================== PRECISION-RECALL LINE PLOTS FOR MEAN-SELECTED MODELS ======================

def plot_precision_recall_curves(models, precision_matrix, recall_matrix, experiment_type, selection_metric):
    """
    Plots precision-recall lines for the top models across thresholds.

    Parameters:
    - models: DataFrame of selected top models
    - precision_matrix: DataFrame containing precision values for each model across thresholds
    - recall_matrix: DataFrame containing recall values for each model across thresholds
    - experiment_type: "tp8" or "tp14" indicating the experiment type
    - selection_metric: "F0.5" or "F0.75" indicating the selection criterion
    """
    plt.figure(figsize=(9, 7), dpi=600)
    color_palette = sns.color_palette("tab10", len(models))

    # Plot precision-recall curves for each model
    for idx, model in enumerate(models.index):
        recall_values = recall_matrix.loc[model, thresholds]
        precision_values = precision_matrix.loc[model, thresholds]
        plt.plot(recall_values, precision_values, label=model, color=color_palette[idx], marker="o", markersize=5)

    # Add plot details
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"Precision vs Recall for Top 10 {experiment_type.lower()} Models by Mean {selection_metric}", fontsize=16, pad=14)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================== GENERATE PRECISION-RECALL PLOTS ======================

# Plot for tp8 models selected by mean F0.5
plot_precision_recall_curves(top_10_models_by_f0_5_mean_tp8, precision_scores_matrix, recall_scores_matrix, "tp8", "F0.5")

# Plot for tp8 models selected by mean F0.75
plot_precision_recall_curves(top_10_models_by_f0_75_mean_tp8, precision_scores_matrix, recall_scores_matrix, "tp8", "F0.75")

# Plot for tp14 models selected by mean F0.5
plot_precision_recall_curves(top_10_models_by_f0_5_mean_tp14, precision_scores_matrix_tp14, recall_scores_matrix_tp14, "tp14", "F0.5")

# Plot for tp14 models selected by mean F0.75
plot_precision_recall_curves(top_10_models_by_f0_75_mean_tp14, precision_scores_matrix_tp14, recall_scores_matrix_tp14, "tp14", "F0.75") 

# ================ GET EXPERIMENTAL DATA FOR MEAN HEATMAPS ==================
"pln_pd1", "bld_pd1", "panc_pd1", 
# Define experimental data column patterns for each threshold
column_patterns = ["pln_igg", "bld_igg", "panc_igg", "pln_spt", "bld_spt", "panc_spt"]

def enrich_with_experimental_data(top_models, data, threshold_key):
    """
    Enriches top models with experimental data for a specific threshold.

    Parameters:
    - top_models: DataFrame of top models selected by mean score
    - data: Dictionary of DataFrames with experimental data
    - threshold_key: The threshold level to fetch experimental data for

    Returns:
    - DataFrame enriched with experimental data columns, without duplicates.
    """
    # Reset index to make model names a regular column for merging
    top_models = top_models.reset_index().rename(columns={"index": "model name"})

    # Find the prefix dynamically for the threshold (e.g., "t50")
    actual_prefix = [col.split('_')[0] for col in data[threshold_key].columns if col.startswith('t')][0]
    columns_to_merge = [f"{actual_prefix}_{pattern}" for pattern in column_patterns]

    # Select only 'model name' and the experimental columns for the threshold
    threshold_data = data[threshold_key][["model name"] + columns_to_merge]

    # Remove any overlapping experimental data columns from top_models to avoid duplicates
    top_models = top_models.drop(columns=[col for col in top_models.columns if col in columns_to_merge], errors='ignore')

    # Merge the experimental data with the top models DataFrame
    enriched_data = top_models.merge(threshold_data, on="model name", how="left")

    # Ensure columns are unique and restore "model name" as index
    return enriched_data.set_index("model name")

# Enrich data for all top model sets by threshold
enriched_top_10_models_by_f0_75_mean_tp8 = {}
enriched_top_10_models_by_f0_5_mean_tp8 = {}
enriched_top_10_models_by_f0_75_mean_tp14 = {}
enriched_top_10_models_by_f0_5_mean_tp14 = {}

for threshold_key in thresholds:
    enriched_top_10_models_by_f0_75_mean_tp8[threshold_key] = enrich_with_experimental_data(
        top_10_models_by_f0_75_mean_tp8, tp8_data, threshold_key
    )
    enriched_top_10_models_by_f0_5_mean_tp8[threshold_key] = enrich_with_experimental_data(
        top_10_models_by_f0_5_mean_tp8, tp8_data, threshold_key
    )
    enriched_top_10_models_by_f0_75_mean_tp14[threshold_key] = enrich_with_experimental_data(
        top_10_models_by_f0_75_mean_tp14, tp14_data, threshold_key
    )
    enriched_top_10_models_by_f0_5_mean_tp14[threshold_key] = enrich_with_experimental_data(
        top_10_models_by_f0_5_mean_tp14, tp14_data, threshold_key
    )

# Assume columns have been enriched and are consistently named in 'columns_to_plot'
columns_to_plot = [f"t{threshold_key}_{pattern}" for pattern in column_patterns]

# Filter only the existing columns to avoid issues if some columns are missing
columns_to_plot = [col for col in columns_to_plot if col in df.columns]

# ========================= FUNCTION TO PLOT HEATMAPS FOR MEAN-SELECTED MODELS =========================

def plot_experimental_heatmap_for_mean_selected_models(top_models_by_threshold, threshold_key, selection_criterion, experiment_type, row_normalize=False):
    df = top_models_by_threshold[threshold_key]
    
    dynamic_prefix = [col.split('_')[0] for col in df.columns if col.startswith('t')][0]
    columns_to_plot = [f"{dynamic_prefix}_{pattern}" for pattern in column_patterns]
    
    available_columns = [col for col in columns_to_plot if col in df.columns]
    if not available_columns:
        print(f"No experimental data columns found for threshold {threshold_key}. Available columns: {df.columns}")
        return
    
    heatmap_data = df[available_columns].copy()
    heatmap_data.columns = [col.split('_', 1)[1] for col in available_columns]
    
    # Determine global min and max for color scaling
    vmin, vmax = heatmap_data.min().min(), heatmap_data.max().max()

    plt.figure(figsize=(10, 8), dpi=600)
    ax = sns.heatmap(
        heatmap_data, cmap="plasma",
        annot=heatmap_data,  # Show non-normalized values
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Absolute Value'},
        vmin=vmin, vmax=vmax  # Use global min and max for consistent scaling
    )
    
    # Set y-tick labels using index since 'model name' is the index
    ax.set_yticklabels(df.index, rotation=0, ha='right')

    plt.title(f"Top 10 {experiment_type.lower()} Models by {selection_criterion} Score at T{threshold_key}", fontsize=18, pad=20)
    plt.xlabel("Experimental Data Sources", fontsize=15, labelpad=15)
    plt.ylabel("Model Name", fontsize=15, labelpad=15)
    plt.tight_layout()
    plt.show()

# ================== GENERATE HEATMAPS FOR MEAN-SELECTED MODELS ======================

# -------------- ROW NORMALISE ON --------------------

# Debugging helper function to verify data being passed
def debug_plot_data(data, threshold_key, selection_criterion, experiment_type):
    print(f"\nData for {experiment_type}, {selection_criterion}, Threshold: {threshold_key}")
    print(data.get(threshold_key).head() if threshold_key in data else "No data found for this threshold.")

def to_dataframe_if_series(data, threshold_key):
    """Ensure that data is a DataFrame, converting from Series if necessary, with appropriate column naming."""
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame and add threshold-specific prefix to columns
        return data.to_frame(name=f"t{threshold_key}_score")
    return data

for threshold_key in thresholds:
    print(f"Threshold {threshold_key} - F0.75, tp8 data:")
    print(enriched_top_10_models_by_f0_75_mean_tp8[threshold_key].head(), "\n")
    
    print(f"Threshold {threshold_key} - F0.5, tp8 data:")
    print(enriched_top_10_models_by_f0_5_mean_tp8[threshold_key].head(), "\n")
    
    print(f"Threshold {threshold_key} - F0.75, tp14 data:")
    print(enriched_top_10_models_by_f0_75_mean_tp14[threshold_key].head(), "\n")
    
    print(f"Threshold {threshold_key} - F0.5, tp14 data:")
    print(enriched_top_10_models_by_f0_5_mean_tp14[threshold_key].head(), "\n")

for threshold_key in thresholds:
    plot_experimental_heatmap_for_mean_selected_models(
        {threshold_key: enriched_top_10_models_by_f0_75_mean_tp8[threshold_key]}, threshold_key, "F0.75", "tp8", row_normalize=True
    )

for threshold_key in thresholds:
    plot_experimental_heatmap_for_mean_selected_models(
        {threshold_key: enriched_top_10_models_by_f0_5_mean_tp8[threshold_key]}, threshold_key, "F0.5", "tp8", row_normalize=True
    )

for threshold_key in thresholds:
    plot_experimental_heatmap_for_mean_selected_models(
        {threshold_key: enriched_top_10_models_by_f0_75_mean_tp14[threshold_key]}, threshold_key, "F0.75", "tp14", row_normalize=True
    )

for threshold_key in thresholds:
    plot_experimental_heatmap_for_mean_selected_models(
        {threshold_key: enriched_top_10_models_by_f0_5_mean_tp14[threshold_key]}, threshold_key, "F0.5", "tp14", row_normalize=True
    )
    
    
    
    
# Plot for tp8 models selected by threshold-based F0.75 and F0.5 scores
for threshold_key in thresholds:
    plot_experimental_heatmap(
        top_models_by_f0_75_threshold, threshold_key, "F0.75", "tp8", row_normalize=True
    )
    plot_experimental_heatmap(
        top_models_by_f0_5_threshold, threshold_key, "F0.5", "tp8", row_normalize=True
    )

# Plot for tp14 models selected by threshold-based F0.75 and F0.5 scores
for threshold_key in thresholds:
    plot_experimental_heatmap(
        top_models_by_f0_75_threshold_tp14, threshold_key, "F0.75", "tp14", row_normalize=True
    )
    plot_experimental_heatmap(
        top_models_by_f0_5_threshold_tp14, threshold_key, "F0.5", "tp14", row_normalize=True
    )