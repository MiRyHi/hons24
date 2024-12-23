# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 01:56:23 2024

@author: Mitch
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from scipy.stats import wilcoxon, friedmanchisquare
from itertools import combinations
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# ==================== 1. IMPORT DATA FOR TP8 and TP14 ==============================

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

# Example tissue/type columns
tissue_columns = ["panc_spt", "bld_spt", "pln_spt", "panc_igg", "bld_igg", "pln_igg"]

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


# ========================= 2. CREATE SCORES DATAFRAMES =============================

# -------- F SCORES AND F SCORE MEANS --------

# Initialize new DataFrames for tp8 and tp14 to store model scores across thresholds
tp8_scores = pd.DataFrame()
tp14_scores = pd.DataFrame()

# Function to initialize a score DataFrame for a group of models
def initialize_score_dataframe(data, thresholds, group_name):
    # Create a list of all models based on the first threshold's data (assumes model names are consistent across thresholds)
    models = data[thresholds[0]]["model name"].tolist()
    # Initialize DataFrame with model names
    score_df = pd.DataFrame(models, columns=["model name"])
    
    # Add columns for each metric across all thresholds
    for threshold in thresholds:
        score_df[f"F1_{threshold}"] = None
        score_df[f"F0.75_{threshold}"] = None
        score_df[f"F0.5_{threshold}"] = None
    
    return score_df

# Initialize the tp8 and tp14 score DataFrames with the appropriate columns
tp8_scores = initialize_score_dataframe(tp8_data, thresholds, "tp8")
tp14_scores = initialize_score_dataframe(tp14_data, thresholds, "tp14")

# Populate the score DataFrames with F1, F0.75, and F0.5 scores from each threshold-specific DataFrame
def populate_scores(score_df, data, thresholds):
    for threshold in thresholds:
        df = data[threshold]
        # Map F1, F0.75, and F0.5 scores into the respective columns
        score_df[f"F1_{threshold}"] = df["F1 score"].values
        score_df[f"F0.75_{threshold}"] = df["F0.75 score"].values
        score_df[f"F0.5_{threshold}"] = df["F0.5 score"].values
    
# Populate tp8_scores and tp14_scores
populate_scores(tp8_scores, tp8_data, thresholds)
populate_scores(tp14_scores, tp14_data, thresholds)

# Function to add mean columns for F1, F0.75, and F0.5 scores
def add_mean_scores(df):
    # Calculate mean across thresholds for each metric
    df['mean_F1'] = df[[f"F1_{threshold}" for threshold in thresholds]].mean(axis=1)
    df['mean_F0.75'] = df[[f"F0.75_{threshold}" for threshold in thresholds]].mean(axis=1)
    df['mean_F0.5'] = df[[f"F0.5_{threshold}" for threshold in thresholds]].mean(axis=1)

# Apply to both tp8_scores and tp14_scores
add_mean_scores(tp8_scores)
add_mean_scores(tp14_scores)

# --------- ADDITIONAL METRICS: PRECISION, RECALL, ACCURACY ---------

# Function to add precision, recall, and accuracy columns for each threshold
def add_additional_metrics(score_df, data, thresholds):
    for threshold in thresholds:
        df = data[threshold]
        score_df[f"precision_{threshold}"] = df["precision"].values
        score_df[f"recall_{threshold}"] = df["recall"].values
        score_df[f"accuracy_{threshold}"] = df["accuracy"].values
    
    score_df['mean_precision'] = score_df[[f"precision_{threshold}" for threshold in thresholds]].mean(axis=1)
    score_df['mean_recall'] = score_df[[f"recall_{threshold}" for threshold in thresholds]].mean(axis=1)
    score_df['mean_accuracy'] = score_df[[f"accuracy_{threshold}" for threshold in thresholds]].mean(axis=1)

# Apply to both tp8_scores and tp14_scores
add_additional_metrics(tp8_scores, tp8_data, thresholds)
add_additional_metrics(tp14_scores, tp14_data, thresholds)

# --------- CONSISTENCY METRICS (VARIANCE AND STD DEV) ---------

# Function to calculate and add variance and std deviation columns to the score DataFrame
def add_consistency_metrics(df):
    for score_type in ["F1", "F0.75", "F0.5"]:
        # Calculate variance and standard deviation across thresholds
        df[f"{score_type}_variance"] = df[[f"{score_type}_{threshold}" for threshold in thresholds]].var(axis=1)
        df[f"{score_type}_std_dev"] = df[[f"{score_type}_{threshold}" for threshold in thresholds]].std(axis=1)
    
    for metric in ["precision", "recall", "accuracy"]:
        df[f"{metric}_variance"] = df[[f"{metric}_{threshold}" for threshold in thresholds]].var(axis=1)
        df[f"{metric}_std_dev"] = df[[f"{metric}_{threshold}" for threshold in thresholds]].std(axis=1)

# Add consistency metrics to tp8_scores and tp14_scores
add_consistency_metrics(tp8_scores)
add_consistency_metrics(tp14_scores)

# ---------- TISSUE/TYPE DATA -----------

# Function to add tissue/type data to a scores DataFrame
def add_tissue_type_data(scores_df, data, thresholds):
    for threshold in thresholds:
        threshold_df = data[threshold]
        threshold_tissue_columns = [f"t{threshold}_{col}" for col in tissue_columns]
        available_columns = [col for col in threshold_tissue_columns if col in threshold_df.columns]
        
        scores_df = scores_df.merge(
            threshold_df[["model name"] + available_columns],
            on="model name",
            how="left"
        )
    return scores_df

# Apply tissue/type data function to tp8_scores and tp14_scores
tp8_scores = add_tissue_type_data(tp8_scores, tp8_data, thresholds)
tp14_scores = add_tissue_type_data(tp14_scores, tp14_data, thresholds)

# Function to add tissue/type total nseq to a scores DataFrame
def add_tissue_type_data_nseq(scores_df, data, thresholds):
    first_threshold = thresholds[0]
    threshold_df_nseq = data[first_threshold]
    threshold_tissue_columns = [f"nseq_{col}" for col in tissue_columns]
    available_columns = [col for col in threshold_tissue_columns if col in threshold_df_nseq.columns]
    
    scores_df = scores_df.merge(
        threshold_df_nseq[["model name"] + available_columns],
        on="model name",
        how="left"
    )
    return scores_df

# Applying the modified function
tp8_scores = add_tissue_type_data_nseq(tp8_scores, tp8_data, thresholds)
tp14_scores = add_tissue_type_data_nseq(tp14_scores, tp14_data, thresholds)

def calculate_counts(scores_df, tissue_columns, thresholds):
    for threshold in thresholds:
        for col in tissue_columns:
            fraction_col = f"t{threshold}_{col}"
            nseq_col = f"nseq_{col}"
            new_col = f"{threshold}_ct_{col}"
            
            if fraction_col in scores_df.columns and nseq_col in scores_df.columns:
                scores_df[new_col] = scores_df[fraction_col] * scores_df[nseq_col]
    
    return scores_df

# Apply the function to tp8_scores and tp14_scores
tp8_scores = calculate_counts(tp8_scores, tissue_columns, thresholds)
tp14_scores = calculate_counts(tp14_scores, tissue_columns, thresholds)

# ----------- SELECTION FUNCTION -----------

def select_top_models(tp_scores_df, score_type="F0.5", num_models=10, mean_performance_threshold=0.5, std_dev_threshold=0.1):
    """
    Select top models based on specified mean F score type, filtered by mean performance and standard deviation.

    Parameters:
    - tp_scores_df: DataFrame with performance metrics for each model.
    - score_type: F score type to prioritize (e.g., "F1", "F0.75", "F0.5").
    - num_models: Number of top models to select.
    - mean_performance_threshold: Minimum mean F score required for a model.
    - std_dev_threshold: Maximum allowable standard deviation for a model's performance.

    Returns:
    - DataFrame containing the final filtered list of top models.
    """
    # Define column names based on score type
    mean_column = f"mean_{score_type}"
    std_dev_column = f"{score_type}_std_dev"

    # Apply mean performance and std dev filtering
    filtered_df = tp_scores_df[
        (tp_scores_df[mean_column] >= mean_performance_threshold) &
        (tp_scores_df[std_dev_column] <= std_dev_threshold)
    ]

    # Sort by the mean F score type in descending order
    sorted_df = filtered_df.sort_values(by=mean_column, ascending=False)

    # Select the top `num_models` models
    top_models_df = sorted_df.head(num_models)

    return top_models_df

# For tp14 models
selected_tp14_models_F1 = select_top_models(
    tp14_scores, 
    score_type="F1", 
    num_models=24,  # or any desired number
    mean_performance_threshold=0.685,
)

# For tp8 models
selected_tp8_models_F1 = select_top_models(
    tp8_scores, 
    score_type="F1", 
    num_models=24,  # or any desired number
    mean_performance_threshold=0.685,
)

# Adding specificity values from tp14_data to tp14_scores
for threshold in ["50", "60", "70", "80", "90"]:
    # Extract specificity values for each threshold and rename the column
    tp14_specificity = tp14_data[threshold][['model name', 'specificity']].copy()
    tp14_specificity.rename(columns={'specificity': f"specificity_{threshold}"}, inplace=True)
    
    # Merge specificity values into tp14_scores on "model name"
    tp14_scores = tp14_scores.merge(tp14_specificity, on="model name", how="left")

# Adding specificity values from tp8_data to tp8_scores
for threshold in ["50", "60", "70", "80", "90"]:
    # Extract specificity values for each threshold and rename the column
    tp8_specificity = tp8_data[threshold][['model name', 'specificity']].copy()
    tp8_specificity.rename(columns={'specificity': f"specificity_{threshold}"}, inplace=True)
    
    # Merge specificity values into tp8_scores on "model name"
    tp8_scores = tp8_scores.merge(tp8_specificity, on="model name", how="left")

# Adding NPV values from tp14_data to tp14_scores
for threshold in ["50", "60", "70", "80", "90"]:
    # Extract specificity values for each threshold and rename the column
    tp14_npv = tp14_data[threshold][['model name', 'NPV']].copy()
    tp14_npv.rename(columns={'NPV': f"NPV_{threshold}"}, inplace=True)
    
    # Merge specificity values into tp14_scores on "model name"
    tp14_scores = tp14_scores.merge(tp14_npv, on="model name", how="left")

# Adding NPV values from tp8_data to tp8_scores
for threshold in ["50", "60", "70", "80", "90"]:
    # Extract specificity values for each threshold and rename the column
    tp8_npv = tp8_data[threshold][['model name', 'NPV']].copy()
    tp8_npv.rename(columns={'NPV': f"NPV_{threshold}"}, inplace=True)
    
    # Merge specificity values into tp8_scores on "model name"
    tp8_scores = tp8_scores.merge(tp8_npv, on="model name", how="left")
    
# =================== PLOT SELECTIONS BY F SCORE, PRECISION, RECALL ==========================
# Define a consistent color palette for both the main plot and the legend
combined_color_palette = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + plt.cm.tab20b(np.linspace(0, 1, 4)).tolist()

# Create a separate legend plot
def plot_legend_only(color_palette, model_names):
    """
    Creates a separate legend plot with both lines and dots for each model.
    
    Parameters:
    - color_palette: List of colors to use for each model.
    - model_names: List of model names to include in the legend.
    """
    plt.figure(figsize=(2, 6), dpi=100)
    for idx, model in enumerate(model_names):
        plt.plot([], [], color=color_palette[idx], marker='o', markersize=8, linestyle='-', linewidth=1.5, label=model)  # Lines and dots for legend

    plt.legend(title="Models", loc="center", prop={'size': 8}, title_fontsize='9')
    plt.axis('off')  # Hide axes
    plt.show()

def plot_scores_across_thresholds(selected_models, all_models, thresholds, select_score, plot_metric, data_label="tp8", color_map=None, x_min=60, x_max=90, y_min=0.7, y_max=0.85):
    """
    Plots the specified metric (e.g., 'F0.5') across thresholds for selected models in color
    and other models in transparent gray.

    Parameters:
    - selected_models: DataFrame of the selected models with the metric across thresholds.
    - all_models: DataFrame containing all models, including selected models.
    - thresholds: List of threshold values (e.g., [50, 60, 70, 80, 90]).
    - select_score: String indicating the criterion used for selection.
    - plot_metric: String of the metric column prefix (e.g., 'F0.5') to plot.
    - data_label: Label for distinguishing tp8 or tp14 in the plot title.
    - color_map: Custom color list to assign unique colors to each model.
    - x_min, x_max, y_min, y_max: Axis limits for X and Y axes.
    """
    # Generate a color palette if none is provided
    if color_map is None:
        # Combine two color maps to reach 24 distinct colors
        color_map = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + plt.cm.tab20b(np.linspace(0, 1, 4)).tolist()

    plt.figure(figsize=(8, 6), dpi=400)

    # Plot gray lines for non-selected models
    for model in all_models['model name']:
        if model not in selected_models['model name'].values:
            scores = [all_models.loc[all_models['model name'] == model, f"{plot_metric}_{threshold}"].values[0] for threshold in thresholds]
            plt.plot(thresholds, scores, color='#E8E8E8', alpha=1, linewidth=1.0, zorder=1)

    # Plot colored lines with dots for selected models
    for idx, model in enumerate(selected_models['model name']):
        scores = [selected_models.loc[selected_models['model name'] == model, f"{plot_metric}_{threshold}"].values[0] for threshold in thresholds]
        color = color_map[idx]  # Assign a unique color from the expanded color palette
        plt.plot(thresholds, scores, color=color, linewidth=1.2, label=model, zorder=2)
        plt.scatter(thresholds, scores, color=color, s=20, zorder=3)  # Add dots on top

    # Custom X and Y limits and ticks
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min, y_max)
    plt.xticks(thresholds)  # Ensure exact threshold labels on x-axis
    plt.yticks(np.arange(y_min, y_max + 0.05, 0.05))  # Y-axis tick marks, adjusting as needed

    # Labels and title
    if plot_metric == "NPV":
        title_metric = "NPV Scores"
    elif plot_metric == "F1":
        title_metric = "F1 Scores"
    elif plot_metric == "precision":
        title_metric = "Precision (PPV)"
    elif plot_metric == "specificity":
        title_metric = "Specificity (TNR)"
    elif plot_metric == "recall":
        title_metric = "Recall/Sensitivity (TPR)"
    elif plot_metric == "accuracy":
        title_metric = "Accuracy"
    else:
        title_metric = f"{plot_metric.capitalize()} Scores"
    
    plt.xlabel("Threshold (T)", fontsize=13, labelpad=10)
    plt.ylabel(f"{title_metric}", fontsize=13, labelpad=10)
    plt.title(f"Top 24 Models ({data_label.capitalize()}) by {select_score} T(mean): {title_metric}", fontsize=16, pad=15)

    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Legend and display
    plt.tight_layout()
    plt.show()

# Example usage
thresholds = [50, 60, 70, 80, 90]

# Example usage to create separate legends
plot_legend_only(combined_color_palette, selected_tp14_models_F1['model name'].tolist())
plot_legend_only(combined_color_palette, selected_tp8_models_F1['model name'].tolist())

plot_scores_across_thresholds(selected_tp14_models_F1, tp14_scores, thresholds, select_score="F1", plot_metric="F1", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.45, y_max=0.85)
plot_scores_across_thresholds(selected_tp14_models_F1, tp14_scores, thresholds, select_score="F1", plot_metric="precision", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.7, y_max=0.95)
plot_scores_across_thresholds(selected_tp14_models_F1, tp14_scores, thresholds, select_score="F1", plot_metric="recall", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.35, y_max=0.8)
plot_scores_across_thresholds(selected_tp14_models_F1, tp14_scores, thresholds, select_score="F1", plot_metric="NPV", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.50, y_max=0.75)
plot_scores_across_thresholds(selected_tp14_models_F1, tp14_scores, thresholds, select_score="F1", plot_metric="specificity", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.70, y_max=0.95)
plot_scores_across_thresholds(selected_tp14_models_F1, tp14_scores, thresholds, select_score="F1", plot_metric="accuracy", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.55, y_max=0.85)
plot_scores_across_thresholds(selected_tp8_models_F1, tp8_scores, thresholds, select_score="F1", plot_metric="F1", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.45, y_max=0.85)
plot_scores_across_thresholds(selected_tp8_models_F1, tp8_scores, thresholds, select_score="F1", plot_metric="precision", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.7, y_max=0.95)
plot_scores_across_thresholds(selected_tp8_models_F1, tp8_scores, thresholds, select_score="F1", plot_metric="recall", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.35, y_max=0.8)
plot_scores_across_thresholds(selected_tp8_models_F1, tp8_scores, thresholds, select_score="F1", plot_metric="NPV", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.50, y_max=0.75)
plot_scores_across_thresholds(selected_tp8_models_F1, tp8_scores, thresholds, select_score="F1", plot_metric="specificity", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.70, y_max=0.95)
plot_scores_across_thresholds(selected_tp8_models_F1, tp8_scores, thresholds, select_score="F1", plot_metric="accuracy", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.55, y_max=0.85)

def add_combined_metric_rankings(scores_df, metrics):
    """
    Adds individual metric rankings and a combined rank for selecting top models.
    
    Parameters:
    - scores_df: DataFrame containing model performance metrics.
    - metrics: List of metric column names to rank.

    Returns:
    - Updated DataFrame with ranking columns and combined rank column added.
    """
    # Calculate the rank for each metric, with higher values ranked better (lower rank number is better)
    for metric in metrics:
        scores_df[f"{metric}_rank"] = scores_df[metric].rank(ascending=False)  # Ascending=False for highest rank first

    # Calculate combined rank based on the mean of the individual metric ranks
    scores_df['combined_rank'] = scores_df[[f"{metric}_rank" for metric in metrics]].mean(axis=1)

    return scores_df

# List of metrics to use for ranking
metrics = ['mean_precision', 'mean_recall', 'mean_specificity']

# Apply ranking to tp8_scores and tp14_scores
tp8_scores = add_combined_metric_rankings(tp8_scores, metrics)
tp14_scores = add_combined_metric_rankings(tp14_scores, metrics)

# Now select top models by the lowest combined rank (higher metrics values ranked better)
top_24_tp8 = tp8_scores.nsmallest(24, 'combined_rank')
top_24_tp14 = tp14_scores.nsmallest(24, 'combined_rank')

# For selecting a smaller subset, like top 12 models
top_12_tp8 = top_24_tp8.nsmallest(12, 'combined_rank')
top_12_tp14 = top_24_tp14.nsmallest(12, 'combined_rank')



# =================== PLOT SELECTIONS BY F SCORE, PRECISION, RECALL ==========================
def plot_scores_across_thresholds(all_models, thresholds, plot_metric, data_label="tp8", color_map=plt.cm.Set1, x_min=50, x_max=90, y_min=0.5, y_max=0.9, legend_pos="upper left"):
    """
    Plots the specified metric (e.g., 'F0.5') across thresholds for all models, with specific highlights
    for the model with the highest, lowest, and median average scores across thresholds.

    Parameters:
    - all_models: DataFrame containing all models, including selected models.
    - thresholds: List of threshold values (e.g., [50, 60, 70, 80, 90]).
    - plot_metric: String of the metric column prefix (e.g., 'F0.5') to plot.
    - data_label: Label for distinguishing tp8 or tp14 in the plot title.
    - color_map: Color map for other models' lines.
    - x_min, x_max, y_min, y_max: Axis limits for X and Y axes.
    """
    plt.figure(figsize=(10, 6), dpi=400)

    # Calculate the average metric across thresholds for each model
    all_models['avg_metric'] = all_models[[f"{plot_metric}_{threshold}" for threshold in thresholds]].mean(axis=1)
    
    # Identify the models with the highest, lowest, and median average metric
    avg_metrics = all_models[[f"{plot_metric}_{threshold}" for threshold in thresholds]].mean(axis=1)
    max_model = all_models.loc[avg_metrics.idxmax()]
    min_model = all_models.loc[avg_metrics.idxmin()]
    median_model = all_models.loc[avg_metrics.sort_values().index[len(all_models) // 2]]
    
    # Plot non-highlighted models without adding them to the legend
    for model in all_models['model name']:
        if model not in [max_model['model name'], min_model['model name'], median_model['model name']]:
            scores = [all_models.loc[all_models['model name'] == model, f"{plot_metric}_{threshold}"].values[0] for threshold in thresholds]
            plt.plot(thresholds, scores, color='#D3D3D3', alpha=0.9, linewidth=0.7, zorder=1, label='_nolegend_')  # `_nolegend_` prevents them from appearing in the legend
    
    # Plot highlighted models with specific colors and labels
    highlighted_handles = []  # Store handles for custom legend order
    for model, color in zip([max_model, median_model, min_model], ['green', 'orange', 'red']):
        scores = [all_models.loc[all_models['model name'] == model['model name'], f"{plot_metric}_{threshold}"].values[0] for threshold in thresholds]
        line, = plt.plot(thresholds, scores, color=color, linewidth=1.5, label=f"{model['model name']} ({'High' if color == 'green' else 'Median' if color == 'orange' else 'Low'})", zorder=2)
        plt.scatter(thresholds, scores, color=color, s=25, zorder=3)
        highlighted_handles.append(line)  # Append handle to control legend order
    
    # Custom X and Y limits and ticks
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min, y_max)
    plt.xticks(thresholds)  # Exact threshold labels on x-axis
    plt.yticks(np.arange(y_min, y_max + 0.05, 0.05))  # Y-axis tick marks

    # Labels and title
    plt.xlabel("Decision Threshold (T)", fontsize=13, labelpad=10)
    plt.ylabel(f"{plot_metric.capitalize()}", fontsize=13, labelpad=10)  # Capitalize the metric name
    plt.title(f"{data_label.capitalize()} Models: {plot_metric.capitalize()} Scores Across Thresholds", fontsize=16, pad=15)

    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Legend and display
    plt.legend(handles=highlighted_handles, title="Mean Across Thresholds", loc=legend_pos, fontsize=10)
    plt.tight_layout()
    plt.show()

# Example usage
thresholds = [50, 60, 70, 80, 90]



plot_scores_across_thresholds(tp14_scores, thresholds, plot_metric="precision", data_label="tp14", x_min=50, x_max=90, y_min=0.7, y_max=0.95)
plot_scores_across_thresholds(tp8_scores, thresholds, plot_metric="precision", data_label="tp8", x_min=50, x_max=90, y_min=0.7, y_max=0.95)
plot_scores_across_thresholds(tp14_scores, thresholds, plot_metric="specificity", data_label="tp14", x_min=50, x_max=90, y_min=0.7, y_max=0.95, legend_pos="lower right")
plot_scores_across_thresholds(tp8_scores, thresholds, plot_metric="specificity", data_label="tp8", x_min=50, x_max=90, y_min=0.7, y_max=0.95, legend_pos="lower right")

plot_scores_across_thresholds(top_12_tp14, thresholds, plot_metric="precision", data_label="Top 12 tp14", x_min=50, x_max=90, y_min=0.80, y_max=1.0)
plot_scores_across_thresholds(top_12_tp8, thresholds, plot_metric="precision", data_label="Top 12 tp8", x_min=50, x_max=90, y_min=0.80, y_max=1.0)
plot_scores_across_thresholds(top_24_tp14, thresholds, plot_metric="precision", data_label="Top 24 tp14", x_min=50, x_max=90, y_min=0.80, y_max=1.0)
plot_scores_across_thresholds(top_24_tp8, thresholds, plot_metric="precision", data_label="Top 24 tp8", x_min=50, x_max=90, y_min=0.80, y_max=1.0)
plot_scores_across_thresholds(top_12_tp14, thresholds, plot_metric="recall", data_label="Top 12 tp14", x_min=50, x_max=90, y_min=0.0, y_max=0.8, legend_pos="lower left")
plot_scores_across_thresholds(top_12_tp8, thresholds, plot_metric="recall", data_label="Top 12 tp8", x_min=50, x_max=90, y_min=0.0, y_max=0.8, legend_pos="lower left")
plot_scores_across_thresholds(top_24_tp14, thresholds, plot_metric="recall", data_label="Top 24 tp14", x_min=50, x_max=90, y_min=0.0, y_max=0.8, legend_pos="lower left")
plot_scores_across_thresholds(top_24_tp8, thresholds, plot_metric="recall", data_label="Top 24 tp8", x_min=50, x_max=90, y_min=0.0, y_max=0.8, legend_pos="lower left")
plot_scores_across_thresholds(top_12_tp14, thresholds, plot_metric="specificity", data_label="Top 12 tp14", x_min=50, x_max=90, y_min=0.8, y_max=1.0, legend_pos="lower right")
plot_scores_across_thresholds(top_12_tp8, thresholds, plot_metric="specificity", data_label="Top 12 tp8", x_min=50, x_max=90, y_min=0.8, y_max=1.0, legend_pos="lower right")
plot_scores_across_thresholds(top_24_tp14, thresholds, plot_metric="specificity", data_label="Top 24 tp14", x_min=50, x_max=90, y_min=0.8, y_max=1.0, legend_pos="lower right")
plot_scores_across_thresholds(top_24_tp8, thresholds, plot_metric="specificity", data_label="Top 24 tp8", x_min=50, x_max=90, y_min=0.8, y_max=1.0, legend_pos="lower right")

thresholds = [50, 60, 70, 80, 90]


def calculate_mean_specificity(scores_df):
    specificity_columns = [f"specificity_{threshold}" for threshold in [50, 60, 70, 80, 90]]
    scores_df['mean_specificity'] = scores_df[specificity_columns].mean(axis=1)
    return scores_df

# Apply to both tp8_scores and tp14_scores
tp8_scores = calculate_mean_specificity(tp8_scores)
tp14_scores = calculate_mean_specificity(tp14_scores)

def get_tick_interval(data_range, plot_type):
    """
    Calculate an appropriate tick interval based on the data range.
    """
    if plot_type == "fraction":
        return 0.1  # for fraction data, tick intervals of 0.1
    else:
        if data_range <= 100:
            return 20
        elif data_range <= 300:
            return 50
        elif data_range <= 500:
            return 100
        else:
            return 200

def plot_raw_scatter_consistent_scale(selected_models, thresholds, data_label, tissue_pairs, f_score_type="F0.5", plot_type="count"):
    """
    Generate scatter plots for each tissue pair across all thresholds
    with consistent scaling, padding, and axis limits, while ensuring
    dots near the axes are not cut off.
    
    Parameters:
    - selected_models: DataFrame containing the selected models.
    - thresholds: List of thresholds to iterate over.
    - data_label: Label for the data source (e.g., "tp8" or "tp14").
    - tissue_pairs: List of tuples specifying tissue pairs for comparison.
    - f_score_type: String indicating the F-score type used for model selection (e.g., "F0.5", "F0.75").
    - plot_type: String indicating whether to plot "count" or "fraction" data.
    """
    # Define the mapping for visible names
    tissue_name_map = {
        "bld_spt": "sptT1D (bld)",
        "bld_igg": "preT1D (bld)",
        "panc_spt": "sptT1D (pan)",
        "panc_igg": "preT1D (pan)",
        "pln_spt": "sptT1D (pln)",
        "pln_igg": "preT1D (pln)"
    }
    
    for x_col, y_col in tissue_pairs:
        # Determine column prefix for fraction or count data
        if plot_type == "fraction":
            x_vals_all = pd.concat([selected_models[f"t{threshold}_{x_col}"] for threshold in thresholds if f"t{threshold}_{x_col}" in selected_models.columns])
            y_vals_all = pd.concat([selected_models[f"t{threshold}_{y_col}"] for threshold in thresholds if f"t{threshold}_{y_col}" in selected_models.columns])
            common_min, common_max = 0, 0.5  # for fractions with max of 0.5
        else:
            x_vals_all = pd.concat([selected_models[f"{threshold}_ct_{x_col}"] for threshold in thresholds if f"{threshold}_ct_{x_col}" in selected_models.columns])
            y_vals_all = pd.concat([selected_models[f"{threshold}_ct_{y_col}"] for threshold in thresholds if f"{threshold}_ct_{y_col}" in selected_models.columns])
            # Set minimum at 0 and maximum rounded up to the nearest hundred for counts
            common_min = 0
            overall_max = max(x_vals_all.max(), y_vals_all.max())
            common_max = (int(overall_max / 100) + 1) * 100  # round up to the next hundred

        tick_interval = get_tick_interval(common_max - common_min, plot_type)

        # Get formatted names for X and Y axes
        x_label = tissue_name_map.get(x_col, x_col)
        y_label = tissue_name_map.get(y_col, y_col)
        axis_label_suffix = f"TETpos {'count' if plot_type == 'count' else 'fraction'} top 24 {data_label} models"

        for threshold in thresholds:
            # Column names for X and Y values for the current threshold
            if plot_type == "fraction":
                x_val_col = f"t{threshold}_{x_col}"
                y_val_col = f"t{threshold}_{y_col}"
            else:
                x_val_col = f"{threshold}_ct_{x_col}"
                y_val_col = f"{threshold}_ct_{y_col}"
            
            if x_val_col in selected_models.columns and y_val_col in selected_models.columns:
                x_vals = selected_models[x_val_col]
                y_vals = selected_models[y_val_col]
                
                plt.figure(figsize=(8, 8), dpi=300)
                
                # Plot scatter points
                plt.scatter(x_vals, y_vals, marker='o', s=25, alpha=0.6, color='blue')
                
                # Plot 45-degree line based on plot type limits
                plt.plot([common_min, common_max], [common_min, common_max], color='red', linestyle='--', alpha=0.5)
                
                # Set plot limits
                plt.xlim(common_min, common_max)
                plt.ylim(common_min, common_max)
                
                # Set tick marks with dynamic intervals
                plt.xticks(np.arange(common_min, common_max + tick_interval, tick_interval), fontsize=10)
                plt.yticks(np.arange(common_min, common_max + tick_interval, tick_interval), fontsize=10)
                
                # Set title and labels
                plt.title(f"{data_label} TETpos Predicted {'Counts' if plot_type == 'count' else 'Fractions'} at T{threshold}: {x_label} vs. {y_label}", fontsize=14, pad=20)
                plt.xlabel(f"{x_label} {axis_label_suffix} by {f_score_type} T(mean, s.d.)", fontsize=12, labelpad=15)
                plt.ylabel(f"{y_label} {axis_label_suffix} by {f_score_type} T(mean, s.d.)", fontsize=12, labelpad=15)
                
                # Add grid for readability
                plt.grid(True, linestyle='--', color='gray', alpha=0.3)
                
                plt.show()
            else:
                print(f"Columns {x_val_col} or {y_val_col} not found in DataFrame for threshold {threshold}")

# Define tissue pairs and thresholds for testing
tissue_pairs = [("bld_spt", "bld_igg"), ("panc_spt", "panc_igg"), ("pln_spt", "pln_igg")]
thresholds = ["50", "60", "70", "80", "90"]



# Calls for tp8 F score-selected models
plot_raw_scatter_consistent_scale(selected_tp8_models_F1, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, f_score_type="F1", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp8_models_F0_75, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, f_score_type="F0.75", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp8_models_F0_5, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, f_score_type="F0.5", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp8_models_F1, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, f_score_type="F1", plot_type="count")
plot_raw_scatter_consistent_scale(selected_tp8_models_F0_75, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, f_score_type="F0.75", plot_type="count")
plot_raw_scatter_consistent_scale(selected_tp8_models_F0_5, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, f_score_type="F0.5", plot_type="count")

# Calls for tp14 F score-selected models
plot_raw_scatter_consistent_scale(selected_tp14_models_F1, thresholds, data_label="tp14", tissue_pairs=tissue_pairs, f_score_type="F1", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp14_models_F0_75, thresholds, data_label="tp14", tissue_pairs=tissue_pairs, f_score_type="F0.75", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp14_models_F0_5, thresholds, data_label="tp14", tissue_pairs=tissue_pairs, f_score_type="F0.5", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp14_models_F1, thresholds, data_label="tp14", tissue_pairs=tissue_pairs, f_score_type="F1", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp14_models_F0_75, thresholds, data_label="tp14", tissue_pairs=tissue_pairs, f_score_type="F0.75", plot_type="fraction")
plot_raw_scatter_consistent_scale(selected_tp14_models_F0_5, thresholds, data_label="tp14", tissue_pairs=tissue_pairs, f_score_type="F0.5", plot_type="fraction")


# Define tissue pairs and thresholds for testing
tissue_pairs = [("bld_spt", "bld_igg"), ("panc_spt", "panc_igg"), ("pln_spt", "pln_igg")]
thresholds = ["50", "60", "70", "80", "90"]


def plot_combined_thresholds_scatter(selected_models, thresholds, data_label, tissue_pairs, plot_type="fraction"):
    """
    Generate a single scatter plot for each tissue pair across all thresholds.
    Each threshold will have a unique color.
    """
    # Define color mapping for each threshold with specific labels
    threshold_colors = {
        "50": "mediumorchid",  # medium purple
        "60": "dodgerblue",  # medium blue
        "70": "forestgreen",  # medium green
        "80": "orange",  # orange
        "90": "red",  # bright red
    }

    # Define the mapping for visible names
    tissue_name_map = {
        "bld_spt": "Diabetic blood",
        "bld_igg": "Pre-diabetic blood",
        "panc_spt": "Diabetic pancreas",
        "panc_igg": "Pre-diabetic pancreas",
        "pln_spt": "Diabetic pLN",
        "pln_igg": "Pre-diabetic pLN"
    }

    for x_col, y_col in tissue_pairs:
        plt.figure(figsize=(8, 8), dpi=300)

        # Determine appropriate axis limits
        if plot_type == "fraction":
            common_max = 0.5 if 'panc' in x_col or 'panc' in y_col else 0.35
            buffer = 0.01  # Add a small buffer for fractions
        else:  # Counts
            common_max = max(
                selected_models[
                    [f"{threshold}_ct_{x_col}" for threshold in thresholds if f"{threshold}_ct_{x_col}" in selected_models.columns] + 
                    [f"{threshold}_ct_{y_col}" for threshold in thresholds if f"{threshold}_ct_{y_col}" in selected_models.columns]
                ].max()
            )
            common_max = round(common_max + 50, -1)  # Round up for padding
            buffer = 0.05 * common_max  # 5% buffer for counts

        for threshold in thresholds:
            x_val_col = f"t{threshold}_{x_col}" if plot_type == "fraction" else f"{threshold}_ct_{x_col}"
            y_val_col = f"t{threshold}_{y_col}" if plot_type == "fraction" else f"{threshold}_ct_{y_col}"

            if x_val_col in selected_models.columns and y_val_col in selected_models.columns:
                x_vals = selected_models[x_val_col]
                y_vals = selected_models[y_val_col]

                # Plot each threshold with its unique color and add label for legend
                plt.scatter(x_vals, y_vals, color=threshold_colors[threshold], s=10, alpha=0.8, linewidths=0.75, label=f"T{threshold}")

        # Apply mapping for axis labels
        x_label = tissue_name_map.get(x_col, x_col)
        y_label = tissue_name_map.get(y_col, y_col)

        # Plot the 45-degree line, extending to the plot limits
        plt.plot([0, common_max], [0, common_max], color='black', linestyle=':', linewidth=1)

        # Configure plot limits with a small buffer below zero
        plt.xlim(-buffer, common_max)
        plt.ylim(-buffer, common_max)
        tick_interval = 0.05 if plot_type == "fraction" else 50
        plt.xticks(np.arange(0, common_max + tick_interval, tick_interval), fontsize=8)
        plt.yticks(np.arange(0, common_max + tick_interval, tick_interval), fontsize=8)

        # Title and axis labels
        plt.xlabel(f"{x_label}", fontsize=11, labelpad=15)
        plt.ylabel(f"{y_label}", fontsize=11, labelpad=15)

        # Add grid for readability
        plt.grid(True, linestyle='--', color='gray', alpha=0.2)

        # Add legend within the plot area
        plt.legend(title="Threshold", loc="upper left", fontsize=10, title_fontsize='11', frameon=True)

        plt.show()

# Call example for fractions
plot_combined_thresholds_scatter(selected_tp8_models_F1, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, plot_type="fraction")
plot_combined_thresholds_scatter(selected_tp14_models_F1, thresholds, data_label="tp14", tissue_pairs=tissue_pairs, plot_type="fraction")

plot_combined_thresholds_scatter(tp8_scores, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, plot_type="fraction")
plot_combined_thresholds_scatter(tp14_scores, thresholds, data_label="tp8", tissue_pairs=tissue_pairs, plot_type="fraction")

plot_combined_thresholds_scatter(top_12_tp14, thresholds, data_label="Top 12 tp14", tissue_pairs=tissue_pairs, plot_type="fraction")
plot_combined_thresholds_scatter(top_24_tp14, thresholds, data_label="Top 24 tp14", tissue_pairs=tissue_pairs, plot_type="fraction")



# ================= SCATTER PLOTS FOR TP8 V TP14 ONE-TISSUE COMPARISON =============

def plot_tp8_vs_tp14_counts(tp8_scores, tp14_scores, thresholds, tissues):
    """
    Generate scatter plots comparing tp8 and tp14 counts for each specified tissue type and threshold.

    Parameters:
    - tp8_scores: DataFrame containing tp8 count data.
    - tp14_scores: DataFrame containing tp14 count data.
    - thresholds: List of thresholds to compare (e.g., ["50", "60", "70", "80", "90"]).
    - tissues: List of tissue types to compare (e.g., ["panc_spt", "bld_spt", "pln_spt"]).
    """
    for threshold in thresholds:
        for tissue in tissues:
            # Column names for tp8 and tp14 data for the current tissue and threshold
            tp8_col = f"{threshold}_ct_{tissue}"
            tp14_col = f"{threshold}_ct_{tissue}"
            
            if tp8_col in tp8_scores.columns and tp14_col in tp14_scores.columns:
                # Get tp8 and tp14 data for the current tissue and threshold
                tp8_data = tp8_scores[tp8_col]
                tp14_data = tp14_scores[tp14_col]
                
                # Scatter plot
                plt.figure(figsize=(8, 8), dpi=300)
                plt.scatter(tp8_data, tp14_data, alpha=1.0, color="blue", s=25)
                
                # Add 45-degree reference line
                min_val = min(tp8_data.min(), tp14_data.min())
                max_val = max(tp8_data.max(), tp14_data.max())
                plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", alpha=0.5)
                
                # Set plot limits and labels
                plt.xlim(min_val, max_val)
                plt.ylim(min_val, max_val)
                plt.xlabel(f"{tissue} Counts in tp8 at T{threshold}", fontsize=12, labelpad=15)
                plt.ylabel(f"{tissue} Counts in tp14 at T{threshold}", fontsize=12, labelpad=15)
                
                # Title and grid
                plt.title(f"{tissue} Count Comparison at T{threshold}: tp8 vs. tp14", fontsize=15, pad=20)
                plt.grid(True, linestyle="--", color="gray", alpha=0.3)
                
                plt.show()
            else:
                print(f"Columns {tp8_col} or {tp14_col} not found in the provided DataFrames.")

# Example usage
thresholds = ["50", "60", "70", "80", "90"]
tissues = ["panc_spt", "bld_spt", "pln_spt", "panc_igg", "bld_igg", "pln_igg"]

plot_tp8_vs_tp14_counts(tp8_scores, tp14_scores, thresholds, tissues)


statistics_df = pd.DataFrame(columns=["data_label", "threshold", "sample_name", "median", "IQR", "top_greater_than_bottom_count"])

def plot_heatmaps_for_models(selected_models, data_label, f_score_type, data_type="count"):
    global statistics_df  # Use the shared DataFrame

    # Heatmap settings
    heatmap_cmap = "magma"
    heatmap_xaxis_title_fontsize = 16
    heatmap_yaxis_title_fontsize = 16
    heatmap_xaxis_tick_label_fontsize = 7
    heatmap_yaxis_tick_label_fontsize = 12
    heatmap_value_fontsize = 7.5

    thresholds = ["50", "60", "70", "80", "90"]
    tissue_columns_template = ["pln_spt","pln_igg"]
    formatted_tissue_labels = ["Diabetic pLN", "Pre-diabetic pLN"]

    # Sort by standard deviation of the F score in ascending order
    std_dev_column = "mean_F1"
    selected_models = selected_models.sort_values(by=std_dev_column, ascending=True)

    for threshold_key in thresholds:
        if data_type == "count":
            tissue_columns = [f"{threshold_key}_ct_{tissue}" for tissue in tissue_columns_template]
            fmt = ".0f"
        elif data_type == "fraction":
            tissue_columns = [f"t{threshold_key}_{tissue}" for tissue in tissue_columns_template]
            fmt = ".2f"

        heatmap_data = selected_models[tissue_columns]
        heatmap_data.columns, heatmap_data = tissue_columns_template, heatmap_data.T
        heatmap_data.columns = selected_models["model name"]

        # Calculate median, IQR, and comparison count for each row
        row_medians = heatmap_data.median(axis=1)
        row_q1 = heatmap_data.quantile(0.25, axis=1)
        row_q3 = heatmap_data.quantile(0.75, axis=1)
        row_iqr = row_q3 - row_q1
        top_greater_than_bottom_count = (heatmap_data.iloc[0] > heatmap_data.iloc[-1]).sum()

        # Append statistics to the shared DataFrame
        for row_name, median, iqr in zip(heatmap_data.index, row_medians, row_iqr):
            statistics_df = statistics_df.append({
                "data_label": data_label,
                "threshold": threshold_key,
                "sample_name": row_name,
                "median": median,
                "IQR": iqr,
                "top_greater_than_bottom_count": top_greater_than_bottom_count
            }, ignore_index=True)
    
        # Display median and IQR results
        print("Row Medians:\n", row_medians)
        print("Row IQRs:\n", row_iqr)
    
        # Check how many values in the top row are greater than corresponding values in the bottom row
        top_row = heatmap_data.iloc[0]
        bottom_row = heatmap_data.iloc[-1]
        top_greater_than_bottom_count = (top_row > bottom_row).sum()
    
        # Display count of values in the top row that are greater than the bottom row
        print("Number of values in the top row greater than in the bottom row:", top_greater_than_bottom_count)
    
        # Proceed with heatmap plotting...
        vmin = heatmap_data.values.min()
        vmax = heatmap_data.values.max()
        
        color_bar_ticks = np.linspace(vmin, vmax, 6)
    
        plt.figure(figsize=(10, 5), dpi=400)
    
        ax = sns.heatmap(
            heatmap_data, cmap=heatmap_cmap, annot=True, fmt=fmt,
            cbar=True, annot_kws={"size": heatmap_value_fontsize},
            cbar_kws={"shrink": 0.6, "ticks": color_bar_ticks},
            square=False, vmin=vmin, vmax=vmax
        )

        colorbar = ax.collections[0].colorbar
        color_format = "{:.2f}".format if data_type == "fraction" else "{:.0f}".format
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: color_format(x)))
        colorbar.ax.set_title("Fraction" if data_type == "fraction" else "Count", fontsize=heatmap_value_fontsize, pad=10, loc='center')

        # Set font style to Times New Roman for axis titles
        plt.xlabel(f"Selected {data_label} models", fontsize=heatmap_xaxis_title_fontsize, fontname="Times New Roman", labelpad=15)
        plt.ylabel("Tissue Samples", fontsize=heatmap_yaxis_title_fontsize, fontname="Times New Roman", labelpad=12)

        centered_y_positions = np.arange(0.5, len(formatted_tissue_labels))
        plt.yticks(ticks=centered_y_positions, labels=formatted_tissue_labels, fontsize=heatmap_yaxis_tick_label_fontsize, fontname="Times New Roman", rotation=0)
        plt.xticks(fontsize=heatmap_xaxis_tick_label_fontsize, rotation=90)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Call the function with your datasets
plot_heatmaps_for_models(selected_tp8_models_F1, data_label="tp8", f_score_type="F1", data_type="fraction")
plot_heatmaps_for_models(selected_tp14_models_F1, data_label="tp14", f_score_type="F1", data_type="fraction")

# Once all heatmaps are plotted, save the statistics DataFrame to a CSV
statistics_df.to_csv("heatmap_statistics.csv", index=False)




def plot_split_diverging_bar_chart(tp8_scores, tp14_scores, tissue, threshold):
    """
    Creates a split diverging bar chart comparing tissue counts of models from tp8 and tp14.
    """
    # Column names for the chosen tissue and threshold
    tp8_col = f"{threshold}_ct_{tissue}"
    tp14_col = f"{threshold}_ct_{tissue}"
    
    def positive_labels(x, pos):
        return f"{abs(int(x))}"

    # Ensure the columns exist and calculate max_count from both sides
    if tp8_col in tp8_scores.columns and tp14_col in tp14_scores.columns:
        max_count = max(tp8_scores[tp8_col].max(), tp14_scores[tp14_col].max())
        tp8_data = tp8_scores[tp8_col].values[::-1]
        tp14_data = tp14_scores[tp14_col].values[::-1]
        
        max_count_padded = max_count * 1.05
        
        # Remove suffixes from model names
        model_names = [name.replace('_tp8', '').replace('_tp14', '') for name in tp8_scores["model name"].values][::-1]

        # Create the figure with adjusted width ratios and spacing
        fig, (ax_left, ax_middle, ax_right) = plt.subplots(
            1, 3, figsize=(12, 12), dpi=300,
            gridspec_kw={'width_ratios': [1, 0.05, 1]},  # Narrow middle axis for closer alignment
        )
        
        # Left axis for tp14 data
        ax_left.xaxis.set_major_formatter(FuncFormatter(positive_labels))
        ax_left.yaxis.tick_right()
        ax_left.tick_params(labelright=True, labelleft=False)
        ax_left.invert_yaxis()
        ax_left.barh(np.arange(len(model_names)), -tp14_data, color="orange", label="tp14", align='center', height=0.9)
        ax_left.set_xlim(-max_count_padded, 0)
        ax_left.set_yticks(np.arange(len(model_names)))
        ax_left.set_yticklabels([])  # Remove y-axis labels from the left
        ax_left.legend(loc="upper right")

        # Right axis for tp8 data
        ax_right.invert_yaxis()
        ax_right.barh(np.arange(len(model_names)), tp8_data, color="blue", label="tp8", align='center', height=0.9)
        ax_right.set_xlim(0, max_count_padded)
        ax_right.set_yticks(np.arange(len(model_names)))
        ax_right.set_yticklabels([])  # Remove y-axis labels from the right
        ax_right.legend(loc="upper left")

        # Middle axis for model names, used only as a label area
        ax_middle.set_yticks(np.arange(len(model_names)) - 3.7)
        ax_middle.set_yticklabels(model_names, ha='center', fontsize=6, rotation=0, va='bottom', x=0.95)
        ax_middle.set_ylim(-9.5, len(model_names) - 0)  # Tighter y-axis limits
        ax_middle.invert_yaxis()  # Model 1 at the top
        ax_middle.set_xticks([])  # No x-axis ticks
        ax_middle.set_xlim(0, 1)  # Set limits for middle axis
        
        # Remove all spines from the middle axis to prevent it from appearing as a box
        for spine in ax_middle.spines.values():
            spine.set_visible(False)

        # Set a unified x-axis label and title
        fig.suptitle(f"Split Diverging Bar Chart of {tissue} Counts at T{threshold}: tp8 vs. tp14", fontsize=16, y=0.94)
        fig.text(0.5, 0.04, "Tissue Count", ha="center", fontsize=12)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.12, wspace=0.12)
        plt.show()
    else:
        print(f"Columns {tp8_col} or {tp14_col} not found in the provided DataFrames.")

# Example usage
plot_split_diverging_bar_chart(tp8_scores, tp14_scores, tissue="panc_spt", threshold="50")
















































# Model orderings based on F1, F0.75, F0.5 scores at each threshold for tp8 and tp14
def get_sorted_models(scores_df, score_type):
    sorted_models = {}
    for threshold in thresholds:
        sorted_models[threshold] = scores_df.sort_values(by=f"{score_type}_{threshold}", ascending=False)["model name"].tolist()
    return sorted_models

# Fetch ordered lists for each score type for tp8 and tp14
tp8_orderings = {
    "F1": get_sorted_models(tp8_scores, "F1"),
    "F0.75": get_sorted_models(tp8_scores, "F0.75"),
    "F0.5": get_sorted_models(tp8_scores, "F0.5")
}
tp14_orderings = {
    "F1": get_sorted_models(tp14_scores, "F1"),
    "F0.75": get_sorted_models(tp14_scores, "F0.75"),
    "F0.5": get_sorted_models(tp14_scores, "F0.5")
}


    # Heatmap settings
new_heatmap_cmap = "Spectral_r"
new_heatmap_xaxis_title_fontsize = 13
new_heatmap_yaxis_title_fontsize = 13
new_heatmap_xaxis_tick_label_fontsize = 9
new_heatmap_yaxis_tick_label_fontsize = 10
new_heatmap_value_fontsize = 7
new_heatmap_subplot_title_fontsize = 14

thresholds = ["50", "60", "70", "80", "90"]
tissue_columns_template = ["bld_spt","bld_igg"]
formatted_tissue_labels = ["sptT1D (bld)", "preT1D (bld)"]
    
# Plotting function with option to order or not
def plot_individual_heatmaps(data_type, scores_df, orderings=None, score_type=None, normalization="none"):
    for threshold_key in thresholds:
        tissue_columns = [f"t{threshold_key}_{tissue}" for tissue in tissue_columns_template]
        
        if score_type:
            # Order models by specified score type
            ordered_models = orderings[score_type][threshold_key]
            x_axis_title = f"{data_type} Models Ordered (L-R) by T{threshold_key} {score_type} Score"
        else:
            # No specific ordering; display models in default order
            ordered_models = scores_df["model name"].tolist()
            x_axis_title = f"{data_type} Models"

        # Prepare heatmap data
        heatmap_data = scores_df[scores_df["model name"].isin(ordered_models)]
        heatmap_data = heatmap_data.set_index("model name").loc[ordered_models][tissue_columns]
        heatmap_data.columns, heatmap_data = tissue_columns_template, heatmap_data.T
        heatmap_data.columns = ordered_models

        # Apply normalization based on the specified option
        if normalization == "column":
            heatmap_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        elif normalization == "row":
            heatmap_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

        # Heatmap color settings
        vmin, vmax = heatmap_data.values.min(), heatmap_data.values.max()
        color_bar_ticks = np.linspace(vmin, vmax, 6)

        # Create a new figure for each heatmap
        fig, ax = plt.subplots(figsize=(15, 6), dpi=600)
        sns.heatmap(heatmap_data, ax=ax, cmap=new_heatmap_cmap, annot=False, fmt=".2f", 
                    cbar=True, annot_kws={"size": new_heatmap_value_fontsize},
                    cbar_kws={"shrink": 0.6, "aspect": 10, "pad": 0.02, "extendfrac": 0.08, "format": "%.2f", "ticks": color_bar_ticks},
                    square=False, vmin=vmin, vmax=vmax)
        
        # Set titles based on whether ordering is applied
        ax.set_title(f"{data_type} Models - Predicted TETpos Fraction at T{threshold_key}", 
                     fontsize=new_heatmap_subplot_title_fontsize, pad=20)
        ax.set_xlabel(x_axis_title, fontsize=new_heatmap_xaxis_title_fontsize, labelpad=12)
        ax.set_ylabel("Experimental Inference Dataset", fontsize=new_heatmap_yaxis_title_fontsize, labelpad=8)
        
        # Set x-axis labels for all models with better alignment
        ax.set_xticks(np.arange(len(ordered_models)) + 0.5)
        ax.set_xticklabels(ordered_models, fontsize=new_heatmap_xaxis_tick_label_fontsize, rotation=90, ha='center', va='top')
        ax.set_yticklabels(formatted_tissue_labels, fontsize=new_heatmap_yaxis_tick_label_fontsize, rotation=0)

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()

# Run the plotting function with and without ordering
# For ordered heatmaps
plot_individual_heatmaps("tp8", tp8_scores, tp8_orderings, score_type="F1", normalization="none")
plot_individual_heatmaps("tp14", tp14_scores, tp14_orderings, score_type="F1", normalization="none")

# For unordered heatmaps
plot_individual_heatmaps("tp8", tp8_scores, tp8_orderings, score_type=None, normalization="none")
plot_individual_heatmaps("tp14", tp14_scores, tp14_orderings, score_type=None, normalization="none")



# Filter models in tp8 and tp14 based on mean_precision and mean_recall criteria
filtered_tp8_scores = tp8_scores[(tp8_scores['mean_precision'] > 0.90) & (tp8_scores['mean_recall'] > 0.50)]
filtered_tp14_scores = tp14_scores[(tp14_scores['mean_recall'] > 0.50)]

# Plotting function for heatmaps without any order, only filtered models
def plot_unordered_heatmaps(data_type, scores_df, normalization="none"):
    for threshold_key in thresholds:
        tissue_columns = [f"t{threshold_key}_{tissue}" for tissue in tissue_columns_template]
        heatmap_data = scores_df[tissue_columns]
        
        # Apply normalization based on the specified option
        if normalization == "column":
            heatmap_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        elif normalization == "row":
            heatmap_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
        
        # Set up heatmap color settings
        vmin, vmax = heatmap_data.values.min(), heatmap_data.values.max()
        color_bar_ticks = np.linspace(vmin, vmax, 6)

        # Plotting
        fig, ax = plt.subplots(figsize=(15, 6), dpi=600)
        sns.heatmap(heatmap_data.T, ax=ax, cmap=new_heatmap_cmap, annot=False, fmt=".2f", 
                    cbar=True, annot_kws={"size": new_heatmap_value_fontsize},
                    cbar_kws={"shrink": 0.6, "aspect": 10, "pad": 0.02, "extendfrac": 0.08, "format": "%.2f", "ticks": color_bar_ticks},
                    square=False, vmin=vmin, vmax=vmax)

        # Titles and labels
        ax.set_title(f"{data_type} Models - Predicted TETpos Fraction at T{threshold_key}", 
                     fontsize=new_heatmap_subplot_title_fontsize, pad=20)
        ax.set_xlabel(f"{data_type} Models", fontsize=12, labelpad=10)
        ax.set_ylabel("Experimental Inference Data", fontsize=12, labelpad=8)
        ax.set_yticklabels(formatted_tissue_labels, fontsize=9, rotation=0)
        
        # Set x-axis ticks to display 1-96 instead of 0-95
        ax.set_xticks(np.arange(heatmap_data.shape[0]) + 0.5)
        ax.set_xticklabels(np.arange(1, heatmap_data.shape[0] + 1), fontsize=7, rotation=90)

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()

# Run the plotting function for the filtered models in tp8 and tp14
plot_unordered_heatmaps("tp8", tp8_scores, normalization="none")
plot_unordered_heatmaps("tp14", tp14_scores, normalization="none")


# ================= TISSUE/TYPE ALTERNATIVE FIGURES - BOXPLOTS ===================

# Customizable plot settings for boxplots
boxplot_settings = {
    "box_colors": ["#1b9e77", "#377eb8", "#e41a1c", "#ff7f00", "#984ea3", "#00bfc4"],  # Colors for each tissue/type
    "title_fontsize": 14,
    "axis_label_fontsize": 12,
    "tick_label_fontsize": 9,
    "outlier_color": "black",
    "outlier_size": 12,
    "outlier_marker": "x",
    "outlier_linewidth": 0.5
}


# Revised function to handle potential mismatches in column names
def plot_boxplots_by_threshold_corrected(data_type, scores_df, thresholds, settings):
    tissue_types = ["pln IgG", "bld IgG", "pan IgG", "pln Spt", "bld Spt", "pan Spt"]
    tissue_columns_template = ["pln_igg", "bld_igg", "panc_igg", "pln_spt", "bld_spt", "panc_spt"]
    
    for threshold in thresholds:
        # Check if the expected columns exist in the DataFrame
        expected_columns = [f"t{threshold}_{tissue}" for tissue in tissue_columns_template]
        missing_columns = [col for col in expected_columns if col not in scores_df.columns]
        if missing_columns:
            print(f"Missing columns for threshold T{threshold}: {missing_columns}")
            continue  # Skip this threshold if columns are missing
        
        # Prepare the data for each tissue type at the current threshold, ensuring correct mapping
        boxplot_data = [scores_df[f"t{threshold}_{tissue}"].values for tissue in tissue_columns_template]
        
        # Check if boxplot_data matches the expected tissue/type order
        print(f"Data preview for T{threshold} ({data_type}):")
        for i, data in enumerate(boxplot_data):
            print(f"{tissue_types[i]}: Mean = {np.mean(data):.3f}, Min = {np.min(data):.3f}, Max = {np.max(data):.3f}")
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        
        # Create the boxplot without default outliers
        sns.boxplot(data=boxplot_data, palette=settings["box_colors"], linewidth=1.0, ax=ax, showfliers=False)
        
        # Add custom jittered outliers
        for i, data in enumerate(boxplot_data):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify outliers
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            if len(outliers) > 0:
                jittered_x = np.random.normal(loc=i, scale=0.13, size=len(outliers))
                jittered_y = outliers + np.random.normal(loc=0, scale=0.005, size=len(outliers))
                ax.scatter(
                    jittered_x, jittered_y,
                    color=settings["outlier_color"],
                    edgecolor="none",
                    s=settings["outlier_size"],
                    marker=settings["outlier_marker"],
                    linewidth=settings["outlier_linewidth"]
                )
        
        # Customization of plot title, labels, and tick marks
        ax.set_title(f"{data_type} Models - Predicted TETpos at T{threshold}", fontsize=settings["title_fontsize"], pad=15)
        ax.set_xlabel("Experimental Inference Data", fontsize=settings["axis_label_fontsize"], labelpad=10)
        ax.set_ylabel("Fraction of Sample", fontsize=settings["axis_label_fontsize"], labelpad=10)
        
        # Set x-ticks and labels
        ax.set_xticks(range(len(tissue_types)))
        ax.set_xticklabels(tissue_types, fontsize=settings["tick_label_fontsize"], rotation=90)
        
        # Set y-ticks font size
        ax.tick_params(axis='y', labelsize=settings["tick_label_fontsize"])
        ax.set_ylim(-0.02, 0.52)
        
        # Add a small amount of buffer above and below using minor ticks
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.tick_params(axis='y', which='minor', length=0) 
    
        # Add grid behind the boxplot for better readability
        ax.grid(True, which='both', axis='y', linestyle="--", linewidth=0.7, color="lightgray", alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.show()

# tp8 and tp14 datasets
plot_boxplots_by_threshold_corrected("tp8", tp8_scores, thresholds, boxplot_settings)
plot_boxplots_by_threshold_corrected("tp14", tp14_scores, thresholds, boxplot_settings)


# ====================== OPTIMIZED STATISTICAL ANALYSIS ==============================

# Dictionary to hold DataFrames for each step
results_dfs = {}

# Step-specific helper function to add results
def add_result_to_step(df_name, analysis, data_type, metric, threshold, description, result, interpretation):
    if df_name not in results_dfs:
        results_dfs[df_name] = pd.DataFrame(columns=["Analysis", "Data Type", "Metric", "Threshold", 
                                                     "Description", "Result", "Interpretation"])
    results_dfs[df_name] = results_dfs[df_name].append({
        "Analysis": analysis,
        "Data Type": data_type,
        "Metric": metric,
        "Threshold": threshold,
        "Description": description,
        "Result": result,
        "Interpretation": interpretation
    }, ignore_index=True)

# Step 1: Variance and Consistency Analysis with Enhanced Plotting and Jittered Outliers
for score_type in ["F1", "F0.75", "F0.5"]:
    for model_type, scores_df in [("tp8", tp8_scores), ("tp14", tp14_scores)]:
        variances = scores_df[[f"{score_type}_{thr}" for thr in ["50", "60", "70", "80", "90"]]].var(axis=1).mean()
        std_devs = scores_df[[f"{score_type}_{thr}" for thr in ["50", "60", "70", "80", "90"]]].std(axis=1).mean()
        
        add_result_to_step("step1_df", "Variance and Consistency Analysis", model_type, score_type, None, 
                           "Variance and standard deviation across thresholds.", 
                           {"Variance": variances, "Standard Deviation": std_devs}, 
                           "Lower variance suggests model stability across thresholds.")      
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.boxplot(data=scores_df[[f"{score_type}_{thr}" for thr in ["50", "60", "70", "80", "90"]]], 
                    linewidth=1.0, ax=ax, showfliers=False)
        for i, threshold in enumerate(["50", "60", "70", "80", "90"]):
            data = scores_df[f"{score_type}_{threshold}"]
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            if not outliers.empty:
                jittered_x = np.random.normal(loc=i, scale=0.10, size=len(outliers))  
                ax.scatter(
                    jittered_x, 
                    outliers,
                    color="black",
                    edgecolor="none",
                    s=15,      
                    marker="x",
                    linewidth=0.6
                )
        ax.grid(True, linestyle='--', linewidth=0.7, color='lightgray', alpha=0.8)
        ax.set_axisbelow(True) 
        ax.set_title(f'{score_type} Variance Across Thresholds ({model_type})', fontsize=16, pad=15)
        ax.set_xlabel("Threshold (T)", fontsize=14, labelpad=10)
        ax.set_ylabel(f"{score_type} score", fontsize=14, labelpad=10)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["T50", "T60", "T70", "T80", "T90"], fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        plt.show()
                
# Step 2: F Score Consistency Analysis (Friedman or Wilcoxon Test)
for data_type, scores_df in [("tp8", tp8_scores), ("tp14", tp14_scores)]:
    for metric in ["F1", "F0.75", "F0.5"]:
        tissue_data = [scores_df[f"{metric}_{thr}"].values for thr in ["50", "60", "70", "80", "90"]]
        
        # Run the Friedman test
        if len(tissue_data) >= 3:
            friedman_stat, friedman_p = friedmanchisquare(*tissue_data)
            add_result_to_step("step3_df", "Friedman Test for Tissue Consistency", data_type, metric, None, 
                               "Friedman test on F score variability across thresholds.", 
                               {"Friedman Statistic": friedman_stat, "P-Value": friedman_p}, 
                               "Significant variability may indicate the need for threshold-specific model selection.")
            
            # If significant, perform pairwise Wilcoxon tests with Bonferroni correction
            if friedman_p < 0.05:
                alpha_corrected = 0.05 / len(list(combinations(range(len(tissue_data)), 2)))
                for (i, j) in combinations(range(len(tissue_data)), 2):
                    thr_i, thr_j = ["50", "60", "70", "80", "90"][i], ["50", "60", "70", "80", "90"][j]
                    stat, p_value = wilcoxon(tissue_data[i], tissue_data[j])
                    comparison_label = f"{thr_i} vs {thr_j}"
                    
                    # Add to results
                    add_result_to_step("step3_df", "Wilcoxon Test for Pairwise Threshold Comparison", data_type, metric, comparison_label, 
                                       f"Wilcoxon test between thresholds {thr_i} and {thr_j}.", 
                                       {"Wilcoxon Statistic": stat, "P-Value": p_value, 
                                        "Significant after correction": p_value < alpha_corrected}, 
                                       "Corrected significance to identify threshold-specific variability.")

        # For two thresholds, Wilcoxon test without Friedman
        elif len(tissue_data) == 2:
            wilcoxon_stat, wilcoxon_p = wilcoxon(tissue_data[0], tissue_data[1])
            add_result_to_step("step3_df", "Wilcoxon Test for Tissue Consistency", data_type, metric, None, 
                               "Wilcoxon test on F score variability across two thresholds.", 
                               {"Wilcoxon Statistic": wilcoxon_stat, "P-Value": wilcoxon_p}, 
                               "Significant variability may indicate the need for threshold-specific model selection.")

# Combine all DataFrames and save to CSV
all_results_df = pd.concat(results_dfs.values(), ignore_index=True)

# Flattening the "Result" dictionary into separate columns
flattened_df = all_results_df.copy()
if 'Result' in flattened_df.columns:
    flattened_df = pd.concat([flattened_df.drop(['Result'], axis=1),
                              flattened_df['Result'].apply(pd.Series)], axis=1)
    print(flattened_df.head())
    flattened_df.to_csv('expanded_statistical_analysis_results.csv', index=False)
else:
    print("The 'Result' column does not exist in all_results_df.")



# Example tissue/type columns
tissue_columns = ["panc_spt", "bld_spt", "pln_spt", "panc_igg", "bld_igg", "pln_igg"]

# Assuming tp8_scores and tp14_scores contain the model data with these columns
def calculate_differences(scores_df, thresholds=["50", "60", "70", "80", "90"]):
    differences = []
    
    for threshold in thresholds:
        spt_columns = [f"t{threshold}_panc_spt", f"t{threshold}_bld_spt", f"t{threshold}_pln_spt"]
        igg_columns = [f"t{threshold}_panc_igg", f"t{threshold}_bld_igg", f"t{threshold}_pln_igg"]
        
        for _, row in scores_df.iterrows():
            # Calculate the averages for Spt and IgG categories for the current threshold
            spt_avg = row[spt_columns].mean()
            igg_avg = row[igg_columns].mean()
            spt_igg_difference = spt_avg - igg_avg
            
            # Calculate panc, pln, bld differences within Spt and IgG categories
            panc_diff = row[f"t{threshold}_panc_spt"] - row[f"t{threshold}_panc_igg"]
            pln_diff = row[f"t{threshold}_pln_spt"] - row[f"t{threshold}_pln_igg"]
            bld_diff = row[f"t{threshold}_bld_spt"] - row[f"t{threshold}_bld_igg"]
            
            differences.append({
                "model": row["model name"],  # Adjust if needed
                "threshold": threshold,
                "spt_igg_diff": spt_igg_difference,
                "panc_diff": panc_diff,
                "pln_diff": pln_diff,
                "bld_diff": bld_diff
            })
    
    # Convert to DataFrame
    return pd.DataFrame(differences)

# Calculate differences for each model in tp8 and tp14
tp8_differences = calculate_differences(tp8_scores)
tp14_differences = calculate_differences(tp14_scores)

# Rank models by each difference measure
tp8_differences["spt_igg_rank"] = tp8_differences["spt_igg_diff"].rank(ascending=False)
tp8_differences["panc_rank"] = tp8_differences["panc_diff"].rank(ascending=False)
tp8_differences["bld_rank"] = tp8_differences["bld_diff"].rank(ascending=False)
tp8_differences["pln_rank"] = tp8_differences["pln_diff"].rank(ascending=False)

tp14_differences["spt_igg_rank"] = tp14_differences["spt_igg_diff"].rank(ascending=False)
tp14_differences["panc_rank"] = tp14_differences["panc_diff"].rank(ascending=False)
tp14_differences["bld_rank"] = tp14_differences["bld_diff"].rank(ascending=False)
tp14_differences["pln_rank"] = tp14_differences["pln_diff"].rank(ascending=False)

# Merge these rankings with F1, F0.75, F0.5, and precision scores for correlation analysis
def merge_and_correlate(differences_df, scores_df):
    # Rename columns to ensure they have the same key for merging
    if 'model' in differences_df.columns:
        differences_df = differences_df.rename(columns={'model': 'model name'})
    elif 'model name' not in differences_df.columns:
        raise ValueError("No 'model' or 'model name' column found in differences_df.")

    # Calculate the average scores across thresholds for each metric
    metrics_df = pd.DataFrame()
    metrics_df["model name"] = scores_df["model name"]
    
    # Calculate the average for each metric (F1, F0.75, F0.5, precision) across thresholds
    for metric in ["F1", "F0.75", "F0.5", "precision"]:
        metric_columns = [f"{metric}_{thr}" for thr in ["50", "60", "70", "80", "90"]]
        metrics_df[metric] = scores_df[metric_columns].mean(axis=1)
    
    # Merge differences and metrics dataframes on "model name"
    merged_df = pd.merge(differences_df, metrics_df, on="model name", how="inner")
    
    # Calculate correlation matrix
    correlation_matrix = merged_df.corr()
    return correlation_matrix

# Perform correlation analysis for tp8 and tp14
tp8_correlation_matrix = merge_and_correlate(tp8_differences, tp8_scores)
tp14_correlation_matrix = merge_and_correlate(tp14_differences, tp14_scores)

# Display correlation matrices
print("Correlation matrix for tp8:")
print(tp8_correlation_matrix)
print("\nCorrelation matrix for tp14:")
print(tp14_correlation_matrix)

# Visualization settings
heatmap_settings = {
    "cmap": "coolwarm",
    "annot": True,
    "fmt": ".2f",
    "linewidths": 0.5,
    "cbar_kws": {"shrink": 0.8},
    "square": True,
}

# Plot heatmap for tp8
plt.figure(figsize=(10, 8))
sns.heatmap(tp8_correlation_matrix, **heatmap_settings)
plt.title("Rank Correlation Heatmap: tp8 F-Scores and Tissue/Type Differences")
plt.show()

# Plot heatmap for tp14
plt.figure(figsize=(10, 8))
sns.heatmap(tp14_correlation_matrix, **heatmap_settings)
plt.title("Rank Correlation Heatmap: tp14 F-Scores and Tissue/Type Differences")
plt.show()