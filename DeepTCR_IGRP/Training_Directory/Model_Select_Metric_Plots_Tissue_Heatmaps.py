# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 01:56:23 2024

@author: Mitch
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ==================== 1. IMPORT DATA FOR TP8 and TP14 ==============================

# define tp8 & tp14 metrics_threshold CSV files within directory
tp8_metrics_dir = os.path.join(os.getcwd(), "tp8_metrics")
tp8_files = [
    "tp8_metrics_50.csv", 
    "tp8_metrics_60.csv", 
    "tp8_metrics_70.csv", 
    "tp8_metrics_80.csv", 
    "tp8_metrics_90.csv"
]

tp14_metrics_dir = os.path.join(os.getcwd(), "tp14_metrics")
tp14_files = [
    "tp14_metrics_50.csv", 
    "tp14_metrics_60.csv", 
    "tp14_metrics_70.csv", 
    "tp14_metrics_80.csv", 
    "tp14_metrics_90.csv"
]

# Define thresholds and tissue/type columns, load CSVs
thresholds = ["50", "60", "70", "80", "90"]
tissue_columns = ["panc_spt", "bld_spt", "pln_spt", "panc_igg", "bld_igg", "pln_igg"]

tp8_data = {}
for file in tp8_files:
    threshold = file.split("_")[-1].split(".")[0]  # extract threshold from filename
    df = pd.read_csv(os.path.join(tp8_metrics_dir, file))
    tp8_data[threshold] = df  # store dataframe in dictionary under threshold key

tp14_data = {}
for file in tp14_files:
    threshold = file.split("_")[-1].split(".")[0] # same for tp14  
    df = pd.read_csv(os.path.join(tp14_metrics_dir, file))
    tp14_data[threshold] = df  

# ========================= 2. CREATE SCORES DATAFRAMES =============================
tp8_scores = pd.DataFrame()
tp14_scores = pd.DataFrame()

def initialize_score_dataframe(data, thresholds, group_name):
    models = data[thresholds[0]]["model name"].tolist()
    score_df = pd.DataFrame(models, columns=["model name"]) 
    for threshold in thresholds:
        score_df[f"F1_{threshold}"] = None
        score_df[f"F0.75_{threshold}"] = None
        score_df[f"F0.5_{threshold}"] = None   
    return score_df

tp8_scores = initialize_score_dataframe(tp8_data, thresholds, "tp8")
tp14_scores = initialize_score_dataframe(tp14_data, thresholds, "tp14")

# Populate score DataFrames with F1, F0.75, and F0.5 scores from each threshold-specific dataFrame
def populate_scores(score_df, data, thresholds):
    for threshold in thresholds:
        df = data[threshold]
        # Map F1, F0.75, and F0.5 scores into the columns
        score_df[f"F1_{threshold}"] = df["F1 score"].values
        score_df[f"F0.75_{threshold}"] = df["F0.75 score"].values
        score_df[f"F0.5_{threshold}"] = df["F0.5 score"].values
    
populate_scores(tp8_scores, tp8_data, thresholds)
populate_scores(tp14_scores, tp14_data, thresholds)

# add mean columns for F1, F0.75, and F0.5 scores
def add_mean_scores(df):
    df['mean_F1'] = df[[f"F1_{threshold}" for threshold in thresholds]].mean(axis=1)
    df['mean_F0.75'] = df[[f"F0.75_{threshold}" for threshold in thresholds]].mean(axis=1)
    df['mean_F0.5'] = df[[f"F0.5_{threshold}" for threshold in thresholds]].mean(axis=1)

add_mean_scores(tp8_scores)
add_mean_scores(tp14_scores)

# add precision, recall, and accuracy columns
def add_additional_metrics(score_df, data, thresholds):
    for threshold in thresholds:
        df = data[threshold]
        score_df[f"precision_{threshold}"] = df["precision"].values
        score_df[f"recall_{threshold}"] = df["recall"].values
        score_df[f"accuracy_{threshold}"] = df["accuracy"].values  
    score_df['mean_precision'] = score_df[[f"precision_{threshold}" for threshold in thresholds]].mean(axis=1)
    score_df['mean_recall'] = score_df[[f"recall_{threshold}" for threshold in thresholds]].mean(axis=1)
    score_df['mean_accuracy'] = score_df[[f"accuracy_{threshold}" for threshold in thresholds]].mean(axis=1)

add_additional_metrics(tp8_scores, tp8_data, thresholds)
add_additional_metrics(tp14_scores, tp14_data, thresholds)

# calculate and add variance and std deviation columns to the score DataFrame
def add_consistency_metrics(df):
    for score_type in ["F1", "F0.75", "F0.5"]:
        # Calculate variance and standard deviation across thresholds
        df[f"{score_type}_variance"] = df[[f"{score_type}_{threshold}" for threshold in thresholds]].var(axis=1)
        df[f"{score_type}_std_dev"] = df[[f"{score_type}_{threshold}" for threshold in thresholds]].std(axis=1)
    for metric in ["precision", "recall", "accuracy"]:
        df[f"{metric}_variance"] = df[[f"{metric}_{threshold}" for threshold in thresholds]].var(axis=1)
        df[f"{metric}_std_dev"] = df[[f"{metric}_{threshold}" for threshold in thresholds]].std(axis=1)

add_consistency_metrics(tp8_scores)
add_consistency_metrics(tp14_scores)

# ---------- TISSUE/TYPE DATA -----------
# add tissue/type data to scores DataFrame
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

tp8_scores = add_tissue_type_data(tp8_scores, tp8_data, thresholds)
tp14_scores = add_tissue_type_data(tp14_scores, tp14_data, thresholds)

# Function to add tissue/type total nseq to scores DataFrame
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

tp8_scores = calculate_counts(tp8_scores, tissue_columns, thresholds)
tp14_scores = calculate_counts(tp14_scores, tissue_columns, thresholds)

# ----------- SELECTED MODEL FUNCTION -----------
def select_top_models(tp_scores_df, score_type="F0.5", num_models=10, mean_performance_threshold=0.5, std_dev_threshold=0.1):
    mean_column = f"mean_{score_type}"
    std_dev_column = f"{score_type}_std_dev"
    filtered_df = tp_scores_df[
        (tp_scores_df[mean_column] >= mean_performance_threshold) &
        (tp_scores_df[std_dev_column] <= std_dev_threshold)
    ]
    sorted_df = filtered_df.sort_values(by=mean_column, ascending=False)
    top_models_df = sorted_df.head(num_models)
    return top_models_df

selected_tp14_models_F = select_top_models(
    tp14_scores, 
    score_type="F0.5", 
    num_models=24,  
    mean_performance_threshold=0.685,
)

selected_tp8_models_F = select_top_models(
    tp8_scores, 
    score_type="F0.5", 
    num_models=24,
    mean_performance_threshold=0.685,
)

# =================== PLOT SELECTIONS BY F SCORE, PRECISION, RECALL ==========================
# Define a consistent color palette for main plot and legend
combined_color_palette = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + plt.cm.tab20b(np.linspace(0, 1, 4)).tolist()

def plot_legend_only(color_palette, model_names):
    plt.figure(figsize=(2, 6), dpi=100)
    for idx, model in enumerate(model_names):
        plt.plot([], [], color=color_palette[idx], marker='o', markersize=8, linestyle='-', linewidth=1.5, label=model)
    plt.legend(title="Models", loc="center", prop={'size': 8}, title_fontsize='9')
    plt.axis('off') 
    plt.show()

def plot_scores_across_thresholds(selected_models, all_models, thresholds, select_score, plot_metric, data_label="tp8", color_map=None, x_min=60, x_max=90, y_min=0.7, y_max=0.85):
    if color_map is None:
        color_map = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + plt.cm.tab20b(np.linspace(0, 1, 4)).tolist()
    plt.figure(figsize=(8, 6), dpi=400)
    for model in all_models['model name']:
        if model not in selected_models['model name'].values:
            scores = [all_models.loc[all_models['model name'] == model, f"{plot_metric}_{threshold}"].values[0] for threshold in thresholds]
            plt.plot(thresholds, scores, color='#E8E8E8', alpha=1, linewidth=1.0, zorder=1)
    for idx, model in enumerate(selected_models['model name']):
        scores = [selected_models.loc[selected_models['model name'] == model, f"{plot_metric}_{threshold}"].values[0] for threshold in thresholds]
        color = color_map[idx]
        plt.plot(thresholds, scores, color=color, linewidth=1.2, label=model, zorder=2)
        plt.scatter(thresholds, scores, color=color, s=20, zorder=3) 
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min, y_max)
    plt.xticks(thresholds)  
    plt.yticks(np.arange(y_min, y_max + 0.05, 0.05))  
    if plot_metric == "F1":
        title_metric = "F1 Scores"
    elif plot_metric == "precision":
        title_metric = "Precision (PPV)"
    elif plot_metric == "recall":
        title_metric = "Recall/Sensitivity (TPR)"
    else:
        title_metric = f"{plot_metric.capitalize()} Scores" 
    plt.xlabel("Threshold (T)", fontsize=13, labelpad=10)
    plt.ylabel(f"{title_metric}", fontsize=13, labelpad=10)
    plt.title(f"Top 24 Models ({data_label.capitalize()}) by {select_score} T(mean): {title_metric}", fontsize=16, pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

thresholds = [50, 60, 70, 80, 90]

plot_legend_only(combined_color_palette, selected_tp14_models_F['model name'].tolist())
plot_legend_only(combined_color_palette, selected_tp8_models_F['model name'].tolist())

plot_scores_across_thresholds(selected_tp14_models_F, tp14_scores, thresholds, select_score="F0.5", plot_metric="F0.5", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.45, y_max=0.85)
plot_scores_across_thresholds(selected_tp14_models_F, tp14_scores, thresholds, select_score="F0.5", plot_metric="precision", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.7, y_max=0.95)
plot_scores_across_thresholds(selected_tp14_models_F, tp14_scores, thresholds, select_score="F0.5", plot_metric="recall", data_label="tp14", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.35, y_max=0.8)
plot_scores_across_thresholds(selected_tp8_models_F, tp8_scores, thresholds, select_score="F0.5", plot_metric="F0.5", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.45, y_max=0.85)
plot_scores_across_thresholds(selected_tp8_models_F, tp8_scores, thresholds, select_score="F0.5", plot_metric="precision", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.7, y_max=0.95)
plot_scores_across_thresholds(selected_tp8_models_F, tp8_scores, thresholds, select_score="F0.5", plot_metric="recall", data_label="tp8", color_map=combined_color_palette, x_min=50, x_max=90, y_min=0.35, y_max=0.8)

# =================== PLOT SELECTIONS HEATMAPS ==========================
statistics_df = pd.DataFrame()
def plot_heatmaps_for_models(selected_models, data_label, f_score_type, data_type="count"):
    global statistics_df 
    heatmap_cmap = "magma"
    heatmap_xaxis_title_fontsize = 16
    heatmap_yaxis_title_fontsize = 16
    heatmap_xaxis_tick_label_fontsize = 7
    heatmap_yaxis_tick_label_fontsize = 12
    heatmap_value_fontsize = 7.5
    thresholds = ["50", "60", "70", "80", "90"]
    tissue_columns_template = ["pln_spt", "pln_igg"] # manually change data name for each tissue type and axis title below
    formatted_tissue_labels = ["Diabetic pLN", "Pre-diabetic pLN"] 
    std_dev_column = "mean_F0.5"
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
        row_medians = heatmap_data.median(axis=1)
        row_q1 = heatmap_data.quantile(0.25, axis=1)
        row_q3 = heatmap_data.quantile(0.75, axis=1)
        top_greater_than_bottom_count = (heatmap_data.iloc[0] > heatmap_data.iloc[-1]).sum()
        for row_name, median, q1, q3 in zip(heatmap_data.index, row_medians, row_q1, row_q3):
            statistics_df = statistics_df.append({
                "data_label": data_label,
                "threshold": threshold_key,
                "sample_name": row_name,
                "median": median,
                "Q1": q1,
                "Q3": q3,
                "top_greater_than_bottom_count": top_greater_than_bottom_count
            }, ignore_index=True)
        print("Row Medians:\n", row_medians)
        print("Row Q1s:\n", row_q1)
        print("Row Q3s:\n", row_q3)  
        top_row = heatmap_data.iloc[0]
        bottom_row = heatmap_data.iloc[-1]
        top_greater_than_bottom_count = (top_row > bottom_row).sum() 
        print("Number of values in the top row greater than in the bottom row:", top_greater_than_bottom_count)
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
        plt.xlabel(f"Selected {data_label} models", fontsize=heatmap_xaxis_title_fontsize, fontname="Times New Roman", labelpad=15)
        plt.ylabel("Tissue Samples", fontsize=heatmap_yaxis_title_fontsize, fontname="Times New Roman", labelpad=12)
        centered_y_positions = np.arange(0.5, len(formatted_tissue_labels))
        plt.yticks(ticks=centered_y_positions, labels=formatted_tissue_labels, fontsize=heatmap_yaxis_tick_label_fontsize, fontname="Times New Roman", rotation=0)
        plt.xticks(fontsize=heatmap_xaxis_tick_label_fontsize, rotation=90)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Call function 
plot_heatmaps_for_models(selected_tp8_models_F, data_label="tp8", f_score_type="F1", data_type="fraction")
plot_heatmaps_for_models(selected_tp14_models_F, data_label="tp14", f_score_type="F1", data_type="fraction")

# MANUAL REPEAT HEATMAPS UNTIL ALL PLOTTED ------------

# Once heatmaps plotted, save statistics DataFrame to CSV
statistics_df.to_csv("heatmap_statistics.csv", index=False)





















































