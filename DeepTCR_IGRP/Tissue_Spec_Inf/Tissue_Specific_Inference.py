# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:49:55 2024
@author: Mitch
"""
# venv = IGRP_DTCR (py3.7.12, deepTCR2.1.0, tf2.7, CUDAtk11.2, cuDNN8.1, fastcluster1.2.6)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from DeepTCR.DeepTCR import DeepTCR_WF
import pandas as pd
import numpy as np
import os
import inspect

def list_model_folders_directory():
    directory = f"{model_group_path}"
    model_folder_list = [f.name for f in directory.iterdir() if f.is_dir()]
    
    print(model_folder_list)
    
def modify_df(inference_pred):
    # Use DTCR_WF's internal attributes for sample list and prediction probabilities
    df_infpred = pd.DataFrame(DTCR_WF.Inference_Pred)
    return df_infpred

def set_inference_trained_name(training_object_name):
    global DTCR_WF
    DTCR_WF = DeepTCR_WF(training_object_name)

def inference_bld_igg():
    inference_data_file = f'{base_path}/WF_DeepTCR/Data_valid/Collier_2023/bld_IgG_cd8'
    inference_name = "bld_igg"
    DTCR_WF.Get_Data(directory=inference_data_file,
                     Load_Prev_Data=False,
                     aggregate_by_aa=False,
                     sep=',',
                     aa_column_beta=6,
                     v_beta_column=2,
                     d_beta_column=3,
                     j_beta_column=4)
    print("inference_bld_igg data input successful")
    return inference_name

def inference_bld_spt():
   inference_data_file = f'{base_path}/WF_DeepTCR/Data_valid/Collier_2023/bld_Spt_cd8'
   inference_name = "bld_spt"
   DTCR_WF.Get_Data(directory=inference_data_file,
                    Load_Prev_Data=False,
                    aggregate_by_aa=False,
                    sep=',',
                    aa_column_beta=6,
                    v_beta_column=2,
                    d_beta_column=3,
                    j_beta_column=4)
   print("inference_bld_spt data input successful")
   return inference_name

def inference_pln_igg():
   inference_data_file = f'{base_path}/WF_DeepTCR/Data_valid/Collier_2023/pLN_IgG_cd8'
   inference_name = "pln_igg"
   DTCR_WF.Get_Data(directory=inference_data_file,
                    Load_Prev_Data=False,
                    aggregate_by_aa=False,
                    sep=',',
                    aa_column_beta=6,
                    v_beta_column=2,
                    d_beta_column=3,
                    j_beta_column=4)
   print("inference_pln_igg data input successful")
   return inference_name

def inference_pln_spt():
   inference_data_file = f'{base_path}/WF_DeepTCR/Data_valid/Collier_2023/pLN_Spt_cd8'
   inference_name = "pln_spt"
   DTCR_WF.Get_Data(directory=inference_data_file,
                    Load_Prev_Data=False,
                    aggregate_by_aa=False,
                    sep=',',
                    aa_column_beta=6,
                    v_beta_column=2,
                    d_beta_column=3,
                    j_beta_column=4)
   print("inference_pln_spt data input successful")
   return inference_name

def inference_panc_igg():
   inference_data_file = f'{base_path}/WF_DeepTCR/Data_valid/Collier_2023/pan_IgG_cd8'
   inference_name = "panc_igg"
   DTCR_WF.Get_Data(directory=inference_data_file,
                    Load_Prev_Data=False,
                    aggregate_by_aa=False,
                    sep=',',
                    aa_column_beta=6,
                    v_beta_column=2,
                    d_beta_column=3,
                    j_beta_column=4)
   print("inference_panc_igg data input successful")
   return inference_name

def inference_panc_spt():
   inference_data_file = f'{base_path}/WF_DeepTCR/Data_valid/Collier_2023/pan_Spt_cd8'
   inference_name = "panc_spt"
   DTCR_WF.Get_Data(directory=inference_data_file,
                    Load_Prev_Data=False,
                    aggregate_by_aa=False,
                    sep=',',
                    aa_column_beta=6,
                    v_beta_column=2,
                    d_beta_column=3,
                    j_beta_column=4)
   print("inference_panc_spt data input successful")
   return inference_name

def set_sample_inference():
    beta_sequences = DTCR_WF.beta_sequences
    v_beta = DTCR_WF.v_beta
    d_beta = DTCR_WF.d_beta
    j_beta = DTCR_WF.j_beta
    return beta_sequences, v_beta, d_beta, j_beta
         
def run_sample_inference(beta_sequences, v_beta, d_beta, j_beta):
    """
    Runs sample inference and calculates the AUC.
    """
    # Perform sample inference first
    DTCR_WF.Sample_Inference(sample_labels=None, 
                             beta_sequences=beta_sequences, 
                             v_beta=v_beta, 
                             d_beta=d_beta, 
                             j_beta=j_beta,
                             batch_size=1000,  
                             return_dist=False)  
    
def calculate_inf_predictions_thresh(df_infpred, training_object_name, threshold, inference_name):
    pred_column = f'Pred_{threshold}'
    
    # Access second column by integer indexing
    df_infpred[pred_column] = np.where(df_infpred[1] > threshold / 100.0, 'TETpos', 'TETneg')
    
    # Count TETpos values present
    tetpos_fraction = df_infpred[pred_column].value_counts(normalize=True).get('TETpos', 0)
    
    return tetpos_fraction
   
    
# =======================================================================================================================

# Model group folder name
model_group_folder_name = 'Supervised_WF'
# base path
base_path = 'D:/Mitch/Documents/DeepTCR_IGRP/Supervised_WF'
# model group folder path
model_group_path = f'D:/Mitch/Documents/DeepTCR_IGRP/{model_group_folder_name}'

# model group log and inf stats folder if using normal IGRP_dist_test_1_ctvt
# log_inf_stats_folder = f'{model_group_path}/1_log_inference_add_dist'

# model group log and inference stats folder if using IGRP_add_dist_test_1_ctvt
log_inf_stats_folder = f'{model_group_path}/1_log_inference_dist'

# model group log and inference stats folder if using IGRP_dist_50_test_1_ctvt
# log_inf_stats_folder = f'{model_group_path}/1_log_inference_dist_50'

# =======================================================================================================================

# metric folders for thresh
t50_csv_path = f'{log_inf_stats_folder}/tp8_metrics_50.csv'
t60_csv_path = f'{log_inf_stats_folder}/tp8_metrics_60.csv'
t70_csv_path = f'{log_inf_stats_folder}/tp8_metrics_70.csv'
t80_csv_path = f'{log_inf_stats_folder}/tp8_metrics_80.csv'
t90_csv_path = f'{log_inf_stats_folder}/tp8_metrics_90.csv'

# metric folders for thresh
t50_csv_path = f'{log_inf_stats_folder}/tp14_metrics_50.csv'
t60_csv_path = f'{log_inf_stats_folder}/tp14_metrics_60.csv'
t70_csv_path = f'{log_inf_stats_folder}/tp14_metrics_70.csv'
t80_csv_path = f'{log_inf_stats_folder}/tp14_metrics_80.csv'
t90_csv_path = f'{log_inf_stats_folder}/tp14_metrics_90.csv'

# Get sorted list of folders that start with "IGRP_WF_"
folders = sorted([f for f in os.listdir(model_group_path) if f.startswith("IGRP_WF_") and os.path.isdir(os.path.join(model_group_path, f))])

# Print list of folder names
print("List of folders for processing:")
for folder in folders:
    print(f"- {folder}")

# Manual check, confirm to proceed with inference
proceed = input("Do you want to proceed with inference for these folders? Enter True to proceed or False to exit: ")

# Convert input to boolean
if proceed.lower() == 'true':
    print("Proceeding with inference...")

    # Loop through each folder, assign training_object_name
    thresholds = [50, 60, 70, 80, 90]

    # Initialize empty DataFrames to store results for each inference type
    inf_frac_bld_igg = pd.DataFrame(columns=['model name', 't50_bld_igg', 't60_bld_igg', 't70_bld_igg', 't80_bld_igg', 't90_bld_igg', 'nseq_bld_igg'])
    inf_frac_bld_spt = pd.DataFrame(columns=['model name', 't50_bld_spt', 't60_bld_spt', 't70_bld_spt', 't80_bld_spt', 't90_bld_spt', 'nseq_bld_spt'])
    inf_frac_pln_igg = pd.DataFrame(columns=['model name', 't50_pln_igg', 't60_pln_igg', 't70_pln_igg', 't80_pln_igg', 't90_pln_igg', 'nseq_pln_igg'])
    inf_frac_pln_spt = pd.DataFrame(columns=['model name', 't50_pln_spt', 't60_pln_spt', 't70_pln_spt', 't80_pln_spt', 't90_pln_spt', 'nseq_pln_spt'])
    inf_frac_panc_igg = pd.DataFrame(columns=['model name', 't50_panc_igg', 't60_panc_igg', 't70_panc_igg', 't80_panc_igg', 't90_panc_igg', 'nseq_panc_igg'])
    inf_frac_panc_spt = pd.DataFrame(columns=['model name', 't50_panc_spt', 't60_panc_spt', 't70_panc_spt', 't80_panc_spt', 't90_panc_spt', 'nseq_panc_spt'])

    # Loop over folders (training_object_names)
    for i, folder in enumerate(folders, start=1):
        training_object_name = folder
        set_inference_trained_name(training_object_name)
        
        # Run all inferences within the same loop for each training_object_name
        
    # ---- Process inference_bld_igg ----
        inference_name = inference_bld_igg()  # This returns 'bld_igg'
        
        # Run inference steps
        beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
        run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
        df_infpred = modify_df(DTCR_WF.Inference_Pred)
        
        # Create a new row for this training_object_name
        new_row_igg = {'model name': training_object_name}
        total_seq = df_infpred.shape[0]
        new_row_igg['nseq_bld_igg'] = total_seq
        
        # Loop through thresholds for bld_igg
        for thresh in thresholds:
            tetpos_fraction = calculate_inf_predictions_thresh(df_infpred, training_object_name, thresh, inference_name)
            new_row_igg[f't{thresh}_{inference_name}'] = tetpos_fraction
        
        # Append the new row to the inf_frac_bld_igg DataFrame
        inf_frac_bld_igg = inf_frac_bld_igg.append(new_row_igg, ignore_index=True)
        
    # ---- Process inference_bld_spt as above ----
        inference_name = inference_bld_spt() 
        beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
        run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
        df_infpred = modify_df(DTCR_WF.Inference_Pred)
        new_row_spt = {'model name': training_object_name}
        total_seq = df_infpred.shape[0]
        new_row_spt['nseq_bld_spt'] = total_seq
        for thresh in thresholds:
            tetpos_fraction = calculate_inf_predictions_thresh(df_infpred, training_object_name, thresh, inference_name)
            new_row_spt[f't{thresh}_{inference_name}'] = tetpos_fraction
            
        inf_frac_bld_spt = inf_frac_bld_spt.append(new_row_spt, ignore_index=True)
    
    # ---- Process inference_pln_igg ----
        inference_name = inference_pln_igg()
        beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
        run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
        df_infpred = modify_df(DTCR_WF.Inference_Pred)
        new_row_igg_pln = {'model name': training_object_name}
        total_seq = df_infpred.shape[0]
        new_row_igg_pln['nseq_pln_igg'] = total_seq
        for thresh in thresholds:
            tetpos_fraction = calculate_inf_predictions_thresh(df_infpred, training_object_name, thresh, inference_name)
            new_row_igg_pln[f't{thresh}_{inference_name}'] = tetpos_fraction
            
        inf_frac_pln_igg = inf_frac_pln_igg.append(new_row_igg_pln, ignore_index=True)

    # ---- Process inference_pln_spt ----
        inference_name = inference_pln_spt() 
        beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
        run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
        df_infpred = modify_df(DTCR_WF.Inference_Pred)
        new_row_spt_pln = {'model name': training_object_name}
        total_seq = df_infpred.shape[0]
        new_row_spt_pln['nseq_pln_spt'] = total_seq
        for thresh in thresholds:
            tetpos_fraction = calculate_inf_predictions_thresh(df_infpred, training_object_name, thresh, inference_name)
            new_row_spt_pln[f't{thresh}_{inference_name}'] = tetpos_fraction
        
        inf_frac_pln_spt = inf_frac_pln_spt.append(new_row_spt_pln, ignore_index=True)
          
    # ---- Process inference_panc_igg ----
        inference_name = inference_panc_igg() 
        beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
        run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
        df_infpred = modify_df(DTCR_WF.Inference_Pred)
        new_row_igg_panc = {'model name': training_object_name}
        total_seq = df_infpred.shape[0]
        new_row_igg_panc['nseq_panc_igg'] = total_seq
        for thresh in thresholds:
            tetpos_fraction = calculate_inf_predictions_thresh(df_infpred, training_object_name, thresh, inference_name)
            new_row_igg_panc[f't{thresh}_{inference_name}'] = tetpos_fraction
        
        inf_frac_panc_igg = inf_frac_panc_igg.append(new_row_igg_panc, ignore_index=True)
           
    # ---- Process inference_panc_spt ----
        inference_name = inference_panc_spt()
        beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
        run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
        df_infpred = modify_df(DTCR_WF.Inference_Pred)
        new_row_spt_panc = {'model name': training_object_name}
        total_seq = df_infpred.shape[0]
        new_row_spt_panc['nseq_panc_spt'] = total_seq
        for thresh in thresholds:
            tetpos_fraction = calculate_inf_predictions_thresh(df_infpred, training_object_name, thresh, inference_name)
            new_row_spt_panc[f't{thresh}_{inference_name}'] = tetpos_fraction
        
        inf_frac_panc_spt = inf_frac_panc_spt.append(new_row_spt_panc, ignore_index=True)
    
    # Save DataFrames to CSV after all loops are complete
    inf_frac_bld_igg.to_csv(f'{log_inf_stats_folder}/inf_frac_bld_igg.csv', index=False)
    inf_frac_bld_spt.to_csv(f'{log_inf_stats_folder}/inf_frac_bld_spt.csv', index=False)
    inf_frac_pln_igg.to_csv(f'{log_inf_stats_folder}/inf_frac_pln_igg.csv', index=False)
    inf_frac_pln_spt.to_csv(f'{log_inf_stats_folder}/inf_frac_pln_spt.csv', index=False)
    inf_frac_panc_igg.to_csv(f'{log_inf_stats_folder}/inf_frac_panc_igg.csv', index=False)
    inf_frac_panc_spt.to_csv(f'{log_inf_stats_folder}/inf_frac_panc_spt.csv', index=False)

else:
    print("Exiting without running inference.")

dataframes = [inf_frac_bld_igg, inf_frac_bld_spt, inf_frac_pln_igg, inf_frac_pln_spt, inf_frac_panc_igg, inf_frac_panc_spt]
for name, df in inspect.currentframe().f_globals.items():
    if name.startswith("inf_frac_") and isinstance(df, pd.DataFrame):
        # Remove "_dist" and add "_tp8" to each value in the 'model name' column
        df["model name"] = df["model name"].str.replace("_add_dist", "", regex=False) + "_tp14"


# Load Metrics_overview_50.csv
metrics_overview_50 = pd.read_csv(t50_csv_path)

# Merge t50_ and nseq_ columns from all inf_frac_ DataFrames by matching 'model name'
metrics_overview_50 = pd.merge(metrics_overview_50, inf_frac_bld_igg[['model name', 't50_bld_igg', 'nseq_bld_igg']], on='model name', how='left')
metrics_overview_50 = pd.merge(metrics_overview_50, inf_frac_bld_spt[['model name', 't50_bld_spt', 'nseq_bld_spt']], on='model name', how='left')
metrics_overview_50 = pd.merge(metrics_overview_50, inf_frac_pln_igg[['model name', 't50_pln_igg', 'nseq_pln_igg']], on='model name', how='left')
metrics_overview_50 = pd.merge(metrics_overview_50, inf_frac_pln_spt[['model name', 't50_pln_spt', 'nseq_pln_spt']], on='model name', how='left')
metrics_overview_50 = pd.merge(metrics_overview_50, inf_frac_panc_igg[['model name', 't50_panc_igg', 'nseq_panc_igg']], on='model name', how='left')
metrics_overview_50 = pd.merge(metrics_overview_50, inf_frac_panc_spt[['model name', 't50_panc_spt', 'nseq_panc_spt']], on='model name', how='left')

# Save back to the original CSV
metrics_overview_50.to_csv(t50_csv_path, index=False)

# Load Metrics_overview_60.csv
metrics_overview_60 = pd.read_csv(t60_csv_path)

# Merge t60_ and nseq_ columns for each inf_frac_ DataFrame
metrics_overview_60 = pd.merge(metrics_overview_60, inf_frac_bld_igg[['model name', 't60_bld_igg', 'nseq_bld_igg']], on='model name', how='left')
metrics_overview_60 = pd.merge(metrics_overview_60, inf_frac_bld_spt[['model name', 't60_bld_spt', 'nseq_bld_spt']], on='model name', how='left')
metrics_overview_60 = pd.merge(metrics_overview_60, inf_frac_pln_igg[['model name', 't60_pln_igg', 'nseq_pln_igg']], on='model name', how='left')
metrics_overview_60 = pd.merge(metrics_overview_60, inf_frac_pln_spt[['model name', 't60_pln_spt', 'nseq_pln_spt']], on='model name', how='left')
metrics_overview_60 = pd.merge(metrics_overview_60, inf_frac_panc_igg[['model name', 't60_panc_igg', 'nseq_panc_igg']], on='model name', how='left')
metrics_overview_60 = pd.merge(metrics_overview_60, inf_frac_panc_spt[['model name', 't60_panc_spt', 'nseq_panc_spt']], on='model name', how='left')

# Save back to CSV
metrics_overview_60.to_csv(t60_csv_path, index=False)

# Load Metrics_overview_70.csv
metrics_overview_70 = pd.read_csv(t70_csv_path)

# Merge t70_ and nseq_ columns from all inf_frac_ DataFrames by matching 'model name'
metrics_overview_70 = pd.merge(metrics_overview_70, inf_frac_bld_igg[['model name', 't70_bld_igg', 'nseq_bld_igg']], on='model name', how='left')
metrics_overview_70 = pd.merge(metrics_overview_70, inf_frac_bld_spt[['model name', 't70_bld_spt', 'nseq_bld_spt']], on='model name', how='left')
metrics_overview_70 = pd.merge(metrics_overview_70, inf_frac_pln_igg[['model name', 't70_pln_igg', 'nseq_pln_igg']], on='model name', how='left')
metrics_overview_70 = pd.merge(metrics_overview_70, inf_frac_pln_spt[['model name', 't70_pln_spt', 'nseq_pln_spt']], on='model name', how='left')
metrics_overview_70 = pd.merge(metrics_overview_70, inf_frac_panc_igg[['model name', 't70_panc_igg', 'nseq_panc_igg']], on='model name', how='left')
metrics_overview_70 = pd.merge(metrics_overview_70, inf_frac_panc_spt[['model name', 't70_panc_spt', 'nseq_panc_spt']], on='model name', how='left')

# Save back to the original CSV
metrics_overview_70.to_csv(t70_csv_path, index=False)

# Load Metrics_overview_80.csv
metrics_overview_80 = pd.read_csv(t80_csv_path)

# Merge t80_ and nseq_ columns from all inf_frac_ DataFrames by matching 'model name'
metrics_overview_80 = pd.merge(metrics_overview_80, inf_frac_bld_igg[['model name', 't80_bld_igg', 'nseq_bld_igg']], on='model name', how='left')
metrics_overview_80 = pd.merge(metrics_overview_80, inf_frac_bld_spt[['model name', 't80_bld_spt', 'nseq_bld_spt']], on='model name', how='left')
metrics_overview_80 = pd.merge(metrics_overview_80, inf_frac_pln_igg[['model name', 't80_pln_igg', 'nseq_pln_igg']], on='model name', how='left')
metrics_overview_80 = pd.merge(metrics_overview_80, inf_frac_pln_spt[['model name', 't80_pln_spt', 'nseq_pln_spt']], on='model name', how='left')
metrics_overview_80 = pd.merge(metrics_overview_80, inf_frac_panc_igg[['model name', 't80_panc_igg', 'nseq_panc_igg']], on='model name', how='left')
metrics_overview_80 = pd.merge(metrics_overview_80, inf_frac_panc_spt[['model name', 't80_panc_spt', 'nseq_panc_spt']], on='model name', how='left')

# Save back to the original CSV
metrics_overview_80.to_csv(t80_csv_path, index=False)

# Load Metrics_overview_90.csv
metrics_overview_90 = pd.read_csv(t90_csv_path)

# Merge t90_ and nseq_ columns from all inf_frac_ DataFrames by matching 'model name'
metrics_overview_90 = pd.merge(metrics_overview_90, inf_frac_bld_igg[['model name', 't90_bld_igg', 'nseq_bld_igg']], on='model name', how='left')
metrics_overview_90 = pd.merge(metrics_overview_90, inf_frac_bld_spt[['model name', 't90_bld_spt', 'nseq_bld_spt']], on='model name', how='left')
metrics_overview_90 = pd.merge(metrics_overview_90, inf_frac_pln_igg[['model name', 't90_pln_igg', 'nseq_pln_igg']], on='model name', how='left')
metrics_overview_90 = pd.merge(metrics_overview_90, inf_frac_pln_spt[['model name', 't90_pln_spt', 'nseq_pln_spt']], on='model name', how='left')
metrics_overview_90 = pd.merge(metrics_overview_90, inf_frac_panc_igg[['model name', 't90_panc_igg', 'nseq_panc_igg']], on='model name', how='left')
metrics_overview_90 = pd.merge(metrics_overview_90, inf_frac_panc_spt[['model name', 't90_panc_spt', 'nseq_panc_spt']], on='model name', how='left')

# Save back to the original CSV
metrics_overview_90.to_csv(t90_csv_path, index=False)
