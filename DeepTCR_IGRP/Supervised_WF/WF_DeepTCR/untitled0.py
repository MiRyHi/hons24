# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:46:20 2024

@author: Mitch
"""

import pandas as pd



# Apply the function to each metrics_overview CSV file with appropriate prefixes
t50_csv_path = 'D:/Mitch/Documents/DeepTCR_IGRP/IGRP_add_dist_test_1_ctvt/1_log_inference_add_dist/Metrics_overview_50.csv'
t60_csv_path = 'D:/Mitch/Documents/DeepTCR_IGRP/IGRP_add_dist_test_1_ctvt/1_log_inference_add_dist/Metrics_overview_60.csv'
t70_csv_path = 'D:/Mitch/Documents/DeepTCR_IGRP/IGRP_add_dist_test_1_ctvt/1_log_inference_add_dist/Metrics_overview_70.csv'
t80_csv_path = 'D:/Mitch/Documents/DeepTCR_IGRP/IGRP_add_dist_test_1_ctvt/1_log_inference_add_dist/Metrics_overview_80.csv'
t90_csv_path = 'D:/Mitch/Documents/DeepTCR_IGRP/IGRP_add_dist_test_1_ctvt/1_log_inference_add_dist/Metrics_overview_90.csv'

def process_metrics_overview(csv_path, t_prefix):
    """
    Process the Metrics_overview CSV file by multiplying the t_prefix columns 
    with their corresponding nseq columns and appending new actual counts at the end of the file.
    """
    print(f"Processing file: {csv_path}...")

    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path}. Shape: {df.shape}")
        
        # Store new columns separately to avoid modifying DataFrame structure during the loop
        new_columns = {}
        
        # Iterate over the columns that match the threshold prefix
        for t_col in [col for col in df.columns if col.startswith(t_prefix)]:
            # Find the corresponding nseq column
            nseq_col = t_col.replace(t_prefix, 'nseq_')
            print(f"Processing t_col: {t_col}, corresponding nseq_col: {nseq_col}")

            if nseq_col in df.columns:
                # Check for missing values in the t_col and nseq_col
                print(f"Checking for missing values in {t_col} and {nseq_col}...")
                missing_t = df[t_col].isnull().sum()
                missing_nseq = df[nseq_col].isnull().sum()
                print(f"Missing values - {t_col}: {missing_t}, {nseq_col}: {missing_nseq}")

                # Perform multiplication if both columns exist and are valid
                print(f"Multiplying {t_col} and {nseq_col} to create actual_{t_col}...")
                actual_values = df[t_col] * df[nseq_col]

                # Store the result in new_columns dict to add later
                new_columns[f'actual_{t_col}'] = actual_values
            else:
                print(f"nseq column {nseq_col} not found for {t_col}. Skipping this pair.")
        
        # Add all new columns to the DataFrame at once, after the loop
        for col_name, values in new_columns.items():
            df[col_name] = values
            print(f"Added new column: {col_name}")
        
        # Save the updated DataFrame back to CSV
        print(f"Saving the updated DataFrame back to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print(f"File saved successfully: {csv_path}")

    except Exception as e:
        print(f"Error occurred while processing {csv_path}: {e}")

# Test the function on t50 first
process_metrics_overview(t50_csv_path, 't50_')
process_metrics_overview(t60_csv_path, 't60_')
process_metrics_overview(t70_csv_path, 't70_')
process_metrics_overview(t80_csv_path, 't80_')
process_metrics_overview(t90_csv_path, 't90_')

def reorder_columns_explicit(csv_path, t_prefix):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Define the desired order of columns explicitly
    desired_columns = ['model_name']  # Start with 'model_name'
    
    # Define the order of the tXX_ and corresponding actual_ and nseq_ columns
    targets = ['bld_igg', 'bld_pd1', 'bld_spt', 'pln_igg', 'pln_pd1', 'pln_spt', 'panc_igg', 'panc_pd1', 'panc_spt']
    
    # Build the desired order of columns with actual and nseq columns between tXX and nseq
    for target in targets:
        t_col = f'{t_prefix}{target}'
        actual_col = f'actual_{t_prefix}{target}'
        nseq_col = f'nseq_{target}'
        desired_columns.extend([t_col, actual_col, nseq_col])
    
    # Ensure remaining columns are at the beginning
    remaining_cols = [col for col in df.columns if col not in desired_columns]
    
    # Reorder the DataFrame
    df = df[remaining_cols + desired_columns]
    
    # Save the reordered DataFrame back to the same CSV file
    df.to_csv(csv_path, index=False)
    print(f"Reordered columns in {csv_path} and saved successfully.")

# Reorder columns for t50
reorder_columns_explicit(t50_csv_path, 't50_')

# Reorder for other files if needed
reorder_columns_explicit(t60_csv_path, 't60_')
reorder_columns_explicit(t70_csv_path, 't70_')
reorder_columns_explicit(t80_csv_path, 't80_')
reorder_columns_explicit(t90_csv_path, 't90_')