# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:02:48 2024

@author: Mitch
"""
# venv = IGRP_DTCR (py3.7.12, deepTCR2.1.0, tf2.7, CUDAtk11.2, cuDNN8.1, fastcluster1.2.6)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import tensorflow as tf
from tensorflow.keras import mixed_precision
from DeepTCR.DeepTCR import DeepTCR_WF
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
import io
import sys
import concurrent.futures
import time
from sklearn.metrics import roc_auc_score, log_loss, cohen_kappa_score
import itertools
from copy import deepcopy
import gc
import tensorflow.keras.backend as K

# Set a timeout duration (in seconds) for each training and inference process
TRAINING_TIMEOUT = 1500
INFERENCE_TIMEOUT = 600

# Output buffer for logs
output = io.StringIO()
# Global variable for storing seeds
seeds = None
# Updated base path
base_path = 'D:/Mitch/Documents/DeepTCR_IGRP/Supervised_WF'
# enable mixed precision training through GPU 
mixed_precision.set_global_policy('mixed_float16')
# Enable memory growth for GPUs to avoid memory fragmentation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print(e)
# ensure cuDNN autotune is enabled 
tf.config.optimizer.set_experimental_options({'autotune_conv': True})
# enable XLA (Accelerated Linear Algebra) compiliation
tf.config.optimizer.set_jit(True)

# Function to save a checkpoint
def save_checkpoint(checkpoint_data, checkpoint_file):
    # Ensure the directory for the checkpoint exists
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)  # Create the directory if it doesn't exist

    # Now save the checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

# Function to load a checkpoint
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

# Global checkpointing directory
checkpoint_dir = os.path.join(base_path, "checkpoints")

# Create a helper function to save checkpoints after every group of 10 models
def checkpoint_after_group(counter, cross_validation, common_settings, log_inference_dir):
    if counter > 0 and counter % 25 == 0:
        checkpoint_data = {
            'counter': counter,
            'Cross validation': cross_validation,
            'settings': common_settings
        }
        checkpoint_file = os.path.join(checkpoint_dir, f"{cross_validation}_checkpoint_group_{counter // 25}.pkl")
        save_checkpoint(checkpoint_data, checkpoint_file)
        print(f"Checkpoint saved for {cross_validation} after {counter} models.")

# ========================================================================
# Helper functions for file operations and directory setup

def calculate_prediction_percentages_thresh(TP, FP, TN, FN, inference_stats, training_object_name, inference_data_name, AUC_value_trn, AUC_value_inf, mcc, log_loss_value, kappa, threshold, log_inference_dir):
    # Ensure the directory exists
    if not os.path.exists(log_inference_dir):
        os.makedirs(log_inference_dir) 
    
    # Define the required columns, including the new metrics
    required_columns = [
        'model_name',
        'TP', 'FP', 'TN', 'FN',
        'accuracy', 'precision (PPV)',
        'recall (TPR)', 'specificity (TNR)', 'f1 score',
        'bal accuracy', 'AUC value inf', 'AUC value trn',
        'mcc', 'log loss', 'kappa',
        'fall out (FPR)', 'miss rate (FNR)', 'NPV',
        'lr plus', 'lr minus'
    ]

    # Ensure required columns exist, initializing any missing ones
    inference_stats = inference_stats.reindex(columns=required_columns, fill_value=None)

    # Calculate total samples for accuracy
    total_samples = TP + FP + TN + FN

    # Safely calculate metrics (avoiding division by zero)
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    b_accuracy = ((TP / (TP + FN)) + (TN / (TN + FP))) / 2 if total_samples > 0 else 0
    fall_out = FP / (FP + TN) if (FP + TN) > 0 else 0
    miss_rate = FN / (FN + TP) if (FN + TP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    lr_plus = recall / fall_out if fall_out > 0 else float('inf')
    lr_minus = miss_rate / specificity if specificity > 0 else float('inf')

    # Define metrics for the current model
    metrics = {
        'TP': TP, 
        'FP': FP, 
        'TN': TN, 
        'FN': FN,
        'accuracy': accuracy,
        'precision (PPV)': precision,
        'recall (TPR)': recall,
        'specificity (TNR)': specificity,
        'f1 score': f1_score,
        'bal accuracy': b_accuracy,
        'AUC value inf': AUC_value_inf,
        'AUC value trn': AUC_value_trn,
        'mcc': mcc, 
        'log loss': log_loss_value, 
        'kappa': kappa,
        'fall out (FPR)': fall_out,
        'miss rate (FNR)': miss_rate,
        'NPV': NPV,
        'lr plus': lr_plus,
        'lr minus': lr_minus
    }

    # Update or insert the metrics for the model
    if training_object_name not in inference_stats['model_name'].values:
        # Append new row if model does not exist
        metrics['model_name'] = training_object_name
        inference_stats = inference_stats.append(metrics, ignore_index=True)
    else:
        # Update existing row for the model
        inference_stats.loc[inference_stats['model_name'] == training_object_name, metrics.keys()] = metrics.values()

    # Save updated inference_stats
    inference_stats.to_csv(os.path.join(log_inference_dir, f'Inference_stats_{threshold}.csv'), index=False)

def calculate_prediction_stats_thresh(df_pred, input_df, training_object_name, inference_data_name, threshold, true_labels, predicted_probs, AUC_value_inf):
    # Apply threshold to make predictions
    df_pred['Pred'] = np.where(df_pred['TETpos_prob'] > threshold / 100.0, 'TETpos', 'TETneg')
    
    # Merge with true labels from input_df
    combined_df = pd.concat([df_pred, input_df[['class']]], axis=1)

    # Confusion matrix components
    TP = ((combined_df['Pred'] == 'TETpos') & (combined_df['class'] == 'TETpos')).sum()
    FP = ((combined_df['Pred'] == 'TETpos') & (combined_df['class'] == 'TETneg')).sum()
    TN = ((combined_df['Pred'] == 'TETneg') & (combined_df['class'] == 'TETneg')).sum()
    FN = ((combined_df['Pred'] == 'TETneg') & (combined_df['class'] == 'TETpos')).sum()

    # MCC calculation
    mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0

    # Log Loss calculation
    try:
        log_loss_value = log_loss(true_labels, predicted_probs)
    except ValueError:
        log_loss_value = None  # Handle cases where log loss is undefined

    # Kappa Score calculation
    pred_labels_numeric = combined_df['Pred'].map({'TETpos': 1, 'TETneg': 0}).values
    kappa = cohen_kappa_score(true_labels, pred_labels_numeric)

    # Additional calculations
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    fall_out = FP / (FP + TN) if (FP + TN) > 0 else 0
    miss_rate = FN / (FN + TP) if (FN + TP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    lr_plus = recall / fall_out if fall_out > 0 else float('inf')
    lr_minus = miss_rate / specificity if specificity > 0 else float('inf')
    
    # Return calculated metrics
    return TP, FP, TN, FN, accuracy, precision, recall, specificity, f1_score, mcc, log_loss_value, kappa, fall_out, miss_rate, NPV, lr_plus, lr_minus

def create_inference_ranks(log_inference_dir, thresholds, inference_data_name):
    """
    Create inference ranks by reading inference stats, ranking them, and saving to CSV.
    """
    # List of columns to rank
    rank_columns_base = ['TP', 'FP', 'TN', 'FN', 'accuracy', 'precision (PPV)', 'recall (TPR)', 'specificity (TNR)', 
                         'f1 score', 'bal accuracy', 'AUC value inf', 'AUC value trn', 'mcc', 'log loss', 'kappa', 
                         'fall out (FPR)', 'miss rate (FNR)', 'NPV', 'lr plus', 'lr minus']
    
    for thresh in thresholds:
        # Load the corresponding inference_stats file for the threshold
        inference_stats_file = os.path.join(log_inference_dir, f'Inference_stats_{thresh}.csv').replace("\\", "/")
        
        if os.path.exists(inference_stats_file):
            inference_stats = pd.read_csv(inference_stats_file)

            # Dynamically adjust column names to include the inference_data_name in brackets
            rank_columns = [f"{col} ({inference_data_name})" for col in rank_columns_base]

            # Ensure that the required columns exist in the dataframe before ranking
            if all(col in inference_stats.columns for col in rank_columns):
                print(f"Ranking columns for threshold {thresh}")

                # Create ranks for each of the relevant columns
                rank_df = inference_stats[['model_name']].copy()  # Start with model_name
                for col in rank_columns:
                    rank_df[f'{col}_rk'] = inference_stats[col].rank(ascending=False, method='min')  # Rank higher values better

                # Save the ranked DataFrame to a new CSV file
                inference_ranks_file = os.path.join(log_inference_dir, f'Inference_ranks_{thresh}.csv').replace("\\", "/")
                rank_df.to_csv(inference_ranks_file, index=False)
            else:
                missing_cols = [col for col in rank_columns if col not in inference_stats.columns]
                print(f"Required columns are missing in Inference Stats for threshold {thresh}: {missing_cols}")
        else:
            print(f"Warning: Inference stats file for threshold {thresh} not found: {inference_stats_file}")

def create_inference_stats_50(log_inference_dir):
    global Inference_stats_50
    inference_stats_file_50 = os.path.join(log_inference_dir, 'Inference_stats_50.csv').replace("\\", "/")

    if os.path.exists(inference_stats_file_50):
        Inference_stats_50 = pd.read_csv(inference_stats_file_50)
    else:
        Inference_stats_50 = pd.DataFrame(columns=['model_name'])
        Inference_stats_50.to_csv(inference_stats_file_50, index=False)  # Save after initialization

def create_inference_stats_60(log_inference_dir):
    global Inference_stats_60
    inference_stats_file_60 = os.path.join(log_inference_dir, 'Inference_stats_60.csv').replace("\\", "/")
    
    if os.path.exists(inference_stats_file_60):
        Inference_stats_60 = pd.read_csv(inference_stats_file_60)
    else:
        Inference_stats_60 = pd.DataFrame(columns=['model_name'])
        Inference_stats_60.to_csv(inference_stats_file_60, index=False)  # Save after initialization

def create_inference_stats_70(log_inference_dir):
    global Inference_stats_70
    inference_stats_file_70 = os.path.join(log_inference_dir, 'Inference_stats_70.csv').replace("\\", "/")
    
    if os.path.exists(inference_stats_file_70):
        Inference_stats_70 = pd.read_csv(inference_stats_file_70)
    else:
        Inference_stats_70 = pd.DataFrame(columns=['model_name'])
        Inference_stats_70.to_csv(inference_stats_file_70, index=False)  # Save after initialization

def create_inference_stats_80(log_inference_dir):
    global Inference_stats_80
    inference_stats_file_80 = os.path.join(log_inference_dir, 'Inference_stats_80.csv').replace("\\", "/")
    
    if os.path.exists(inference_stats_file_80):
        Inference_stats_80 = pd.read_csv(inference_stats_file_80)
    else:
        Inference_stats_80 = pd.DataFrame(columns=['model_name'])
        Inference_stats_80.to_csv(inference_stats_file_80, index=False)  # Save after initialization

def create_inference_stats_90(log_inference_dir):
    global Inference_stats_90
    inference_stats_file_90 = os.path.join(log_inference_dir, 'Inference_stats_90.csv').replace("\\", "/")
    
    if os.path.exists(inference_stats_file_90):
        Inference_stats_90 = pd.read_csv(inference_stats_file_90)
    else:
        Inference_stats_90 = pd.DataFrame(columns=['model_name'])
        Inference_stats_90.to_csv(inference_stats_file_90, index=False)  # Save after initialization

def create_log_inference_directory(training_data_name):
    log_inference_dir = os.path.join(base_path, f'1_log_inference_{training_data_name}/').replace("\\", "/")
    if not os.path.exists(log_inference_dir):
        os.makedirs(log_inference_dir)
        print(f"Directory created: {log_inference_dir}")
    return log_inference_dir

def create_metrics_overview(threshold, log_inference_dir):
    training_log_path = os.path.join(log_inference_dir, 'Training_Log.csv').replace("\\", "/")
    inference_stats_path = os.path.join(log_inference_dir, f'Inference_stats_{threshold}.csv').replace("\\", "/")
    inference_ranks_path = os.path.join(log_inference_dir, f'Inference_ranks_{threshold}.csv').replace("\\", "/")
    
    # Check if the files exist
    if not os.path.exists(training_log_path) or not os.path.exists(inference_stats_path) or not os.path.exists(inference_ranks_path):
        print(f"Error: Required files for threshold {threshold} do not exist.")
        return
    
    training_log = pd.read_csv(training_log_path)
    inference_stats = pd.read_csv(inference_stats_path)
    inference_ranks = pd.read_csv(inference_ranks_path)

    # Merge on model_name
    merged_df = pd.merge(training_log, inference_ranks, on='model_name', how='inner')
    merged_df = pd.merge(merged_df, inference_stats, on='model_name', how='inner')
    
    # Save merged metrics overview
    merged_file_path = os.path.join(log_inference_dir, f't{threshold}_{training_data_name}.csv').replace("\\", "/")
    merged_df.to_csv(merged_file_path, index=False)

def create_training_folders(start_number, model_type, training_data_name):
    folder_name = f'IGRP_WF_{start_number}_{model_type}_{training_data_name}'
    folder_path = os.path.join(base_path, folder_name).replace("\\", "/")
    safe_create_directory(folder_path)
    return folder_name, folder_path  # Return both the folder name and path

def initiate_training(training_object_name, settings, log_inference_dir):
    global DTCR_WF
    AUC_value_trn = None
    cross_validation = None

    # Extract the model type and the dictionary of common settings from the tuple
    model_type, common_settings = settings

    # Extract common settings
    hinge_loss_t = common_settings.get('hinge_loss_t', 0.0)
    qualitative_agg = common_settings.get('qualitative_agg', True)
    quantitative_agg = common_settings.get('quantitative_agg', False)
    units_agg = common_settings.get('units_agg', 12)
    units_fc = common_settings.get('units_fc', 12)
    epochs_min = common_settings.get('epochs_min', 25)
    combine_train_valid = common_settings.get('combine_train_valid', False)
    drop_out_rate = common_settings.get('drop_out_rate', 0.0)
    subsample_valid_test = common_settings.get('subsample_valid_test', False)
    accuracy_min = common_settings.get('accuracy_min', None)
    subsample = common_settings.get('subsample', None)
    batch_size = common_settings.get('batch_size', 25)

    # Handle K-fold training
    if 'Kf' in training_object_name:
        print(f"Training model: {training_object_name}")
        cross_validation = 'K-fold'
        suppress_output = False
        folds = common_settings.get('folds', kf_fold_values)  # Use kf_fold_values for K-fold
        weight_by_class = common_settings['weight_by_class']
        kernel = common_settings['kernel']
        num_concepts = common_settings['num_concepts']
        size_of_net = common_settings['size_of_net']
        num_fc_layers = common_settings['num_fc_layers']
        num_agg_layers = common_settings['num_agg_layers']
        DTCR_WF.K_Fold_CrossVal(
            folds=folds,
            weight_by_class=weight_by_class,
            kernel=kernel,
            num_concepts=num_concepts,
            size_of_net=size_of_net,
            combine_train_valid=combine_train_valid,
            epochs_min=epochs_min,
            hinge_loss_t=hinge_loss_t,
            qualitative_agg=qualitative_agg,
            quantitative_agg=quantitative_agg,
            num_fc_layers=num_fc_layers,
            units_fc=units_fc,
            num_agg_layers=num_agg_layers,
            units_agg=units_agg,
            drop_out_rate=drop_out_rate,
            accuracy_min=accuracy_min,
            subsample=subsample,
            subsample_valid_test=subsample_valid_test,
            batch_size=batch_size,
            suppress_output=suppress_output
        )
        # AUC curve and logging after K-fold training
        DTCR_WF.AUC_Curve()
        AUC_value_trn = DTCR_WF.AUC_DF['AUC'].mean()

        # Log training details for K-fold model
        log_training_details(training_object_name, cross_validation, log_inference_dir, **common_settings)

    # Handle Monte Carlo training
    elif 'Mc' in training_object_name:
        print(f"Training model: {training_object_name}")
        cross_validation = 'Monte Carlo'
        suppress_output = False
        weight_by_class = common_settings['weight_by_class']
        test_size = common_settings['test_size']
        LOO = common_settings['LOO']
        folds = common_settings.get('folds', mc_fold_values)  # Use mc_fold_values for Monte Carlo
        kernel = common_settings['kernel']
        num_concepts = common_settings['num_concepts']
        size_of_net = common_settings['size_of_net']
        num_fc_layers = common_settings['num_fc_layers']
        num_agg_layers = common_settings['num_agg_layers']
        DTCR_WF.Monte_Carlo_CrossVal(
            LOO=LOO,
            folds=folds,
            weight_by_class=weight_by_class,
            test_size=test_size,
            kernel=kernel,
            num_concepts=num_concepts,
            size_of_net=size_of_net,
            combine_train_valid=combine_train_valid,
            epochs_min=epochs_min,
            hinge_loss_t=hinge_loss_t,
            qualitative_agg=qualitative_agg,
            quantitative_agg=quantitative_agg,
            num_fc_layers=num_fc_layers,
            units_fc=units_fc,
            num_agg_layers=num_agg_layers,
            units_agg=units_agg,
            drop_out_rate=drop_out_rate,
            accuracy_min=accuracy_min,
            subsample=subsample,
            subsample_valid_test=subsample_valid_test,
            batch_size=batch_size,
            suppress_output=suppress_output
        )
        # AUC curve and logging after Monte Carlo cross-validation
        DTCR_WF.AUC_Curve()
        AUC_value_trn = DTCR_WF.AUC_DF['AUC'].mean()

        # Log training details for Monte Carlo model
        log_training_details(training_object_name, cross_validation, log_inference_dir, **common_settings)

    return AUC_value_trn

def initiate_training_with_timeout(training_object_name, settings, log_inference_dir):
    """Wrap initiate_training to enforce a timeout."""
    # Use run_with_timeout to ensure timeout is applied to initiate_training
    result = run_with_timeout(initiate_training, training_object_name, settings, log_inference_dir, timeout=TRAINING_TIMEOUT)
    
    # Check if training timed out
    if result is None:
        print(f"Training for {training_object_name} was aborted due to timeout.")
    return result

def input_inference_data(inference_data_name):
    inference_data_directory = f'{base_path}/WF_DeepTCR/Data_valid/{inference_data_name}'
    DTCR_WF.Get_Data(directory=inference_data_directory,
                     Load_Prev_Data=False,
                     aggregate_by_aa=False,
                     sep=',',
                     aa_column_beta=6,
                     v_beta_column=7,
                     d_beta_column=8,
                     j_beta_column=9)
    print("Inference data input successful")

def input_training_name(training_object_name):
    global DTCR_WF
    DTCR_WF = DeepTCR_WF(training_object_name)

def import_validation_dataset(inference_data_directory):
    # Look for CSV files in the specified directory
    csv_files = [f for f in os.listdir(inference_data_directory) if f.endswith('.csv')]
    
    # Ensure that only one CSV file is present in the directory
    if len(csv_files) == 1:
        csv_file_path = os.path.join(inference_data_directory, csv_files[0])
        input_df = pd.read_csv(csv_file_path)
    else:
        print("Error: No CSV file or multiple CSV files found in the directory.")
        sys.exit(1)
    
    # Print a message confirming that the data has been loaded
    print("Validation data loaded")
    
    return input_df

def log_training_details(training_object_name, cross_validation, log_inference_dir, **kwargs):
    # Determine the correct values for LOO and Folds based on training type
    if cross_validation == 'K-fold':
        loo_value = 'N.A.'  # LOO is N.A. for K-fold
        test_size_value = 'N.A.'
        folds_value = kwargs.get('folds', 'N.A.')  # Use the actual fold value for K-fold models
    elif cross_validation == 'Monte Carlo':
        loo_value = kwargs.get('LOO', 'None')  # For Monte Carlo, use the actual LOO value
        folds_value = kwargs.get('folds', 'N.A.')  # Use the actual fold value for Monte Carlo models
        test_size_value = kwargs.get('test_size', 'N.A.')

    # Prepare the training details dictionary, ensuring only specific values are logged
    training_details = {
        'model_name': training_object_name,
        'Cross validation': cross_validation,
        'folds': folds_value,  # Correctly logging the fold value for each model
        'weight_by_class': kwargs.get('weight_by_class', 'N.A.'),  # Specific weight_by_class value used
        'test_size': test_size_value,
        'LOO': loo_value,
        'kernel': kwargs.get('kernel', 'N.A.'),  # Specific kernel value used
        'size_of_net': kwargs.get('size_of_net', 'N.A.'),  # Specific size_of_net value used
        'num_concepts': kwargs.get('num_concepts', 'N.A.'),  # Specific num_concepts value used
        'num_fc_layers': kwargs.get('num_fc_layers', 'N.A.'),  # Specific num_fc_layers value used
        'units_fc': kwargs.get('units_fc', 'N.A.'),  # Specific units_fc value used
        'num_agg_layers': kwargs.get('num_agg_layers', 'N.A.'),  # Specific num_agg_layers value used
        'units_agg': kwargs.get('units_agg', 'N.A.'),  # Specific units_agg value used
        'epochs_min': kwargs.get('epochs_min', 'N.A.'),  # Specific epochs_min value used
        'combine_train_valid': kwargs.get('combine_train_valid', 'N.A.'),  # Specific combine_train_valid
        'train_loss_min': kwargs.get('train_loss_min', 'N.A.'),  # Specific train_loss_min value used
        'hinge_loss_t': kwargs.get('hinge_loss_t', 'N.A.'),  # Specific hinge_loss_t value
        'accuracy_min': str(kwargs.get('accuracy_min', 'None')) if kwargs.get('accuracy_min') is None else kwargs.get('accuracy_min'),
        'qualitative_agg': kwargs.get('qualitative_agg', 'N.A.'),  # Specific value used
        'quantitative_agg': kwargs.get('quantitative_agg', 'N.A.'),  # Specific value used   
        'drop_out_rate': kwargs.get('drop_out_rate', 'N.A.'),  # Specific drop_out_rate value used
        'multisample_dropout': kwargs.get('multisample_dropout', 'N.A.'),  # Specific value used
        'multisample_dropout_rate': kwargs.get('multisample_dropout_rate', 'N.A.'),  # Specific value used
        'subsample': str(kwargs.get('subsample', 'None')) if kwargs.get('subsample', 'None') is None else kwargs.get('subsample'),
        'subsample_valid_test': kwargs.get('subsample_valid_test', 'N.A.'),  # Specific value used
        'l2_reg': kwargs.get('l2_reg', 'N.A.')  # Specific l2_reg value used
    }

    # Append or create the CSV file
    log_file = os.path.join(log_inference_dir, 'Training_Log.csv').replace("\\", "/")
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = log_df.append(training_details, ignore_index=True)
    else:
        log_df = pd.DataFrame([training_details])

    # Save the updated DataFrame
    log_df.to_csv(log_file, index=False)
    print("Training details logged")

def modify_df(sample_list, inference_pred):
    # Use DTCR_WF's internal attributes for sample list and prediction probabilities
    df_samplelist = pd.DataFrame(DTCR_WF.Inference_Sample_List)
    df_infpred = pd.DataFrame(DTCR_WF.Inference_Pred)
    return df_samplelist, df_infpred

def modify_df_1(df_samplelist, df_infpred):
    # Concatenate sample list and predictions, with appropriate column names
    list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
    list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
    list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
    return list_pred

def modify_df_2(list_pred):
    # Sort and reset index for proper alignment
    sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
    sorted_pred = sorted_pred.reset_index(drop=True)
    return sorted_pred

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

def AUC_sample_inference(input_df, df_pred):
    # Extract true labels and predicted probabilities
    true_labels = input_df['class'].map({'TETpos': 1, 'TETneg': 0}).values
    predicted_probs = df_pred['TETpos_prob']

    # Calculate AUC for inference
    AUC_value_inf = roc_auc_score(true_labels, predicted_probs)

    return true_labels, predicted_probs, AUC_value_inf

def run_sample_inference_with_timeout(beta_sequences, v_beta, d_beta, j_beta):
    """Wrap run_sample_inference to enforce a timeout."""
    result = run_with_timeout(run_sample_inference, beta_sequences, v_beta, d_beta, j_beta, timeout=INFERENCE_TIMEOUT)
    
    # Check if inference timed out
    if result is None:
        print("Sample inference was aborted due to timeout.")
    return result

def run_with_timeout(func, *args, timeout=TRAINING_TIMEOUT):
    """Run the given function with arguments and a specified timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print(f"Operation timed out after {timeout} seconds.")
            future.cancel()
            return None

def safe_create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")

def set_data_train_directory():
    global input_data_folder_name, directory_path, training_object_name
    input_data_folder_name = training_object_name.split('_', maxsplit=4)[-1]
    directory_path = os.path.join(base_path, f'WF_DeepTCR/Data_train/IGRP_{input_data_folder_name}').replace("\\", "/")

def set_inference_name(training_object_name):
    global DTCR_WF
    DTCR_WF = DeepTCR_WF(training_object_name)

def set_input_train_data():
    global DTCR_WF, directory_path
    DTCR_WF.Get_Data(directory=directory_path,
                     Load_Prev_Data=False,
                     aggregate_by_aa=True,
                     sep=',',
                     type_of_data_cut='Read_Cut',
                     data_cut=10,
                     aa_column_beta=6,
                     count_column=3,
                     v_beta_column=7,
                     d_beta_column=8,
                     j_beta_column=9)
    print("Training data input successful")

def set_sample_inference():
    beta_sequences = DTCR_WF.beta_sequences
    v_beta = DTCR_WF.v_beta
    d_beta = DTCR_WF.d_beta
    j_beta = DTCR_WF.j_beta
    return beta_sequences, v_beta, d_beta, j_beta

def update_inference_stats(Inference_stats, training_object_name, inference_data_name, TP, FP, TN, FN, TPR, FPR, TNR, FNR, PPV, NPV, b_accuracy, F1, accuracy, AUC_value_trn, AUC_value_inf, mcc, log_loss_value, kappa, fall_out, miss_rate, lr_plus, lr_minus):
    # Remove duplicated columns
    Inference_stats = Inference_stats.loc[:, ~Inference_stats.columns.duplicated()]

    # Create dynamic column names
    col_TP = 'TP'
    col_FP = 'FP'
    col_TN = 'TN'
    col_FN = 'FN'
    col_accuracy = 'accuracy'
    col_b_accuracy = 'b_accuracy'
    col_PPV = 'PPV'
    col_NPV = 'NPV'
    col_TPR = 'TPR'
    col_TNR = 'TNR'
    col_F1 = 'F1'
    col_ROC_trn = 'ROC Trn'
    col_ROC_inf = 'ROC Inf'
    col_FPR = 'FPR'
    col_FNR = 'FNR'
    col_MCC = 'MCC'
    col_Log_Loss = 'Log Loss'
    col_Kappa = 'Kappa'
    col_fall_out = 'fall_out'
    col_miss_rate = 'miss_rate'
    col_lr_plus = 'lr_plus'
    col_lr_minus = 'lr_minus'

    # Ensure that all columns are created, even if the dataframe is empty or missing columns
    for col in [col_TP, col_FP, col_TN, col_FN, col_accuracy, col_b_accuracy, col_PPV, col_NPV, col_TPR, col_TNR, col_F1, col_ROC_trn, col_ROC_inf, col_FPR, col_FNR, col_MCC, col_Log_Loss, col_Kappa, col_fall_out, col_miss_rate, col_lr_plus, col_lr_minus]:
        if col not in Inference_stats.columns:
            Inference_stats[col] = np.nan  # Initialize missing columns

    # Check if 'model_name' exists in the dataframe and if the model already exists in the stats
    if 'model_name' in Inference_stats.columns:
        matching_indices = Inference_stats.index[Inference_stats['model_name'] == training_object_name]

        if len(matching_indices) > 0:
            index = matching_indices[0]

            # Update the values for the current model
            Inference_stats.at[index, col_TP] = TP
            Inference_stats.at[index, col_FP] = FP
            Inference_stats.at[index, col_TN] = TN
            Inference_stats.at[index, col_FN] = FN
            Inference_stats.at[index, col_accuracy] = accuracy
            Inference_stats.at[index, col_b_accuracy] = b_accuracy
            Inference_stats.at[index, col_PPV] = PPV
            Inference_stats.at[index, col_NPV] = NPV
            Inference_stats.at[index, col_TPR] = TPR
            Inference_stats.at[index, col_TNR] = TNR
            Inference_stats.at[index, col_F1] = F1
            Inference_stats.at[index, col_ROC_trn] = AUC_value_trn
            Inference_stats.at[index, col_ROC_inf] = AUC_value_inf
            Inference_stats.at[index, col_FPR] = FPR
            Inference_stats.at[index, col_FNR] = FNR
            Inference_stats.at[index, col_MCC] = mcc
            Inference_stats.at[index, col_Log_Loss] = log_loss_value
            Inference_stats.at[index, col_Kappa] = kappa
            Inference_stats.at[index, col_fall_out] = fall_out
            Inference_stats.at[index, col_miss_rate] = miss_rate
            Inference_stats.at[index, col_lr_plus] = lr_plus
            Inference_stats.at[index, col_lr_minus] = lr_minus

        else:
            # If no matching model_name exists, add a new row with the model's stats
            new_row = {
                'model_name': training_object_name,
                col_TP: TP,
                col_FP: FP,
                col_TN: TN,
                col_FN: FN,
                col_accuracy: accuracy,
                col_b_accuracy: b_accuracy,
                col_PPV: PPV,
                col_NPV: NPV,
                col_TPR: TPR,
                col_TNR: TNR,
                col_F1: F1,
                col_ROC_trn: AUC_value_trn,
                col_ROC_inf: AUC_value_inf,
                col_FPR: FPR,
                col_FNR: FNR,
                col_MCC: mcc,
                col_Log_Loss: log_loss_value,
                col_Kappa: kappa,
                col_fall_out: fall_out,
                col_miss_rate: miss_rate,
                col_lr_plus: lr_plus,
                col_lr_minus: lr_minus
            }
            Inference_stats = Inference_stats.append(new_row, ignore_index=True)

    else:
        # If 'model_name' column doesn't exist, create a new dataframe
        new_row = {
            'model_name': training_object_name,
            col_TP: TP,
            col_FP: FP,
            col_TN: TN,
            col_FN: FN,
            col_accuracy: accuracy,
            col_b_accuracy: b_accuracy,
            col_PPV: PPV,
            col_NPV: NPV,
            col_TPR: TPR,
            col_TNR: TNR,
            col_F1: F1,
            col_ROC_trn: AUC_value_trn,
            col_ROC_inf: AUC_value_inf,
            col_FPR: FPR,
            col_FNR: FNR,
            col_MCC: mcc,
            col_Log_Loss: log_loss_value,
            col_Kappa: kappa,
            col_fall_out: fall_out,
            col_miss_rate: miss_rate,
            col_lr_plus: lr_plus,
            col_lr_minus: lr_minus
        }
        Inference_stats = pd.DataFrame([new_row])

    # Remove any duplicate columns again
    Inference_stats = Inference_stats.loc[:, ~Inference_stats.columns.duplicated()]

    # Re-order the columns, placing MCC, Log Loss, and Kappa after the ROC columns
    ordered_columns = ['model_name', col_TP, col_FN, col_TN, col_FP, col_accuracy, col_b_accuracy, col_PPV, col_NPV, col_TPR, col_TNR, col_F1, col_ROC_trn, col_ROC_inf, col_MCC, col_Log_Loss, col_Kappa, col_FPR, col_FNR, col_fall_out, col_miss_rate, col_lr_plus, col_lr_minus]
    Inference_stats = Inference_stats[[col for col in ordered_columns if col in Inference_stats.columns]]

    return Inference_stats

# ================= Training Settings and Parameters ======================

# Base common settings (non-varying settings)
base_common_settings = {
    'batch_size': 25,
    'epochs_min': 25,
    'combine_train_valid': False,
    'hinge_loss_t': 0.1,
    'accuracy_min': None,
    'qualitative_agg': True,
    'quantitative_agg': False,
    'units_agg': 12,
    'units_fc': 12,
    'drop_out_rate': 0.1,
    'multisample_dropout': True,
    'multisample_dropout_rate': 0.1,
    'subsample': 500,
    'subsample_valid_test': False,
    'l2_reg': 0.01
}

# Variable settings (the ones that will create different setting combinations)
test_size_values = [0.2]
LOO_values = [2]
size_of_net_values = ['small', 'medium']
kernel_values = [3, 5]
num_concepts_values = [24, 36]
num_agg_layers_values = [0]
num_fc_layers_values = [0, 1]
kf_fold_values = [5]
mc_fold_values = [20]
weight_by_class_values = [True, False]

# Initialize an empty list to hold all setting combinations
common_settings_variations = []

# Generate K-fold settings (using kf_fold_values, no LOO)
for size_of_net, kernel, num_concepts, kf_folds, num_fc_layers, num_agg_layers, weight_by_class in itertools.product(
    size_of_net_values, kernel_values, num_concepts_values, kf_fold_values, num_fc_layers_values, num_agg_layers_values, weight_by_class_values):
    
    kf_settings = deepcopy(base_common_settings)
    kf_settings['folds'] = kf_folds  # Using kf_fold_values for K-fold
    kf_settings['size_of_net'] = size_of_net
    kf_settings['kernel'] = kernel
    kf_settings['num_concepts'] = num_concepts
    kf_settings['num_fc_layers'] = num_fc_layers
    kf_settings['num_agg_layers'] = num_agg_layers
    kf_settings['weight_by_class'] = weight_by_class
    common_settings_variations.append(('Kf', kf_settings))

# Generate Monte Carlo settings (using both mc_fold_values and LOO_values)
for size_of_net, kernel, num_concepts, mc_folds, LOO, test_size, num_fc_layers, num_agg_layers, weight_by_class in itertools.product(
    size_of_net_values, kernel_values, num_concepts_values, mc_fold_values, LOO_values, test_size_values, num_fc_layers_values, num_agg_layers_values, weight_by_class_values):
    
    mc_settings = deepcopy(base_common_settings)
    mc_settings['test_size'] = test_size
    mc_settings['LOO'] = LOO
    mc_settings['folds'] = mc_folds  # Using mc_fold_values for Monte Carlo
    mc_settings['size_of_net'] = size_of_net
    mc_settings['kernel'] = kernel
    mc_settings['num_concepts'] = num_concepts
    mc_settings['num_fc_layers'] = num_fc_layers
    mc_settings['num_agg_layers'] = num_agg_layers
    mc_settings['weight_by_class'] = weight_by_class
    common_settings_variations.append(('Mc', mc_settings))

# Main Training and Inference Loop settings ==== INPUT TRAINING FOLDER START NUMBER ===========================
training_setting_variations = len(common_settings_variations)
training_folder_start_number = 1
triplicate_training = ['Kf', 'Mc']
training_data_name = 'dist'
inference_data_name = 'Kas_Gearty_no_cut'

log_inference_dir = create_log_inference_directory(training_data_name)

# Iterate over all common settings variations
kf_combinations = []
mc_combinations = []

# Reset counts
kf_count, mc_count = 0, 0

# Iterate over all common settings variations
for settings in common_settings_variations:
    model_type, common_settings = settings

    # K-fold Models - only use folds, no LOO
    if model_type == 'Kf':
        if 'LOO' in common_settings or 'test_size' in common_settings:
            print(f"Skipping K-fold model due to 'LOO' or 'test_size': {settings}")
            continue  # Skip any K-fold models that have 'LOO' or 'test_size' set
        if 'folds' not in common_settings:
            print(f"Skipping K-fold model due to missing 'folds': {settings}")
            continue  # Skip K-fold models without 'folds'
        kf_combinations.append(settings)
        kf_count += 1

    # Monte Carlo Models - use both LOO and folds
    elif model_type == 'Mc':
        if 'LOO' not in common_settings or 'folds' not in common_settings or 'test_size' not in common_settings:
            print(f"Skipping Monte Carlo model due to missing 'LOO' or 'folds' or 'test_size': {settings}")
            continue  # Skip Monte Carlo models without either LOO or folds or test_size
        mc_combinations.append(settings)
        mc_count += 1

# Sort each group logically, accessing the settings dictionary inside the tuple
kf_combinations.sort(key=lambda x: (
    x[1]['weight_by_class'],
    x[1]['kernel'],
    x[1]['size_of_net'],
    x[1]['num_concepts'],
    x[1]['folds'],  # Folds is always present in Kf settings
    x[1]['num_fc_layers'],
    x[1]['num_agg_layers']
))

mc_combinations.sort(key=lambda x: (
    x[1]['weight_by_class'],
    x[1]['test_size'],
    x[1]['LOO'],    # LOO is always present in Mc settings
    x[1]['kernel'],
    x[1]['size_of_net'],
    x[1]['num_concepts'],
    x[1]['folds'],  # Folds is always present in Mc settings
    x[1]['num_fc_layers'],
    x[1]['num_agg_layers']
))

# Calculate the total models for each type
total_kf_models = len(kf_combinations)
total_mc_models = len(mc_combinations)

# Display the total number of models for each type and ask for confirmation
print(f"Total K-fold Models: {total_kf_models}")
print(f"Total Monte Carlo Models: {total_mc_models}\n")

# Ask for confirmation to start the training process
start_training = input("Do you want to start the training process? (Enter 'True' to proceed, 'False' to exit): ").lower()

if start_training == 'true':
    print("\nStarting the training process...\n")

    # Initialize progress bar counters for each type
    kf_counter = 0
    mc_counter = 0
    
    # Main Training Loop - Adjusting based on the model type
    for model_type in ['Kf', 'Mc']:
        # Get the appropriate combination list for each model type
        if model_type == 'Kf':
            combinations = kf_combinations
        elif model_type == 'Mc':
            combinations = mc_combinations
    
        # Initialize progress bar for each type of model
        pbar = tqdm(total=len(combinations), desc=f"Processing {model_type} Models", unit="model")
    
        for variation_num, settings_tuple in enumerate(combinations, start=1):
            model_type, common_settings = settings_tuple  # Unpack the tuple properly
        
            if model_type == 'Kf':
                # Ensure LOO and test_size are not applied to K-fold
                common_settings = deepcopy(common_settings)  # Make sure to deep copy before modifying
                common_settings.pop('LOO', None)
                common_settings.pop('test_size', None)
        
            # For Monte Carlo (Mc), LOO, folds, and test_size are all necessary, so no need for changes
        
            # Create a unique folder for each model
            folder_name, folder_path = create_training_folders(training_folder_start_number, model_type, training_data_name)
            training_object_name = folder_name
            print(f"Starting training for {training_object_name} ({model_type} Model)")
        
            # Set the directory and load the data
            set_data_train_directory()
            input_training_name(training_object_name)
            set_input_train_data()
        
            # Pass the full tuple (model_type, common_settings)
            AUC_value_trn = initiate_training_with_timeout(training_object_name, (model_type, common_settings), log_inference_dir)
        
            if AUC_value_trn is None:
                print(f"Training for {training_object_name} was aborted due to timeout.")
                continue  # Skip to the next folder if training timed out
    
            # Inference steps with timeout
            set_inference_name(training_object_name)
            input_inference_data(inference_data_name)
            beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
            
            # Run sample inference (without AUC calculation here)
            run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
          
            # Modify dataframes for results
            df_samplelist, df_infpred = modify_df(DTCR_WF.Inference_Sample_List, DTCR_WF.Inference_Pred)
            df_pred = modify_df_2(modify_df_1(df_samplelist, df_infpred))
            input_df = import_validation_dataset(os.path.join(base_path, f'WF_DeepTCR/Data_valid/{inference_data_name}')).replace("\\", "/")
            true_labels, predicted_probs, AUC_value_inf = AUC_sample_inference(input_df, df_pred)
            thresholds = [50, 60, 70, 80, 90]
            
            for thresh in thresholds:
                # Call the appropriate function to load or create the inference stats for each threshold
                if thresh == 50:
                    create_inference_stats_50(log_inference_dir)
                elif thresh == 60:
                    create_inference_stats_60(log_inference_dir)
                elif thresh == 70:
                    create_inference_stats_70(log_inference_dir)
                elif thresh == 80:
                    create_inference_stats_80(log_inference_dir)
                elif thresh == 90:
                    create_inference_stats_90(log_inference_dir)
            
                # Access the corresponding inference stats for this threshold
                inference_stats = globals()[f'Inference_stats_{thresh}']
            
                # At this point, df_pred['Pred'] already contains predictions based on prior thresholding.
                # You can move on directly to calculating the stats without re-applying the threshold.
            
                # Calculate the prediction stats (metrics) for the current threshold
                TP, FP, TN, FN, accuracy, precision, recall, specificity, f1_score, mcc, log_loss_value, kappa, fpr, fnr, npv, lr_plus, lr_minus = calculate_prediction_stats_thresh(
                    df_pred, input_df, training_object_name, inference_data_name, thresh, true_labels, predicted_probs, AUC_value_inf
                )

                # Update inference stats for the current threshold
                calculate_prediction_percentages_thresh(
                    TP, FP, TN, FN, inference_stats, training_object_name, inference_data_name, 
                    AUC_value_trn, AUC_value_inf, mcc, log_loss_value, kappa, thresh, log_inference_dir
                )
            
    
            # Create ranks and generate metrics overview after processing all thresholds
            create_inference_ranks(log_inference_dir, thresholds, inference_data_name)
            
            for thresh in thresholds:
                create_metrics_overview(thresh, log_inference_dir)        
        
            # Increment the global folder number for the next model
            training_folder_start_number += 1
        
            # Update progress bar
            pbar.update(1)
            
            
            K.clear_session()  # This clears the TensorFlow session
            gc.collect()  # This forces garbage collection to free memory
            
            # Save a checkpoint after every 25 models
            checkpoint_after_group(variation_num, model_type, common_settings, log_inference_dir)

        # Always save a final checkpoint after processing all models
        if len(combinations) % 25 != 0:  # If remaining models after the last group
            checkpoint_after_group(len(combinations), model_type, common_settings, log_inference_dir)

        pbar.close()

else:
    print("Training process has been cancelled.")