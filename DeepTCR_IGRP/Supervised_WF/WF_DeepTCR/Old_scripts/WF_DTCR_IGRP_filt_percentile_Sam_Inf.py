# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:10:06 2024

@author: Mitch
"""

# venv = IGRP_DTCR (py3.7.12, deepTCR2.1.0, tf2.7, CUDAtk11.2, cuDNN8.1, fastcluster1.2.6)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from DeepTCR.DeepTCR import DeepTCR_WF
import pandas as pd
import numpy as np
import pickle
import os
import sys
from contextlib import redirect_stdout
import io
import pdb
output = io.StringIO()
# Global variable for storing seeds
seeds = None

def safe_create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)  # exist_ok=True avoids errors if the directory already exists
        print(f"Directory created or already exists: {path}")
    except OSError as e:
        print(f"Error creating directory {path}: {e}")

def load_seeds(seed_file_path):
    try:
        seeds = np.load(seed_file_path)
        print(f"Seeds loaded from: {seed_file_path}")
        return seeds
    except FileNotFoundError:
        print(f"Seed file not found: {seed_file_path}")
        return None
    except Exception as e:
        print(f"Error loading seed file {seed_file_path}: {e}")
        return None
    
def save_seeds(seed_file_path, seeds):
    try:
        np.save(seed_file_path, seeds)
        print(f"Seeds saved to: {seed_file_path}")
    except Exception as e:
        print(f"Error saving seeds to {seed_file_path}: {e}")

def initialize_or_load_seeds(folds, training_object_name, seed_type):
    global seeds  # Use the global variable for seeds
    seed_directory = '/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/Shared_Seeds/'
    
    # Create a unique seed file for each combination of seed_type and folds
    seed_file_name = f'{seed_type}_seeds_{folds}_folds.npy'
    seed_file_path = os.path.join(seed_directory, seed_file_name)
    
    # Create the directory if it doesn't exist
    safe_create_directory(seed_directory)
    
    # Attempt to load seeds from file
    if os.path.exists(seed_file_path):
        seeds = np.load(seed_file_path)
        print(f"Seeds for {folds} folds loaded from: {seed_file_path}")
    else:
        # If no seed exists for this folds value, generate new seeds
        seeds = np.random.randint(0, 10000, size=folds)
        np.save(seed_file_path, seeds)
        print(f"New seeds generated and saved to: {seed_file_path}")
    
    return seeds
    
def set_data_train_directory():
    global input_data_folder_name, directory_path  # Declare as global
    input_data_folder_name = training_object_name.split('_', maxsplit=4)[-1]
    directory_path = f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_train/IGRP_{input_data_folder_name}/'
    print("Training data directory is set:", directory_path)

def log_training_details(training_object_name, input_data_folder_name, training_type, test_size, folds, epochs_min, size_of_net,
                         kernel, num_concepts, LOO, combine_train_valid, train_loss_min, hinge_loss_t, qualitative_agg,
                         quantitative_agg, num_agg_layers, units_agg, num_fc_layers, units_fc, weight_by_class, l2_reg, multisample_dropout, multisample_dropout_rate):
    
    # Define the file path for the log CSV
    log_file = '/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/Training_Log.csv'
    
    # Create a dictionary to store the training details
    training_details = {
        'training_object_name': training_object_name,
        'Training_data': input_data_folder_name,
        'training_type': training_type,
        'test_size': test_size,
        'folds': folds,
        'epochs_min': epochs_min,
        'size_of_net': size_of_net,
        'kernel': kernel,
        'num_fc_layers': num_fc_layers,
        'units_fc': units_fc,
        'num_concepts': num_concepts,
        'ms_dropout': multisample_dropout,
        'ms_drop_rate': multisample_dropout_rate,
        'LOO': LOO,
        'l2_reg': l2_reg,
        'combine_train_valid': combine_train_valid,
        'train_loss_min': train_loss_min,
        'hinge_loss_t': hinge_loss_t,
        'qualitative_agg': qualitative_agg,
        'quantitative_agg': quantitative_agg,
        'num_agg_layers': num_agg_layers,
        'units_agg': units_agg,
        'weight_by_class': weight_by_class
    }
    
    # If the log file exists, append to it. Otherwise, create a new one.
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = log_df.append(training_details, ignore_index=True)
    else:
        log_df = pd.DataFrame([training_details])

    # Save the updated DataFrame to the CSV file
    log_df.to_csv(log_file, index=False)
    print(f"Training details logged to {log_file}.")

def input_training_name():
    global DTCR_WF  # Declare DTCR_WF global so other functions can use it
    DTCR_WF = DeepTCR_WF(training_object_name)

def set_input_train_data():
    global DTCR_WF, directory_path  # Use global variables
    DTCR_WF.Get_Data(directory=directory_path,  # Continue as usual...
                     Load_Prev_Data=False,
                     aggregate_by_aa=True,
                     sep=',',
                     aa_column_beta=6,
                     count_column=3,
                     v_beta_column=7,
                     d_beta_column=8,
                     j_beta_column=9)
    print("Training data input successful")







def set_inference_name():
    from DeepTCR.DeepTCR import DeepTCR_WF
    global DTCR_WF
    DTCR_WF = DeepTCR_WF(training_object_name)
        
def input_inference_data():  
    inference_data_directory = f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_cut/{inference_data_name}'
    DTCR_WF.Get_Data(directory=inference_data_directory,
                         Load_Prev_Data=False,
                         aggregate_by_aa=False,
                         sep=',',
                         aa_column_beta=6,
                         v_beta_column=7,
                         d_beta_column=8,
                         j_beta_column=9)
    print("Inference data input successful")

def set_sample_inference():
    beta_sequences = DTCR_WF.beta_sequences
    v_beta = DTCR_WF.v_beta
    d_beta = DTCR_WF.d_beta
    j_beta = DTCR_WF.j_beta
    return beta_sequences, v_beta, d_beta, j_beta

def run_sample_inference(beta_sequences, v_beta, d_beta, j_beta):  
    DTCR_WF.Sample_Inference(sample_labels=None, 
                             beta_sequences=beta_sequences, 
                             v_beta=v_beta, 
                             d_beta=d_beta, 
                             j_beta=j_beta,
                             batch_size=1000,  
                             return_dist=False)

def modify_df(sample_list, inference_pred):
    df_samplelist = DTCR_WF.Inference_Sample_List
    df_samplelist = pd.DataFrame(df_samplelist)
    df_infpred = DTCR_WF.Inference_Pred
    df_infpred = pd.DataFrame(df_infpred)
    return df_samplelist, df_infpred
    
def modify_df_1(df_samplelist, df_infpred):
    list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
    list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
    list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
    return list_pred
    
def modify_df_2(list_pred):
    sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
    sorted_pred = sorted_pred.reset_index(drop=True)
    return sorted_pred

def import_validation_dataset(inference_data_directory):
    # Look for CSV files in the specified directory
    csv_files = [f for f in os.listdir(inference_data_directory) if f.endswith('.csv')]
    
    # Ensure that only one CSV file is present in the directory
    if len(csv_files) == 1:
        csv_file_path = os.path.join(inference_data_directory, csv_files[0])
        print(f"CSV file found: {csv_file_path}")
        input_df = pd.read_csv(csv_file_path)
    else:
        print("Error: No CSV file or multiple CSV files found in the directory.")
        sys.exit(1)
    
    # Print a message confirming that the data has been loaded
    print(f"Validation data loaded from {csv_file_path}:")
    
    return input_df

def calculate_prediction_stats(df_pred, input_df, training_object_name, inference_data_name):
    # Check if df_pred and input_df have the same number of rows
    if df_pred.shape[0] == input_df.shape[0]:
        print("Both df_pred and input_df have the same number of rows.")
    else:
        print(f"Row count mismatch: df_pred has {df_pred.shape[0]} rows and input_df has {input_df.shape[0]} rows.")

    # Create 'Pred' column based on comparison of 'CD8_prob' and 'TETpos_prob'
    df_pred['Pred'] = np.where(df_pred['CD8_prob'] > df_pred['TETpos_prob'], 'TETneg', 'TETpos')
    
    # Combine input_df with df_pred on the right side
    combined_df = pd.concat([df_pred, input_df[['class']]], axis=1)
    print(combined_df)

    # Calculate TP, FP, TN, FN based on the class and predicted values
    TP = ((combined_df['Pred'] == 'TETpos') & (combined_df['class'] == 'TETpos')).sum()
    FP = ((combined_df['Pred'] == 'TETpos') & (combined_df['class'] == 'TETneg')).sum()
    TN = ((combined_df['Pred'] == 'TETneg') & (combined_df['class'] == 'TETneg')).sum()
    FN = ((combined_df['Pred'] == 'TETneg') & (combined_df['class'] == 'TETpos')).sum()

    # Calculate the total number of positive (P) and negative (N) samples in the class
    P = (combined_df['class'] == 'TETpos').sum()
    N = (combined_df['class'] == 'TETneg').sum()

    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    
    # Return the TP, FP, TN, FN values and pass them to update_inference_stats
    return TP, FP, TN, FN, P, N

def create_inference_stats():
    # Load the existing Inference_stats.csv if it exists, otherwise create an empty DataFrame
    global Inference_stats  # Use global variable
    inference_stats_file = '/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/Inference_stats.csv'
    
    if os.path.exists(inference_stats_file):
        Inference_stats = pd.read_csv(inference_stats_file)
    else:
        # Create an empty DataFrame if the file doesn't exist
        Inference_stats = pd.DataFrame(columns=['model_name'])


def update_inference_stats(Inference_stats, training_object_name, inference_data_name, TP, FP, TN, FN, TPR, FPR, TNR, FNR, PPV, NPV, b_accuracy, F1, AUC_value):
    # Ensure no duplicate columns exist
    Inference_stats = Inference_stats.loc[:, ~Inference_stats.columns.duplicated()]

    # Dynamically create column names that include inference_data_name
    col_TP = f'TP ({inference_data_name})'
    col_FP = f'FP ({inference_data_name})'
    col_TN = f'TN ({inference_data_name})'
    col_FN = f'FN ({inference_data_name})'
    col_b_accuracy = f'Balanced Accuracy ({inference_data_name})'
    col_PPV = f'PPV ({inference_data_name})'
    col_NPV = f'NPV ({inference_data_name})'
    col_TPR = f'TPR ({inference_data_name})'
    col_TNR = f'TNR ({inference_data_name})'
    col_F1 = f'F1 ({inference_data_name})'
    col_ROC = f'ROC ({inference_data_name})'
    col_FPR = f'FPR ({inference_data_name})'
    col_FNR = f'FNR ({inference_data_name})'

    # Check if the model_name already exists in the DataFrame
    if 'model_name' in Inference_stats.columns:
        matching_indices = Inference_stats.index[Inference_stats['model_name'] == training_object_name]

        if len(matching_indices) > 0:
            index = matching_indices[0]
            
            # Add columns if they don't exist for the specific dataset (e.g., Kas_25_cut)
            for col in [col_TP, col_FP, col_TN, col_FN, col_b_accuracy, col_PPV, col_NPV, col_TPR, col_TNR, col_F1, col_ROC, col_FPR, col_FNR]:
                if col not in Inference_stats.columns:
                    Inference_stats[col] = np.nan

            # Now update the values for the current inference_data_name
            Inference_stats.at[index, col_TP] = TP
            Inference_stats.at[index, col_FP] = FP
            Inference_stats.at[index, col_TN] = TN
            Inference_stats.at[index, col_FN] = FN
            Inference_stats.at[index, col_b_accuracy] = b_accuracy
            Inference_stats.at[index, col_PPV] = PPV
            Inference_stats.at[index, col_NPV] = NPV
            Inference_stats.at[index, col_TPR] = TPR
            Inference_stats.at[index, col_TNR] = TNR
            Inference_stats.at[index, col_F1] = F1
            Inference_stats.at[index, col_ROC] = AUC_value
            Inference_stats.at[index, col_FPR] = FPR
            Inference_stats.at[index, col_FNR] = FNR
        else:
            # Add a new row if no matching model_name is found, with the new dataset's statistics
            new_row = {
                'model_name': training_object_name,
                col_TP: TP,
                col_FP: FP,
                col_TN: TN,
                col_FN: FN,
                col_b_accuracy: b_accuracy,
                col_PPV: PPV,
                col_NPV: NPV,
                col_TPR: TPR,
                col_TNR: TNR,
                col_F1: F1,
                col_ROC: AUC_value,
                col_FPR: FPR,
                col_FNR: FNR
            }
            Inference_stats = Inference_stats.append(new_row, ignore_index=True)
    else:
        # If the DataFrame is empty or missing columns, initialize with a new row
        new_row = {
            'model_name': training_object_name,
            col_TP: TP,
            col_FP: FP,
            col_TN: TN,
            col_FN: FN,
            col_b_accuracy: b_accuracy,
            col_PPV: PPV,
            col_NPV: NPV,
            col_TPR: TPR,
            col_TNR: TNR,
            col_F1: F1,
            col_ROC: AUC_value,
            col_FPR: FPR,
            col_FNR: FNR
        }
        Inference_stats = pd.DataFrame([new_row])

    # Ensure no duplicate columns
    Inference_stats = Inference_stats.loc[:, ~Inference_stats.columns.duplicated()]

    # Enforce the correct column order for all datasets in the DataFrame
    all_inference_data_names = [col.split('(')[1].split(')')[0] for col in Inference_stats.columns if '(' in col]
    unique_inference_data_names = sorted(set(all_inference_data_names))  # Get unique inference_data_name

    # Build the desired column order for all datasets
    ordered_columns = ['model_name']
    for data_name in unique_inference_data_names:
        ordered_columns.extend([
            f'TP ({data_name})',
            f'FN ({data_name})',
            f'TN ({data_name})',
            f'FP ({data_name})',
            f'Balanced Accuracy ({data_name})',
            f'PPV ({data_name})',
            f'NPV ({data_name})',
            f'TPR ({data_name})',
            f'TNR ({data_name})',
            f'F1 ({data_name})',
            f'ROC ({data_name})',
            f'FPR ({data_name})',
            f'FNR ({data_name})'
        ])

    # Reorder DataFrame to follow the correct column order
    Inference_stats = Inference_stats[[col for col in ordered_columns if col in Inference_stats.columns]]

    return Inference_stats


def calculate_prediction_percentages(TP, FP, TN, FN, P, N, Inference_stats, training_object_name, inference_data_name, AUC_value):
    # Calculate additional metrics (Balanced Accuracy, F1, etc.)
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Positive Predictive Value
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0  # Negative Predictive Value
    TPR = TP / P if P > 0 else 0  # True Positive Rate (Recall)
    FPR = FP / N if N > 0 else 0  # False Positive Rate
    TNR = TN / N if N > 0 else 0  # True Negative Rate
    FNR = FN / P if P > 0 else 0  # False Negative Rate
    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0  # F1 Score
    b_accuracy = 0.5 * (TPR + TNR) if P > 0 and N > 0 else 0  # Balanced Accuracy

    # Define the file path for inference stats CSV
    inference_stats_file = '/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/Inference_stats.csv'

    # Load existing CSV if it exists
    if os.path.exists(inference_stats_file):
        Inference_stats = pd.read_csv(inference_stats_file)
    else:
        # If no CSV exists, create a new DataFrame with an empty structure
        Inference_stats = pd.DataFrame(columns=['model_name'])

    # Update the Inference_stats DataFrame with the new values, including TP, FP, TN, FN
    Inference_stats = update_inference_stats(
        Inference_stats, training_object_name, inference_data_name,
        TP, FP, TN, FN, TPR, FPR, TNR, FNR, PPV, NPV, b_accuracy, F1, AUC_value
    )

    # Save the updated Inference_stats DataFrame, ensuring it appends new columns if necessary
    Inference_stats.to_csv(inference_stats_file, index=False)

    # Also, update or create the Model_rank_ref.csv
    create_or_update_model_rank_ref(training_object_name, b_accuracy, PPV, TPR, F1, AUC_value)

    print("Finished sample inference, calculations, and CSV writing.")



def create_or_update_model_rank_ref(model_name, b_accuracy, PPV, TPR, F1, AUC_value):
    # Define the file path for Model_rank_ref.csv
    model_rank_ref_file = '/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/Model_rank_ref.csv'
    
    # Create a dictionary to store the model's metrics
    model_metrics = {
        'model_name': model_name,
        'b_accuracy': b_accuracy,
        'PPV': PPV,
        'TPR': TPR,
        'F1': F1,
        'ROC': AUC_value
    }

    # If the file exists, append to it, otherwise create a new DataFrame
    if os.path.exists(model_rank_ref_file):
        model_rank_ref_df = pd.read_csv(model_rank_ref_file)
        model_rank_ref_df = model_rank_ref_df.append(model_metrics, ignore_index=True)
    else:
        model_rank_ref_df = pd.DataFrame([model_metrics])


def get_auc_value(DTCR_WF):
    """
    Attempt to retrieve AUC data from the DeepTCR_WF object.
    Fallback to loading the AUC.csv file if AUC_DF is not available.
    """
    AUC_value = None
    
    # Try accessing the AUC_DF directly from the DeepTCR_WF object
    if hasattr(DTCR_WF, 'AUC_DF'):
        try:
            AUC_value = DTCR_WF.AUC_DF['AUC'].mean()
            print(f"AUC value calculated from AUC_DF: {AUC_value}")
            return AUC_value
        except KeyError:
            print("AUC_DF is present, but 'AUC' column is missing.")
    
    # Fallback to reading AUC from the saved CSV file
    auc_file = os.path.join(DTCR_WF.results_dir, 'AUC.csv')
    try:
        auc_df = pd.read_csv(auc_file)
        AUC_value = auc_df['AUC'].mean()
        print(f"AUC value calculated from AUC.csv file: {AUC_value}")
        return AUC_value
    except (FileNotFoundError, KeyError):
        print(f"Failed to retrieve AUC value from both AUC_DF and {auc_file}.")
        return None 


def initiate_training():
    global DTCR_WF, seeds, input_data_folder_name  
    AUC_value = None
    
    if 'Train' in training_object_name:
        training_type = 'Train'
        test_size = 0.25  
        LOO = None  
        combine_train_valid = False 
        
        print("Train selected for model training...")
        DTCR_WF.Get_Train_Valid_Test(test_size=test_size)
        DTCR_WF.Train()
        DTCR_WF.AUC_Curve()
        
        AUC_value = get_auc_value(DTCR_WF)
        
        # Log training details to CSV using actual values
        log_training_details(training_object_name, input_data_folder_name, training_type, test_size=test_size, folds=None,
                             epochs_min=None, size_of_net=None, num_concepts=None, LOO=LOO,
                             combine_train_valid=combine_train_valid, train_loss_min=None, 
                             hinge_loss_t=None, qualitative_agg=None,
                             quantitative_agg=None, num_agg_layers=None, 
                             units_agg=None, weight_by_class=None)

    elif 'Kf' in training_object_name:
        training_type = 'K-fold'
        folds = 5
        batch_seed=78
        seed_type = "K_Fold"
        combine_train_valid = True
        epochs_min = 25
        size_of_net = 'small'
        kernel = 5
        num_concepts = 12
        train_loss_min = 0.1
        hinge_loss_t = 0.0
        qualitative_agg = True
        quantitative_agg = False
        num_agg_layers = 0
        units_agg = 12
        weight_by_class = False
        multisample_dropout = False
        multisample_dropout_rate = 0.5
       
        initialize_or_load_seeds(folds, training_object_name, seed_type)

        DTCR_WF.K_Fold_CrossVal(combine_train_valid=combine_train_valid, hinge_loss_t=hinge_loss_t, 
                                train_loss_min=train_loss_min, qualitative_agg=qualitative_agg, 
                                quantitative_agg=quantitative_agg, weight_by_class=weight_by_class, 
                                folds=folds, batch_seed=batch_seed, epochs_min=epochs_min, num_concepts=num_concepts, 
                                size_of_net=size_of_net, kernel=kernel, multisample_dropout=multisample_dropout, multisample_dropout_rate=multisample_dropout_rate)
        DTCR_WF.AUC_Curve()
        
        AUC_value = get_auc_value(DTCR_WF)
        
        # Log training details to CSV using actual values
        log_training_details(training_object_name, input_data_folder_name, training_type, test_size=None, folds=folds,
                             epochs_min=epochs_min, size_of_net=size_of_net, num_concepts=num_concepts, LOO=None,
                             combine_train_valid=combine_train_valid, train_loss_min=train_loss_min, 
                             hinge_loss_t=hinge_loss_t, qualitative_agg=qualitative_agg, 
                             quantitative_agg=quantitative_agg, num_agg_layers=num_agg_layers, 
                             units_agg=units_agg, weight_by_class=weight_by_class, kernel=kernel,
                             multisample_dropout=multisample_dropout, multisample_dropout_rate=multisample_dropout_rate)

    elif 'Mc' in training_object_name:
        training_type = 'Monte Carlo'
        test_size = 0.25
        folds = 25
        seed_type = "Monte_Carlo" 
        LOO = 4
        combine_train_valid = True
        epochs_min = 25
        size_of_net = 'large'
        kernel = 7
        num_concepts = 64
        train_loss_min = 0.1
        hinge_loss_t = 0.1
        qualitative_agg = True
        quantitative_agg = False
        num_agg_layers = 0
        units_agg = 12
        num_fc_layers = 0
        units_fc = 12
        weight_by_class = False
        multisample_dropout = False
        multisample_dropout_rate = 0.5
        l2_reg = 0.0


        # Initialize or load seeds for Monte Carlo
        initialize_or_load_seeds(folds, training_object_name, seed_type)

        DTCR_WF.Monte_Carlo_CrossVal(combine_train_valid=combine_train_valid, qualitative_agg=qualitative_agg, 
                                     quantitative_agg=quantitative_agg, weight_by_class=weight_by_class, 
                                     folds=folds, l2_reg=l2_reg, test_size=test_size, epochs_min=epochs_min, 
                                     num_concepts=num_concepts, LOO=LOO, num_fc_layers=num_fc_layers, units_fc=units_fc, size_of_net=size_of_net, kernel=kernel,
                                     multisample_dropout=multisample_dropout, multisample_dropout_rate=multisample_dropout_rate)
        DTCR_WF.AUC_Curve()
        
        AUC_value = get_auc_value(DTCR_WF)
        
        # Log training details to CSV using actual values
        log_training_details(training_object_name, input_data_folder_name, training_type, test_size, folds=folds,
                             epochs_min=epochs_min, size_of_net=size_of_net, num_concepts=num_concepts, LOO=LOO,
                             combine_train_valid=combine_train_valid, train_loss_min=train_loss_min, 
                             hinge_loss_t=hinge_loss_t, qualitative_agg=qualitative_agg, l2_reg=l2_reg, 
                             quantitative_agg=quantitative_agg, num_agg_layers=num_agg_layers,
                             units_agg=units_agg, num_fc_layers=num_fc_layers, units_fc=units_fc, weight_by_class=weight_by_class, kernel=kernel,
                             multisample_dropout=multisample_dropout, multisample_dropout_rate=multisample_dropout_rate)

    return AUC_value







# (1) USE THIS FOR NEW MODEL TRAINING THEN CONDUCT INFERENCE


training_object_name = 'IGRP_WF_36_Mc_Tp1-14_Tn1-8_25_cut'        #change to name of folder within 'Supervised_WF' that you have set up with this name format, where 'Train', 'Kf', 'Mc' are codes for the type of training, followed by _ and then the name of the training data folder (without IGRP_ at the beginning)
create_inference_stats()
set_data_train_directory()
input_training_name()
set_input_train_data()
AUC_value = initiate_training()


# RUN (1) THEN COMMENCE FROM HERE FOR SAMPLE INFERENCE. CHECK INFERENCE_DATA_NAME AND ALL FILE DIRECTORIES FOR INFERENCE ARE SET CORRECTLY

inference_data_name = 'Kas_no_cut' #change to folder name that contains the csv within the below inference data directory
set_inference_name()
input_inference_data()
beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
df_samplelist, df_infpred = modify_df(DTCR_WF.sample_list, DTCR_WF.predicted)
list_pred = modify_df_1(df_samplelist, df_infpred)
df_pred = modify_df_2(list_pred)
input_df = import_validation_dataset(f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_cut/{inference_data_name}/')
TP, FP, TN, FN, P, N = calculate_prediction_stats(df_pred, input_df, training_object_name, inference_data_name)
calculate_prediction_percentages(TP, FP, TN, FN, P, N, Inference_stats, training_object_name, inference_data_name, AUC_value)
inference_data_name = 'Kas_25_cut' #change to folder name that contains the csv within the below inference data directory
set_inference_name()
input_inference_data()
beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
df_samplelist, df_infpred = modify_df(DTCR_WF.sample_list, DTCR_WF.predicted)
list_pred = modify_df_1(df_samplelist, df_infpred)
df_pred = modify_df_2(list_pred)
input_df = import_validation_dataset(f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_cut/{inference_data_name}/')
TP, FP, TN, FN, P, N = calculate_prediction_stats(df_pred, input_df, training_object_name, inference_data_name)
calculate_prediction_percentages(TP, FP, TN, FN, P, N, Inference_stats, training_object_name, inference_data_name, AUC_value)
inference_data_name = 'Kas_50_cut' #change to folder name that contains the csv within the below inference data directory
set_inference_name()
input_inference_data()
beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
df_samplelist, df_infpred = modify_df(DTCR_WF.sample_list, DTCR_WF.predicted)
list_pred = modify_df_1(df_samplelist, df_infpred)
df_pred = modify_df_2(list_pred)
input_df = import_validation_dataset(f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_cut/{inference_data_name}/')
TP, FP, TN, FN, P, N = calculate_prediction_stats(df_pred, input_df, training_object_name, inference_data_name)
calculate_prediction_percentages(TP, FP, TN, FN, P, N, Inference_stats, training_object_name, inference_data_name, AUC_value)
inference_data_name = 'Kas_75_cut' #change to folder name that contains the csv within the below inference data directory
set_inference_name()
input_inference_data()
beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
df_samplelist, df_infpred = modify_df(DTCR_WF.sample_list, DTCR_WF.predicted)
list_pred = modify_df_1(df_samplelist, df_infpred)
df_pred = modify_df_2(list_pred)
input_df = import_validation_dataset(f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_cut/{inference_data_name}/')
TP, FP, TN, FN, P, N = calculate_prediction_stats(df_pred, input_df, training_object_name, inference_data_name)
calculate_prediction_percentages(TP, FP, TN, FN, P, N, Inference_stats, training_object_name, inference_data_name, AUC_value)
inference_data_name = 'Kas_all_cut' #change to folder name that contains the csv within the below inference data directory
set_inference_name()
input_inference_data()
beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
df_samplelist, df_infpred = modify_df(DTCR_WF.sample_list, DTCR_WF.predicted)
list_pred = modify_df_1(df_samplelist, df_infpred)
df_pred = modify_df_2(list_pred)
input_df = import_validation_dataset(f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_cut/{inference_data_name}/')
TP, FP, TN, FN, P, N = calculate_prediction_stats(df_pred, input_df, training_object_name, inference_data_name)
calculate_prediction_percentages(TP, FP, TN, FN, P, N, Inference_stats, training_object_name, inference_data_name, AUC_value)
inference_data_name = 'Kas_Gearty_no_cut' #change to folder name that contains the csv within the below inference data directory
set_inference_name()
input_inference_data()
beta_sequences, v_beta, d_beta, j_beta = set_sample_inference()
run_sample_inference(beta_sequences, v_beta, d_beta, j_beta)
df_samplelist, df_infpred = modify_df(DTCR_WF.sample_list, DTCR_WF.predicted)
list_pred = modify_df_1(df_samplelist, df_infpred)
df_pred = modify_df_2(list_pred)
input_df = import_validation_dataset(f'/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_cut/{inference_data_name}/')
TP, FP, TN, FN, P, N = calculate_prediction_stats(df_pred, input_df, training_object_name, inference_data_name)
calculate_prediction_percentages(TP, FP, TN, FN, P, N, Inference_stats, training_object_name, inference_data_name, AUC_value)














