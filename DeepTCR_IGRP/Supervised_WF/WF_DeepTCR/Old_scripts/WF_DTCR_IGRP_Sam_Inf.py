# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:10:06 2024

@author: Mitch
"""

# venv = IGRP_DTCR (py3.7.12, deepTCR2.1.0, tf2.7, CUDAtk11.2, cuDNN8.1, fastcluster1.2.6)

import tensorflow as tf
from DeepTCR.DeepTCR import DeepTCR_WF
import pandas as pd
import numpy as np
import pickle
DTCR_WF = DeepTCR_WF('IGRP_WF_8_added_k_fold')

# Import data (original)

DTCR_WF.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_train/IGRP_orig/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=True,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)


# Import filtered mean cutoff data

DTCR_WF.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_train/IGRP_filt/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# Import added data with TETpos M9-14

DTCR_WF.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_train/IGRP_added/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# Import filtered 20% cutoff data 

DTCR_WF.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_train/IGRP_filt_20/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# Import filtered 35% cutoff data 

DTCR_WF.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_train/IGRP_filt_35/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)


# ==================== TRAIN MODEL ====================

# --- 1. Standard ---

%%capture
DTCR_WF.Get_Train_Valid_Test(test_size=0.25)
DTCR_WF.Train()
DTCR_WF.AUC_Curve()

# --- 2. K-fold Cross Validation ---

%%capture
test_size=0.2
folds=5
size_of_net='small'
num_concepts=64
hinge_loss_t=0.1
train_loss_min=0.1
DTCR_WF.K_Fold_CrossVal(combine_train_valid=True, hinge_loss_t = hinge_loss_t,train_loss_min = train_loss_min,folds=folds,
                       num_concepts=num_concepts, size_of_net=size_of_net)
DTCR_WF.AUC_Curve()


# --- 3. Monte-Carlo Cross Validation ---

%%capture
test_size=0.25
folds=25
LOO=4
epochs_min=10
size_of_net='small'
num_concepts=64
hinge_loss_t=0.1
train_loss_min=0.1
DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,num_concepts=num_concepts,size_of_net=size_of_net,
                             train_loss_min=train_loss_min,hinge_loss_t=hinge_loss_t,combine_train_valid=True)
DTCR_WF.AUC_Curve()



# ==================== SAMPLE INFERENCE INPUT ====================

from DeepTCR.DeepTCR import DeepTCR_WF
DTCR_WF_Kasmani = DeepTCR_WF('kasmani')

# --- Normal data input ---

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_all/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Filtered 5% Kasmani input

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_filt/Kasmani_filt_5/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Filtered 10% Kasmani input

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_filt/Kasmani_filt_10/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Filtered 15% Kasmani input

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_filt/Kasmani_filt_15/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Kasmani all duplicates removed between CD8 and TETpos

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_dupe_remove/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Gearty 2022 TETpos all data ---

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Gearty_TETpos_all/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)




# --- Filtered 20% Kasmani input

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_filt/Kasmani_filt_20/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Filtered 25% Kasmani input

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_filt/Kasmani_filt_25/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Merged Kasmani input ---

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_all_merged/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)

# --- Kasmani all duplicates removed between CD8 and TETpos, duplicates within CD8 and TETpos merged ---

DTCR_WF_Kasmani.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_valid/Kasmani_dupe_rem_merged/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=False,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)



# --- CONDUCT SAMPLE INFERENCE --- 

beta_sequences = DTCR_WF_Kasmani.beta_sequences
v_beta = DTCR_WF_Kasmani.v_beta
d_beta = DTCR_WF_Kasmani.d_beta
j_beta = DTCR_WF_Kasmani.j_beta

%%capture
DTCR_WF.Sample_Inference(sample_labels=None, 
                         beta_sequences=beta_sequences, 
                         v_beta=v_beta, 
                         d_beta=d_beta, 
                         j_beta=j_beta,
                         batch_size=100,  
                         return_dist=False)

df_samplelist = DTCR_WF.Inference_Sample_List
df_samplelist = pd.DataFrame(df_samplelist)
df_infpred = DTCR_WF.Inference_Pred
df_infpred = pd.DataFrame(df_infpred)

list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
sorted_pred = sorted_pred.reset_index(drop=True)
sorted_pred.to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/IGRP_WF_8_added_k_fold/inference/kasmani_all/sorted_pred.csv', index=False)
print(sorted_pred)

list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
sorted_pred = sorted_pred.reset_index(drop=True)
sorted_pred.to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/IGRP_WF_8_added_k_fold/inference/kasmani_filt_5/sorted_pred.csv', index=False)
print(sorted_pred)

list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
sorted_pred = sorted_pred.reset_index(drop=True)
sorted_pred.to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/IGRP_WF_8_added_k_fold/inference/kasmani_filt_10/sorted_pred.csv', index=False)
print(sorted_pred)

list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
sorted_pred = sorted_pred.reset_index(drop=True)
sorted_pred.to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/IGRP_WF_8_added_k_fold/inference/kasmani_filt_15/sorted_pred.csv', index=False)
print(sorted_pred)

list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
sorted_pred = sorted_pred.reset_index(drop=True)
sorted_pred.to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/IGRP_WF_8_added_k_fold/inference/kasmani_dupe_remove/sorted_pred.csv', index=False)
print(sorted_pred)

list_pred = pd.concat([df_samplelist, df_infpred], axis=1)
list_pred.columns = ['sample_list_ID', 'CD8_prob', 'TETpos_prob']
list_pred['sample_list_ID'] = pd.to_numeric(list_pred['sample_list_ID'], errors='coerce')
sorted_pred = list_pred.sort_values(by='sample_list_ID', ascending=True)
sorted_pred = sorted_pred.reset_index(drop=True)
sorted_pred.to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/IGRP_WF_8_added_k_fold/inference/gearty_tetpos_all/sorted_pred.csv', index=False)
print(sorted_pred)

# Copy columns from sorted_pred.csv into same columns in Pred_kasmani.xlsx or Pred_kasmani_5 etc., and correct prediction percentages automatically calculated

