# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:10:06 2024

@author: Mitch
"""

# venv = IGRP_DTCR (py3.7.12, deepTCR2.1.0, tf2.7, CUDAtk11.2, cuDNN8.1, fastcluster1.2.6)


import tensorflow as tf

from DeepTCR.DeepTCR import DeepTCR_WF

DTCR_WF = DeepTCR_WF('IGRP_WF_2')

DTCR_WF.Get_Data(directory='/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data/IGRPdata/',
                 Load_Prev_Data=False,
                 aggregate_by_aa=True,
                 sep=',',
                 aa_column_beta=6,
                 count_column=3,
                 v_beta_column=7,
                 d_beta_column=8,
                 j_beta_column=9)


# --- AUC_1 ---

DTCR_WF.Get_Train_Valid_Test(test_size=0.20)
DTCR_WF.Train()

DTCR_WF.AUC_Curve()

# --- AUC_2 ---

%%capture
folds = 5
size_of_net = 'small'
num_concepts=64
hinge_loss_t = 0.1
train_loss_min=0.1
DTCR_WF.K_Fold_CrossVal(combine_train_valid=True, hinge_loss_t = hinge_loss_t,train_loss_min = train_loss_min,folds=folds,
                       num_concepts=num_concepts, size_of_net=size_of_net)

DTCR_WF.AUC_Curve()

# --- AUC_3 ---

%%capture
folds = 25
LOO = 4
epochs_min = 10
size_of_net = 'small'
num_concepts=64
hinge_loss_t = 0.1
train_loss_min=0.1
DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,num_concepts=num_concepts,size_of_net=size_of_net,
                             train_loss_min=train_loss_min,hinge_loss_t=hinge_loss_t,combine_train_valid=True)

DTCR_WF.AUC_Curve()

# --- Representative Sequences & Motif ID ---

DTCR_WF.Representative_Sequences()

print(DTCR_WF.Rep_Seq['TETpos'])

print(DTCR_WF.Rep_Seq['CD8'])

DTCR_WF.Motif_Identification('TETpos')

DTCR_WF.Motif_Identification('CD8')
