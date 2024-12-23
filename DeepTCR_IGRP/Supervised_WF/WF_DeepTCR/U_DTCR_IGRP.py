# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 02:55:04 2024

@author: Mitch
"""
# venv = IGRP_DTCR (py3.7.12, deepTCR2.1.0, tf2.7, CUDAtk11.2, cuDNN8.1, fastcluster1.2.6)

# import tensorflow, deepTCR, and TCR data

import tensorflow as tf

print("Num GPUs Available", len(tf.config.experimental.list_physical_devices('GPU')))

from DeepTCR.DeepTCR import DeepTCR_U

DTCRU = DeepTCR_U('tp8')

DTCRU.Get_Data(directory = '/Users/Mitch/Documents/DeepTCR_IGRP/Supervised_WF/WF_DeepTCR/Data_train/IGRP_dist/',
               Load_Prev_Data=False,
               aggregate_by_aa=True,
               sep=',',
               aa_column_beta=6,
               count_column=3,
               v_beta_column=7,
               d_beta_column=8,
               j_beta_column=9)

#Train VAE

# 1 --------------------



DTCRU.Train_VAE(Load_Prev_Data=False)

features = DTCRU.features
print(features.shape)

DTCRU.HeatMap_Sequences(figsize=(18, 18))

import matplotlib.pyplot as plt
plt.plot(DTCRU.explained_variance_ratio_)

import numpy as np
plt.plot(np.cumsum(DTCRU.explained_variance_ratio_))



# 2 --------------------


DTCRU.Train_VAE(Load_Prev_Data=False,accuracy_min=0.9)

%%capture
DTCRU.Train_VAE(Load_Prev_Data=False,accuracy_min=0.9,latent_dim=12,use_only_seq=True)

features = DTCRU.features
print(features.shape)

DTCRU.HeatMap_Sequences(figsize=(18, 18))

import matplotlib.pyplot as plt
plt.plot(DTCRU.explained_variance_ratio_)

import numpy as np
plt.plot(np.cumsum(DTCRU.explained_variance_ratio_))



# 3 --------------------

%%capture
DTCRU.Train_VAE(Load_Prev_Data=False,sparsity_alpha=1.0,var_explained=0.99,use_only_seq=True)

features = DTCRU.features
print(features.shape)

DTCRU.HeatMap_Sequences(figsize=(18, 18))

import matplotlib.pyplot as plt
plt.plot(DTCRU.explained_variance_ratio_)

import numpy as np
plt.plot(np.cumsum(DTCRU.explained_variance_ratio_))

#Clustering (uses last trained VAE)

DTCRU.Cluster(clustering_method='phenograph',write_to_sheets=True)
DFs = DTCRU.Cluster_DFs
print(DFs[0])

DTCRU.Cluster(clustering_method='hierarchical',write_to_sheets=True)
DFs = DTCRU.Cluster_DFs
print(DFs[0])

DTCRU.Cluster(clustering_method='dbscan',write_to_sheets=True)
DFs = DTCRU.Cluster_DFs
print(DFs[0])

for i in range(0, len(DFs)):
    print(DFs[i].shape)
    DFs[i].to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Unsupervised/IGRP_U_3/output_all/' + 'VAEcluster_' + str(i) + '.csv')


DTCRU.UMAP_Plot(by_class=True,show_legend=True,freq_weight=True,scale=1000,alpha=0.7,filename="clusterplot.png",Load_Prev_Data=False,plot_by_class=False)

DTCRU.UMAP_Plot(by_class=True,show_legend=True,freq_weight=False,scale=5,Load_Prev_Data=True)

DTCRU.UMAP_Plot(by_class=True,show_legend=True,freq_weight=True,scale=750,alpha=0.7,Load_Prev_Data=True)

DTCRU.HeatMap_Samples()

DTCRU.Structural_Diversity()

print(DTCRU.Structural_Diversity_DF)

DTCRU.Repertoire_Dendrogram(n_jobs=20,distance_metric='KL')

DTCRU.Repertoire_Dendrogram(n_jobs=20,distance_metric='KL',lw=5,gridsize=40,gaussian_sigma=0.5,Load_Prev_Data=True,
                           dendrogram_radius=0.40,repertoire_radius=0.47)

DTCRU.UMAP_Plot_Samples(scale=100)
DTCRU.Sample_Features()
Samplefeatures = DTCRU.sample_features

Samplefeatures.to_csv('/Users/Mitch/Documents/DeepTCR_IGRP/Unsupervised/IGRP_U_3/output_all/samplefeatures.csv')

# --- Motif ID ---

DTCRU.Motif_Identification(group='TETpos',by_samples=False)

DTCRU.Motif_Identification(group='CD8',by_samples=False)






