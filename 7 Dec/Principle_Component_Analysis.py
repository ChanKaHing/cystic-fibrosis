#!/usr/bin/env python
# coding: utf-8

# In[89]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# # PCA

# # Rat bead study  (baseline vs post beads)

# In[96]:


def pca_2D(df, title):
    
    features = df[['VDP(%)','MSV(mL/mL)','TV(L)','VH(%)','VHSS(%)','VHLS(%)']]
    
    if 'Beads' in df.columns:
        target = df['Beads']
    else:
        target = df['Genotype']

    # Standardize the features
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    # Apply PCA with two components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_standardized)
    explained_variance_ratios = pca.explained_variance_ratio_
    print('The amount of variance explained by [PC1 PC2] =', explained_variance_ratios)

    # Create a new DataFrame with the principal components and class labels
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['target'] = target

    # Plot the PCA results
    plt.figure(figsize=(7, 5))
    classes = pca_df['target'].unique()

    for c in classes:
        subset = pca_df[pca_df['target'] == c]
        plt.scatter(subset['PC1'], subset['PC2'], label=c)

    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()


# In[128]:


def pca_3D(df, title):
    
    features = df[['VDP(%)','MSV(mL/mL)','TV(L)','VH(%)','VHSS(%)','VHLS(%)']]
    
    if 'Beads' in df.columns:
        target = df['Beads']
    else:
        target = df['Genotype']

    # Standardize the features
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    # Apply PCA with two components
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features_standardized)
    explained_variance_ratios = pca.explained_variance_ratio_
    print('The amount of variance explained by [PC1 PC2 PC3] =', explained_variance_ratios)

    # Create a new DataFrame with the principal components and class labels
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Label'] = target
    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Label', labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'}, title=title)
    fig.show()






