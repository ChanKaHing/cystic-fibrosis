#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


# In[70]:


def corr_3D(df, labels):
    df_1 = df[['MSV(mL/mL)','TV(L)','VH(%)']]
    df_2 = df[['MSV(mL/mL)','TV(L)','VDP(%)']]
    
    # Create 3D scatter plots
    fig1 = px.scatter_3d(df_1, x='MSV(mL/mL)', y='TV(L)', z='VH(%)', color=labels, title='MSV vs TV vs VH')
    fig2 = px.scatter_3d(df_2, x='MSV(mL/mL)', y='TV(L)', z='VDP(%)', color=labels, title='MSV vs TV vs VDP')

    # Combine figures side by side
    fig = make_subplots(rows=1, cols=2, subplot_titles=['MSV vs TV vs VH', 'MSV vs TV vs VDP'], specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

    # Add subplots to the combined figure
    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig2['data']:
        fig.add_trace(trace, row=1, col=2)

    # Update layout for better visualization
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(scene=dict(aspectmode="cube"))
    fig.update_layout(scene=dict(xaxis_title='MSV', yaxis_title='TV', zaxis_title='VH'), 
                      scene2=dict(xaxis_title='MSV', yaxis_title='TV', zaxis_title='VDP'))

    # Show the figure
    fig.show()


# In[71]:


def rat_bead_study_data(df_rat_sterile_baseline,df_rat_sterile_post_beads):
    df_rat_combine = pd.concat([df_rat_sterile_baseline, df_rat_sterile_post_beads], ignore_index=True)
    labels = df_rat_combine['Beads']
    return df_rat_combine, labels


# In[73]:


def rat_pa_study_data(df_rat_pa):
    df_rat_WT = df_rat_pa[df_rat_pa['Genotype'] == 'WT']
    df_rat_KO = df_rat_pa[df_rat_pa['Genotype'] == 'KO']
    df_rat_CF = df_rat_pa[df_rat_pa['Genotype'] == 'CF']
    df_rat_pa_combine = pd.concat([df_rat_WT, df_rat_KO, df_rat_CF], ignore_index=True)
    labels = df_rat_pa_combine['Genotype']
    return df_rat_pa_combine, labels


# In[75]:


def mouse_b_enac_study_data(df_mouse_b_enac):
    df_WT = df_mouse_b_enac[df_mouse_b_enac['Genotype'] == 'WT']
    df_b_enac = df_mouse_b_enac[df_mouse_b_enac['Genotype'] == 'B_Enac']
    df_mouse_enac_combine = pd.concat([df_WT, df_b_enac], ignore_index=True)
    labels = df_mouse_enac_combine['Genotype']
    return df_mouse_enac_combine, labels


# In[84]:


def mouse_mps_study_data(df_mouse_mps):
    df_WT = df_mouse_mps[df_mouse_mps['Genotype'] == 'WT']
    df_MPS_I = df_mouse_mps[df_mouse_mps['Genotype'] == 'MPS I']
    df_Het = df_mouse_mps[df_mouse_mps['Genotype'] == 'Het']
    df_mouse_mps_combine = pd.concat([df_WT, df_MPS_I, df_Het], ignore_index=True)
    labels = df_mouse_mps_combine['Genotype']
    return df_mouse_mps_combine, labels


# In[ ]:




