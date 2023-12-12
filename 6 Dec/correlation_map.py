#!/usr/bin/env python
# coding: utf-8

# In[19]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns


# In[54]:


def corr_map(df, method, title):
    df_fea = df[['VDP(%)','MSV(mL/mL)','TV(L)','VH(%)','VHSS(%)','VHLS(%)']]
    corr_matrix = df_fea.corr(method=method)
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 18}, cmap='viridis')
    plt.title(title)
    plt.show()


# In[ ]:




