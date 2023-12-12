#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
def files_in_folders(path):
    file_list = os.listdir(path)
    file_list = [file for file in file_list if 'non-norm' not in file]
    return file_list


# In[28]:


file_list = files_in_folders('./Mouse B-ENaC Study/csv')


# In[29]:


import csv 
def to_comparison(file_list, name):
    with open("comparison.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name])
        writer.writerows([[item] for item in file_list])

