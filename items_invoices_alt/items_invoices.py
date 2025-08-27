#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import glob
import os


# In[12]:


directory_path = os.path.expanduser("~/Jupyter/items_invoices/Invoices/*.txt")
file_paths = glob.glob(directory_path)


# In[13]:


dfs = [pd.read_csv(file, delimiter='\t', skiprows=6, low_memory=False) for file in file_paths]


# In[14]:


df = pd.concat(dfs, ignore_index=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[15]:


len(df)


# In[16]:


df.to_csv('items_sold_on_invoices.txt', sep='\t', index=False)


# In[ ]:




