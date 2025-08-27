#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import numpy as np


# In[22]:


df_isi = pd.read_csv('items_sold_on_invoices_alt.txt', delimiter='\t', low_memory=False)
df = pd.read_csv('product_listing_with_variables.csv')


# In[23]:


df_isi.columns = df_isi.columns.str.lower()
df_isi.rename(columns={'part#': 'part_code', 'bill-to': 'customer_id'}, inplace=True)
df_isi = df_isi.dropna(subset=['part_code'])
df_isi['part_code'] = pd.to_numeric(df_isi['part_code'], errors='coerce')
df_isi['part_code'] = df_isi['part_code'].astype(float)
df_isi['invoice date'] = pd.to_datetime(df_isi['invoice date'])
df_isi = df_isi[df_isi['invoice date'].dt.year >= 2025]


# In[24]:


print(df_isi.columns)


# In[25]:


numeric_cols = ['base price', 'sold for', 'extended price', 'cost', 'extended cost', 'total gross profit']

def clean_and_convert(col):
    return pd.to_numeric(col.replace('[\$,]', '', regex=True), errors='coerce')

for col in numeric_cols:
    df_isi[col] = clean_and_convert(df_isi[col])


# In[26]:


customer_data = df_isi.groupby('customer_id').agg(
    purchase_frequency=('invoice date', 'count'),  
    average_order_value=('extended price', 'mean'), 
    total_spent=('extended price', 'sum'), 
    recency=('invoice date', lambda x: (pd.to_datetime('today') - x.max()).days)  
).reset_index()

print(customer_data.head())
customer_data.to_csv('customer_data.csv', index=False)


# In[27]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['purchase_frequency', 'average_order_value', 'total_spent', 'recency']])

customer_data_scaled = pd.DataFrame(customer_data_scaled, columns=['purchase_frequency', 'average_order_value', 'total_spent', 'recency'])

print(customer_data_scaled.head())


# In[28]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[29]:


n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(customer_data_scaled)

print(customer_data.head())
customer_data.to_csv('customer_data.csv', index=False)
      


# In[30]:


cluster_summary = customer_data.groupby('cluster').agg({
    'purchase_frequency': 'mean',
    'average_order_value': 'mean',
    'total_spent': 'mean',
    'recency': 'mean'
}).reset_index()

print(cluster_summary)


# In[31]:


import seaborn as sns

customer_data_scaled['cluster'] = customer_data['cluster']

sns.pairplot(customer_data_scaled, hue='cluster', palette='viridis')
plt.show()


# In[32]:


df_isi_with_clusters = pd.merge(df_isi, customer_data[['customer_id', 'cluster']], on='customer_id', how='left')

item_cluster_summary = df_isi_with_clusters.groupby(['part_code', 'cluster']).agg(
    total_quantity_sold=('qty sold', 'sum'),
    total_revenue=('extended price', 'sum'),
    average_order_value=('extended price', 'mean')
).reset_index()

print(item_cluster_summary.head())


# In[33]:


cluster_priority = {
    0: 6,  
    1: 5,  
    2: 3,  
    3: 10, 
    4: 4,  
    5: 7  
}

item_cluster_summary['priority'] = item_cluster_summary['cluster'].map(cluster_priority)

item_cluster_summary['weighted_score'] = item_cluster_summary['total_revenue'] * item_cluster_summary['priority']

item_priority = item_cluster_summary.groupby('part_code').agg(
    total_weighted_score=('weighted_score', 'sum'),
    total_revenue=('total_revenue', 'sum'),
    total_quantity_sold=('total_quantity_sold', 'sum')
).reset_index()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(1, 10))
item_priority['normalized_score'] = scaler.fit_transform(item_priority[['total_weighted_score']])

item_priority['normalized_score'] = item_priority['normalized_score'].round(2)

item_priority = item_priority.sort_values(by='normalized_score', ascending=False)

print(item_priority.head())

item_priority = item_priority.sort_values(by='total_weighted_score', ascending=False)

print(item_priority.head())


# In[34]:


item_priority['percentile_rank'] = item_priority['total_weighted_score'].rank(pct=True)

item_priority['normalized_score'] = 1 + 9 * item_priority['percentile_rank']

item_priority['normalized_score'] = item_priority['normalized_score'].round(2)

print(item_priority[['part_code', 'total_weighted_score', 'normalized_score']].head())


# In[35]:


item_priority['decision'] = 'Discontinue'
item_priority.loc[item_priority['normalized_score'] >= 1, 'decision'] = 'Reduce'
item_priority.loc[item_priority['normalized_score'] >= 4, 'decision'] = 'Keep'

print(item_priority[['part_code', 'normalized_score', 'decision']].head())


# In[36]:


print(item_priority.head())

print("Min normalized_score:", item_priority['normalized_score'].min())
print("Max normalized_score:", item_priority['normalized_score'].max())

print(item_priority['decision'].value_counts())


# In[37]:


df['total_weighted_score'] = 0.0
df['decision'] = ''


# In[38]:


rc_1 = ['part_code']
rc_2 = ['part_code', 'normalized_score', 'decision']

df = pd.merge(df, item_priority[rc_2], on=['part_code'], how='left', suffixes=('_new', '_old'))
df = df.drop(columns=[col for col in df.columns if col.endswith('_new')])
df.columns = df.columns.str.replace('_old', '', regex=False)


# In[39]:


print("Min normalized_score:", df['normalized_score'].min())
print("Max normalized_score:", df['normalized_score'].max())


# In[40]:


df.to_csv('plv_v4.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




