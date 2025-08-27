#!/usr/bin/env python
# coding: utf-8

# In[15]:


from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.disable(logging.CRITICAL)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df_isi = pd.read_csv('items_sold_on_invoices_alt.txt', delimiter='\t', low_memory=False)
plv = pd.read_csv('product_listing_with_variables.csv')


# In[16]:


df_isi.columns = df_isi.columns.str.lower()
df_isi.rename(columns={'part#': 'part_code'}, inplace=True)
df_isi = df_isi.dropna(subset=['part_code'])
df_isi['part_code'] = pd.to_numeric(df_isi['part_code'], errors='coerce')
df_isi = df_isi.dropna(subset=['part_code'])
df_isi['part_code'] = df_isi['part_code'].astype(int)
df_isi['invoice date'] = pd.to_datetime(df_isi['invoice date'])
df_isi = df_isi[df_isi['invoice date'].dt.year >= 2022]
agg_data = df_isi.groupby(['part_code', 'invoice date'])['qty sold'].sum().reset_index()


# In[17]:


agg_data = agg_data.set_index('invoice date')


# In[18]:


agg_data = (
    agg_data.groupby('part_code')
    .resample('W')['qty sold']
    .sum()
    .reset_index()
)
forecast_df = pd.DataFrame()


# In[19]:


for part_code in agg_data['part_code'].unique():
    part_code_data = agg_data[agg_data['part_code'] == part_code][['invoice date', 'qty sold']]
        
    if len(part_code_data) < 2:
        print(f"Skipping part_code {part_code} due to insufficient data")
        continue
        
    part_code_data.columns = ['ds', 'y']
    
    part_code_data['y'] = part_code_data['y'].fillna(0).replace([np.inf, -np.inf], 0)
        
    if len(part_code_data) < 2:
        print(f"Skipping part_code {part_code} due to insufficient data")
        continue

    part_code_data.columns = ['ds', 'y']
    
    part_code_data['y'] = part_code_data['y'].fillna(0).replace([np.inf, -np.inf], 0)
    
    model = Prophet()
    model.fit(part_code_data)
    
    future = model.make_future_dataframe(periods=52, freq='W')  
    
    if future.empty:
        print(f"Skipping part_code {part_code} because the future DataFrame is empty")
        continue
    
    future = future[future['ds'].dt.year == 2025]
    
    if future.empty:
        print(f"Skipping part_code {part_code} because no dates in 2025 were generated")
        continue
    
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat']]  
    
    forecast['part_code'] = part_code
    
    forecast_df = pd.concat([forecast_df, forecast])

forecast_df.rename(columns={'ds': 'date', 'yhat': 'forecasted_sales'}, inplace=True)

monthly_forecast = forecast_df.groupby(['part_code', pd.Grouper(key='date', freq='W')])['forecasted_sales'].sum().reset_index()

pivot_forecast = monthly_forecast.pivot(index='date', columns='part_code', values='forecasted_sales').fillna(0)


# In[20]:


monthly_forecast = monthly_forecast.round()
monthly_forecast['forecasted_sales'].clip(lower=0, inplace=True)


# In[21]:


na = ''
na_int = 0.0
monthly_forecast = monthly_forecast.assign(
    product_name=np,
    description=np.nan,
    vendor_code=na,
    vendor_name=na,
    category_name=na,
    sub_category_name=na,
    price=na,
)


# In[22]:


rc_df_isi = ['part_code']
rc_df_items = ['part_code', 'product_name', 'description', 'vendor_code', 'vendor_name', 'category_name', 'sub_category_name', 'price']

monthly_forecast = pd.merge(monthly_forecast, plv[rc_df_items], on=['part_code'], how='outer', suffixes=('_new', '_old'))
monthly_forecast = monthly_forecast.drop(columns=[col for col in monthly_forecast.columns if col.endswith('_new')])
monthly_forecast.columns = monthly_forecast.columns.str.replace('_old', '', regex=False)


# In[23]:


import mysql.connector
from sqlalchemy import create_engine

try:
    conn = mysql.connector.connect(
        host='', 
        user='',
        password='',
        database='',
        port=000
    )
    print("Connection successful!")
    conn.close()
except mysql.connector.Error as err:
    print(f"Error: {err}")

try:
    engine = create_engine('')
    print("SQLAlchemy connection successful!")
except Exception as e:
    print(f"SQLAlchemy Error: {e}")


# In[24]:


mf_name = "part_code_forecasting"


# In[25]:


monthly_forecast.to_sql(name=mf_name, con=engine, index=False, if_exists='replace')


conn.commit()
conn.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




