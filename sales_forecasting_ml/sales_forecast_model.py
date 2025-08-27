#!/usr/bin/env python
# coding: utf-8

# In[4]:


import plotly.graph_objects as go 
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt

df_isi = pd.read_csv('items_sold_on_invoices_alt.txt', delimiter='\t', low_memory=False)

df_isi.columns = df_isi.columns.str.lower()
df_isi.rename(columns={'part#': 'part_code'}, inplace=True)
df_isi = df_isi.dropna(subset=['part_code'])
df_isi['part_code'] = pd.to_numeric(df_isi['part_code'], errors='coerce')
df_isi['part_code'] = df_isi['part_code'].astype(float)
df_isi['invoice date'] = pd.to_datetime(df_isi['invoice date'])

numeric_cols = ['base price', 'sold for', 'extended price', 'cost', 'extended cost', 'total gross profit']

def clean_and_convert(col):
    return pd.to_numeric(col.replace('[\$,]', '', regex=True), errors='coerce')

for col in numeric_cols:
    df_isi[col] = clean_and_convert(df_isi[col])

df_isi['invoice date'] = pd.to_datetime(df_isi['invoice date'])

weekly_sales = df_isi.resample('W', on='invoice date')['extended price'].sum().reset_index()
weekly_sales.columns = ['ds', 'y'] 


# In[5]:


weekly_sales.to_csv('weekly_sales.csv', index=False)


# In[6]:


model = Prophet()
model.fit(weekly_sales)

future = model.make_future_dataframe(periods=52, freq='W')  


forecast = model.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))


# In[7]:


fig1 = model.plot_components(forecast)


# In[8]:


forecast.to_csv('forecast.csv', index=False)


# In[9]:


forecast_2025 = forecast[forecast['ds'].dt.year == 2025][['ds', 'yhat']]
forecast_2025.columns = ['Weeks', 'Predicted_Sales']

print(forecast_2025)


# In[10]:


forecast_2025.to_csv('2025_sales_forecast.csv', index=False)


# In[11]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(weekly_sales['y'], forecast['yhat'][:-52])
print(f"Mean Absolute Error: {mae}")


# In[34]:


import mysql.connector
from sqlalchemy import create_engine

try:
    conn = mysql.connector.connect(
        host='',  
        user='',
        password='',
        database='',
        port=3306
    )
    print("Connection successful!")
    conn.close()
except mysql.connector.Error as err:
    print(f"Error: {err}")

try:
    engine = create_engine('mysql+mysqlconnector://root:toor@localhost:3306/purchase_reports')
    print("SQLAlchemy connection successful!")
except Exception as e:
    print(f"SQLAlchemy Error: {e}")


# In[35]:


mf_name = "sales_forecasting_2025"


# In[36]:


import pandas as pd
from sqlalchemy import create_engine
import mysql.connector

conn = engine.connect()

try:
    forecast_2025.to_sql(name=mf_name, con=engine, index=False, if_exists='replace')
    print("Data successfully written to database")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()


# In[ ]:





# In[ ]:





# In[ ]:




