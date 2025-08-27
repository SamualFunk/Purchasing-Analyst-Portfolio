#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
from sqlalchemy import create_engine
import mysql.connector
from dateutil.relativedelta import relativedelta


# In[21]:


df_isi = pd.read_csv('items_sold_on_invoices_alt.txt', delimiter='\t', low_memory=False)


# In[22]:


df_isi.rename(columns={'PART#': 'part_code',
                    'PRODUCT NAME': 'product_name',
                    'QTY SOLD': 'qty_sold',
                    'VENDOR': 'vendor_code',
                    'INVOICE#': 'invoice_number',
                    'INVOICE DATE': 'invoice_date',
                    'SALESPERSON': 'salesperson',
                    'SOLD FOR': 'sold_for',
                    'BILL-TO': 'customer',
                    'BASE PRICE': 'base_price',
                    'EXTENDED PRICE': 'extended_price',
                    'EXTENDED COST': 'extended_cost',
                    'TOTAL GROSS PROFIT': 'total_gross_profit',
                    'MARGIN%': 'margin'}, inplace=True)


# In[24]:


df_isi['part_code'] = df_isi['part_code'].replace(to_replace=["DA"],
           value=99999)
df_isi['part_code'] = df_isi['part_code'].fillna(9999)


# In[25]:


df_isi.columns = df_isi.columns.str.lower()


# In[26]:


df_isi.fillna({'part_code': 99999}, inplace=True)
df_isi['part_code'] = pd.to_numeric(df_isi['part_code'], errors='coerce')
df_isi['part_code'] = df_isi['part_code'].astype(float)


# In[27]:


df_isi['invoice_date'] = pd.to_datetime(df_isi['invoice_date'], format='%m/%d/%y')


# In[28]:


df_isi = df_isi.sort_values(by='invoice_date', ascending=False)


# In[29]:


cols_to_clean = [
    'extended_price', 'sold_for', 'cost', 'extended_cost', 
    'total_gross_profit', 'base_price', 
]

df_isi[cols_to_clean] = df_isi[cols_to_clean].replace({r'\$|,': ''}, regex=True)


# In[30]:


df_isi['extended_price'] = pd.to_numeric(df_isi['extended_price'], errors='coerce').astype('float64')
df_isi['sold_for'] = pd.to_numeric(df_isi['sold_for'], errors='coerce').astype('float64')
df_isi['cost'] = pd.to_numeric(df_isi['cost'], errors='coerce').astype('float64')
df_isi['extended_cost'] = pd.to_numeric(df_isi['extended_cost'], errors='coerce').astype('float64')
df_isi['total_gross_profit'] = pd.to_numeric(df_isi['total_gross_profit'], errors='coerce').astype('float64')
df_isi['base_price'] = pd.to_numeric(df_isi['base_price'], errors='coerce').astype('float64')
df_isi['status'] = ''


# In[31]:


#TESTING


# In[32]:


df_grouped = df_isi.groupby('customer').agg({'salesperson': 'first'}).reset_index()


# In[35]:


current_salespeople = df_grouped

df_isi = pd.merge(
    df_isi,
    current_salespeople,
    on='customer', 
    suffixes=('_new', '_old')
)
df_isi = df_isi.drop(columns=[col for col in df_isi.columns if col.endswith('_new')])
df_isi.columns = df_isi.columns.str.replace('_old', '', regex=False)


# In[36]:


print(df_isi.columns)


# In[39]:


import mysql.connector
from sqlalchemy import create_engine

try:
    conn = mysql.connector.connect(
        host='',  
        user='',
        password='',
        database='',
        port=0000
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


# In[40]:


df_isi_name = "sales_invoice"

df_isi.to_sql(name=df_isi_name, con=engine, index=False, if_exists='replace')


conn.commit()
conn.close()


# In[ ]:





# In[42]:


df = df_isi

current_date = datetime(2025, 8, 6)
previous_year = datetime(current_date.year - 1, 1, 1)
previous_year_current_date = current_date - relativedelta(years=1)
previous_year_end = datetime(current_date.year - 1, 12, 31)
year_to_date_start = datetime(current_date.year, 1, 1)
current_month = datetime(current_date.year, current_date.month, 1)
previous_month_year = datetime(current_date.year, current_date.month, 1) - relativedelta(years=1)
next_month = datetime(current_date.year + 1, 1, 1)
previous_month_end = next_month.replace(day=1) - relativedelta(days=1)
previous_month_year_end = previous_month_end.replace(year=current_date.year - 1)
df = df.fillna(0)


# In[43]:


ytd = df[(df['invoice_date'] >= year_to_date_start) & (df['invoice_date'] <= current_date)]

lytd = df[(df['invoice_date'] >= previous_year) & (df['invoice_date'] <= previous_year_current_date)]

ytd = ytd.groupby('customer')[['extended_price', 'invoice_date']].agg(
    {'extended_price': 'sum', 'invoice_date': 'max'}).reset_index()

ytd.rename(columns={'extended_price': 'year_to_date'}, inplace=True)

lytd = lytd.groupby('customer')[['extended_price', 'invoice_date']].agg(
    {'extended_price': 'sum', 'invoice_date': 'max'}).reset_index()

lytd.rename(columns={'extended_price': 'last_year_to_date'}, inplace=True)

lytotal = df[(df['invoice_date'] >= previous_year) & (df['invoice_date'] <= previous_year_end)]

lytotal = lytotal.groupby('customer')[['extended_price', 'invoice_date']].agg(
    {'extended_price': 'sum', 'invoice_date': 'max'}).reset_index()

lytotal.rename(columns={'extended_price': 'last_year_total'}, inplace=True)

curmonth = df[(df['invoice_date'] >= current_month) & (df['invoice_date'] <= current_date)]

curmonth = curmonth.groupby('customer')[['extended_price', 'invoice_date']].agg(
    {'extended_price': 'sum', 'invoice_date': 'max'}).reset_index()

curmonth.rename(columns={'extended_price': 'current_month'}, inplace=True)

lymonth = df[(df['invoice_date'] >= previous_month_year) & (df['invoice_date'] <= previous_year_current_date)]
lymonth.drop_duplicates()

lymonth = lymonth.groupby('customer')[['extended_price', 'invoice_date']].agg(
    {'extended_price': 'sum', 'invoice_date': 'max'}).reset_index()

lymonth.rename(columns={'extended_price': 'month_previous_year'}, inplace=True)

df = df.groupby('customer')[['extended_price', 'invoice_date', 'salesperson']].agg(
    {'extended_price': 'sum', 'invoice_date': 'max', 'salesperson': 'max'}).reset_index()


# In[44]:


cols_to_initialize = [
    'year_to_date', 'last_year_to_date', 'last_year_total', 'current_month', 
    'previous_month_year', 'last_month_to_date', 'month_previous_year', 
    'previous_year_month_total'
]

df[cols_to_initialize] = 0.0


# In[45]:


dataframes = {
    'year_to_date': ytd,
    'last_year_to_date': lytd,
    'last_year_total': lytotal,
    'current_month': curmonth,
    'month_previous_year': lymonth
}

for col, df_merge in dataframes.items():
    df = df.merge(df_merge[['customer', col]], on='customer', how='left', suffixes=('_new', '_old'))
    df.drop(columns=[f'{col}_new'], inplace=True)
    df.rename(columns={f'{col}_old': col}, inplace=True)

df.drop_duplicates(inplace=True)


# In[46]:


df.fillna(0, inplace=True)

df['current_month'] = df['current_month'].astype(float).round(2)
df['month_previous_year'] = df['month_previous_year'].astype(float).round(2)
df['difference_ytd'] = df['year_to_date'] - df['last_year_to_date']
df['difference_month'] = (df['current_month'] - df['month_previous_year'])
df = df.drop_duplicates()


# In[47]:


import mysql.connector
from sqlalchemy import create_engine

try:
    conn = mysql.connector.connect(
        host='',  
        user='',
        password='',
        database='',
        port=0000
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


# In[48]:


df_name = "sales_dataframe"

df.to_sql(name=df_name, con=engine, index=False, if_exists='replace')


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




