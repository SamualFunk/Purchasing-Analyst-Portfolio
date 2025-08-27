#!/usr/bin/env python
# coding: utf-8

# In[232]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# In[233]:


act_inv = pd.read_csv('inventory_report_alt.txt', sep='\t')
act_pos = pd.read_csv('active_pos_alt.txt', sep='\t')
invc_total = pd.read_csv('items_sold_on_invoices_alt.txt', sep='\t', low_memory=False)
up_inv = pd.read_csv('unposted_invoices.txt', sep='\t')


# In[234]:


#ACTIVE INVENTORY


# In[235]:


act_inv['up_qty'] = 0.0
act_inv['po_qty'] = 0.0
act_inv['real_qty'] = 0.0
act_inv['count'] = 0.0
act_inv['s_qty'] = 0.0
act_inv.rename(columns={'total_onhand': 'pims_qty'}, inplace=True)


# In[236]:


act_inv['part_code'] = pd.to_numeric(act_inv['part_code'], errors='coerce').astype('float')


# In[237]:


act_inv = act_inv[act_inv['discontinued'] == False]


# In[238]:


#ACTIVE UNPOSTED INVOICES


# In[239]:


up_inv['unposted'] = 'TRUE'
up_inv.rename(columns={'ORDER_NUMBER': 'invoice'}, inplace=True)


# In[240]:


invc_total.head(1)


# In[241]:


invc_total.rename(columns={'PART#': 'part_code',
                       'PRODUCT NAME': 'product_name',
                       'QTY SOLD': 'qty_sold',
                       'INVOICE#': 'invoice',
                        'INVOICE DATE': 'order_date'}, inplace=True)
invc_total['unposted'] = '' 

columns = ['invoice', 'order_date', 'part_code', 'product_name', 'qty_sold', 'unposted']
invc_total = invc_total[columns]
invc_total['invoice'] = pd.to_numeric(invc_total['invoice'], errors='coerce').astype('float')


# In[242]:


invc_total = pd.merge(invc_total, up_inv[['invoice', 'unposted']], on='invoice', how='outer',
                    suffixes=('_new', '_old'))
invc_total.drop(columns=['unposted_new'], inplace=True)
invc_total.rename(columns={'unposted_old': 'unposted'}, inplace=True)


# In[243]:


invc_total = invc_total.groupby(['part_code', 'unposted'], as_index=False).agg({'product_name': 'last', 'order_date': 'last', 'qty_sold': 'sum', 'invoice':'count'}).rename(
        columns={'qty_sold': 'up_qty'})
invc_total['part_code'] = pd.to_numeric(invc_total['part_code'], errors='coerce').astype('float')


# In[244]:


#ACTIVE POS


# In[245]:


act_pos.columns = act_pos.columns.str.lower()
act_pos.rename(columns={'product#': 'part_code',
                    'product name': 'product_name',
                    'p.o. reference': 'active_po',
                    'date ordered': 'date',
                    'original order': 'previous_purchase_qty'}, inplace=True)
act_pos.fillna({'date': '01/01/01'}, inplace=True)
act_pos.dropna(subset=['part_code'], inplace=True)  # Drop rows with missing 'part_code'


# In[246]:


act_pos = act_pos.groupby('part_code', as_index=False).agg({'product_name': 'last', 'previous_purchase_qty': 'sum', 'active_po':'count'}).rename(
        columns={'previous_purchase_qty': 'po_qty'})


# In[ ]:





# In[247]:


#UNPOSTED LISTING


# In[248]:


act_inv = pd.merge(act_inv, invc_total[['part_code', 'up_qty']], on='part_code', how='outer',
                    suffixes=('_new', '_old'))
act_inv.drop(columns=['up_qty_new'], inplace=True)
act_inv.rename(columns={'up_qty_old': 'up_qty'}, inplace=True)


# In[249]:


act_inv = pd.merge(act_inv, act_pos[['part_code', 'po_qty']], on='part_code', how='outer',
                    suffixes=('_new', '_old'))
act_inv.drop(columns=['po_qty_new'], inplace=True)
act_inv.rename(columns={'po_qty_old': 'po_qty'}, inplace=True)


# In[250]:


act_inv['pims_qty'] = act_inv['pims_qty'].fillna(0.0)
act_inv['up_qty'] = act_inv['up_qty'].fillna(0.0)
act_inv['po_qty'] = act_inv['po_qty'].fillna(0.0)
act_inv['s_qty'] = act_inv['s_qty'].fillna(0.0)


# In[251]:


act_inv['real_qty'] = act_inv['pims_qty'] - act_inv['up_qty']


# In[252]:


act_inv['count'] = 0.0
act_inv['datetime'] = datetime.now()


# In[253]:


act_inv = act_inv.dropna(subset=['product_name'])


# In[254]:


act_inv = act_inv.sort_values(by=['location', 'pims_qty'], ascending=True)
act_inv['location'] = act_inv['location'].fillna('N/A')


# In[255]:


col_adj = act_inv


# In[256]:


columns_1 = [ 'part_code', 'product_name', 'vendor_code', 'vendor_name', 'description', 'location', 'pims_qty', 'up_qty', 's_qty', 'po_qty', 'real_qty', 'count']
act_inv = act_inv[columns_1]


# In[257]:


col_adj['total_cost'] = col_adj['cost'] * col_adj['real_qty']


# In[258]:


columns_2 = [ 'part_code', 'product_name', 'weighted_cost', 'total_cost', 'price', 'vendor_code', 'vendor_name', 'description', 'location', 'pims_qty', 'up_qty', 'po_qty', 'real_qty', 'datetime', 'cost']
col_adj = col_adj[columns_2]


# In[259]:


act_inv.to_csv('inventory_audit.csv', index=False)


# In[260]:


#SQL


# In[261]:


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
    engine = create_engine('')
except Exception as e:
    print(f"SQLAlchemy Error: {e}")


# In[262]:


act_inv_name = "inventory_listing"


# In[263]:


act_inv.to_sql(name=act_inv_name, con=engine, index=False, if_exists='replace')


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





# In[ ]:





# In[ ]:





# In[ ]:




