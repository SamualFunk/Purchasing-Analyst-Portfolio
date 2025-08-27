#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import glob 
import os
import math


# In[109]:


items_sold = pd.read_csv('items_sold_on_invoices_alt.txt',  delimiter='\t', low_memory=False)
vendor_name = pd.read_csv('vendor_index_alt.csv')
products = pd.read_csv('product_listing_with_variables.csv')
result = pd.DataFrame()
da_total = pd.read_csv('da_total_info_alt.csv')


# In[110]:


rc_1 = ['part_code']
rc_2 = ['part_code', 'description']

da_total = pd.merge(da_total, products[rc_2], on=['part_code'], how='left', suffixes=('_new', '_old'))
da_total = da_total.drop(columns=[col for col in da_total.columns if col.endswith('_new')])
da_total.columns = da_total.columns.str.replace('_old', '', regex=False)


# In[111]:


da_total = da_total.dropna(subset=['description'])
da_total['da_type'] = 'none'
da_total['vintage'] = da_total['vintage'].fillna('na')


# In[112]:


columns = ['vendor_code', 'part_code', 'description', 'vintage', 'sell_price', 'da_qty', 'da_amount', 'da_type', 'date_active', 'expiration']


# In[113]:


da_total = da_total[columns]


# In[114]:


items_sold.columns = items_sold.columns.str.lower()
items_sold.rename(columns={'part#':'part_code',
                   'qty sold':'qty',
                          'bill-to': 'customer'}, inplace=True)
items_sold['part_code'] = pd.to_numeric(items_sold['part_code'], errors='coerce').fillna(0).astype(int)


# In[115]:


items_sold['invoice date'] = pd.to_datetime(items_sold['invoice date'], format='%m/%d/%y')

start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 6, 30)

items_sold = items_sold[items_sold['invoice date'] >= start_date]
items_sold = items_sold[items_sold['invoice date'] <= end_date]


# In[116]:


items_sold['sold for'] = items_sold['sold for'].str.replace('$', '')
items_sold['sold for'] = items_sold['sold for'].str.replace(',', '')
items_sold['sold for'] = pd.to_numeric(items_sold['sold for'], errors='coerce').astype('float64')


# In[117]:


for _, row in da_total.iterrows():
    condition_qty = row['da_qty']  
    part_code = row['part_code']  
    
    da = items_sold.loc[(items_sold['part_code'] == part_code) & (items_sold['qty'] >= condition_qty)]
    
    result = pd.concat([result, da])


# In[118]:


result = result.drop_duplicates()


# In[119]:


result['invoice_count'] = result['qty']
result.rename(columns={'product name': 'product_name'}, inplace=True)


# In[120]:


result = result.groupby(['invoice#', 'customer','part_code', 'product_name', 'invoice date', 'qty', 'sold for'], as_index=False).agg({'invoice_count': 'count'})


# In[121]:


result['da_amount'] = 0
result['description'] = 0
result['vendor_code'] = 0
result['vintage'] = ''
result['da_type'] = ''
result['sell_price'] = ''
result['date_active'] = ''
result['expiration'] = ''


# In[122]:


da_total.rename(columns={'da_qty': 'qty'}, inplace=True)


# In[123]:


rc_1 = ['part_code']
rc_2 = ['part_code','da_amount', 'da_type', 'description', 'sell_price', 'date_active', 'expiration']

merged = pd.merge(result, da_total[rc_2], on=['part_code'], how='outer', suffixes=('_new', '_old'))
merged = merged.drop(columns=[col for col in merged.columns if col.endswith('_new')])
merged.columns = merged.columns.str.replace('_old', '', regex=False)


# In[124]:


merged = pd.merge(merged, da_total[['part_code', 'vendor_code']], 
                         on=['part_code'], 
                         how='outer')
merged.drop(columns=['vendor_code_x'], inplace=True)
merged.rename(columns={'vendor_code_y': 'vendor_code'}, inplace=True)


# In[125]:


merged['vendor_name'] = ''
merged['start_date'] = start_date
merged['end_date'] = end_date


# In[126]:


merged = pd.merge(merged, da_total[['part_code', 'vendor_code']], 
                         on=['part_code'], 
                         how='outer')
merged.drop(columns=['vendor_code_x'], inplace=True)
merged.rename(columns={'vendor_code_y': 'vendor_code'}, inplace=True)


# In[127]:


merged['cases_sold'] = np.floor((merged['qty'] / merged['description']) * merged['invoice_count'])


# In[128]:


merged['da_total'] = merged['cases_sold'] * merged['da_amount']


# In[129]:


merged = merged.dropna(subset='invoice#')
merged = merged.drop_duplicates(subset=['invoice#', 'product_name'])


# In[130]:


merged = pd.merge(merged, vendor_name[['vendor_code', 'vendor_name']], 
                         on=['vendor_code'], 
                         how='left')
merged.drop(columns=['vendor_name_x'], inplace=True)
merged.rename(columns={'vendor_name_y': 'vendor_name'}, inplace=True)


# In[131]:


merged['da_hit'] = (merged['sold for'] <= merged['sell_price']).map({True: 'true', False: 'false'})
merged['active'] = ((merged['invoice date'] >= merged['date_active']) & (merged['invoice date'] <= merged['expiration'])).map({True: 'true', False: 'false'})


# In[132]:


columns = ['start_date', 'end_date', 'vendor_code', 'vendor_name', 'invoice date', 'invoice#', 'customer', 'part_code', 'product_name', 'vintage', 'qty', 'sell_price', 'sold for', 'invoice_count', 'cases_sold', 'da_amount', 'da_total',
          'date_active', 'expiration', 'da_hit', 'active']


# In[133]:


merged = merged[columns]


# In[134]:


merged.to_csv('all_das_da_total.csv', index=False)

