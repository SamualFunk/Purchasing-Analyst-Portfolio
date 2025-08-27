#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import math


# In[2]:


pd.set_option('future.no_silent_downcasting', True)


# In[3]:


df_isi = pd.read_csv('items_sold_on_invoices_alt.txt', delimiter='\t', low_memory=False)
df_items = pd.read_csv('inventory_report_alt.txt', delimiter='\t', low_memory=False)
df_pos = pd.read_csv('active_pos_alt.txt', sep='\t')
df_cust = pd.read_csv('customer_list_check_alt.csv')
df_total = pd.read_csv('total_wine_items_alt.csv')
df_po = pd.read_csv('po_listing_alt.txt', delimiter='\t', low_memory=False)
df_wf = pd.read_excel('wf_items_alt.xlsx')
df_ven_oos = pd.read_csv('vendor_sku_info_alt.csv')
df_ms = pd.read_excel('monthly_specials_alt.xlsx')


# In[7]:


df_items.head()


# In[8]:


df_isi.columns = df_isi.columns.str.lower()


# In[9]:


df_isi.rename(columns={'part#': 'part_code'}, inplace=True)
df_isi = df_isi.dropna(subset=['part_code'])
df_items = df_items.dropna(subset=['part_code'])
df_isi['part_code'] = pd.to_numeric(df_isi['part_code'], errors='coerce')
df_isi['part_code'] = df_isi['part_code'].astype(float)
df_items['part_code'] = pd.to_numeric(df_items['part_code'], errors='coerce')
df_items['part_code'] = df_items['part_code'].astype(float)
df_items = df_items.dropna(subset=['date_created'])
df_wf['part_code'] = pd.to_numeric(df_wf['part_code'], errors='coerce')
df_wf['part_code'] = df_wf['part_code'].astype(float)
df_total['part_code'] = pd.to_numeric(df_total['part_code'], errors='coerce')
df_total['part_code'] = df_total['part_code'].astype(float)


# In[10]:


df_isi['invoice date'] = pd.to_datetime(df_isi['invoice date'], format='%m/%d/%y')

columns_to_clean = ['base price', 'sold for', 'extended price', 'cost', 'extended cost', 'total gross profit']

for col in columns_to_clean:
    df_isi[col] = df_isi[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)

df_isi[columns_to_clean] = df_isi[columns_to_clean].apply(pd.to_numeric, errors='coerce').round(2)


# In[11]:


na = ''
na_int = 0.0
df_isi = df_isi.assign(
    description=np.nan,
    category_name=na,
    sub_category_name=na,
    vendor_name=na,
    key_account=na,
)


# In[12]:


# ACCOUNT - BTG / KEY ACCOUNTS


# In[13]:


rc_df_isi = ['bill-to']
rc_df_cust = ['bill-to', 'key_account']

df_isi = pd.merge(df_isi, df_cust[rc_df_cust], on=['bill-to'], how='outer', suffixes=('_new', '_old'))
df_isi = df_isi.drop(columns=[col for col in df_isi.columns if col.endswith('_new')])
df_isi.columns = df_isi.columns.str.replace('_old', '', regex=False)


# In[14]:


#MERGE WITH ITEMS


# In[15]:


rc_df_isi = ['part_code']
rc_df_items = ['part_code', 'description', 'vendor_name', 'category_name', 'sub_category_name']

df_isi = pd.merge(df_isi, df_items[rc_df_items], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_isi = df_isi.drop(columns=[col for col in df_isi.columns if col.endswith('_new')])
df_isi.columns = df_isi.columns.str.replace('_old', '', regex=False)


# In[16]:


# Item Projections
# df_items = average movements
# df_previous = previous year quantity sold / accounts / averages


# In[17]:


df_items = df_items.assign(
    tier=na_int,
    case_count=na_int,
    average_movement_month=na_int,
    average_movement_two_month=na_int,
    average_movement_three_month=na_int,
    average_movement_six_month=na_int,
    average_movement_one_year=na_int,
    average_movement_month_ly=na_int,
    account_sprawl=na_int,
    average_orders_week=na_int,
    weeks_remaining=na_int,
    active_po=na,
    date_ordered=na,
    qty_on_order=na,
    key_account=na,
    is_whole_foods=na_int,
    wf_last_order_date=na,
    total_wine=na,
    whole_foods=na,
    status=na,
    start_date=na,
    end_date=na,
    vs_notes=na,
    order=na,
                )


# In[18]:


# Average Movements


# In[19]:


current_date = datetime.now()
end_date = current_date

#Month
start_date_one = current_date - timedelta(weeks=4)
isi_one_month = df_isi[(df_isi['invoice date'] >= start_date_one) & (df_isi['invoice date'] <= end_date)]
isi_one_month = isi_one_month.groupby('part_code').agg({'qty sold': 'sum'}).reset_index()
isi_one_month['average_movement_month'] = (isi_one_month['qty sold'] / 4).round(2)
#Two Month
start_date_two = current_date - timedelta(weeks=8)
isi_two_month = df_isi[(df_isi['invoice date'] >= start_date_two) & (df_isi['invoice date'] <= end_date)]
isi_two_month = isi_two_month.groupby('part_code').agg({'qty sold': 'sum'}).reset_index()
isi_two_month['average_movement_two_month'] = (isi_two_month['qty sold'] / 8).round(2)
#Three Month
start_date_three = current_date - timedelta(weeks=12)
isi_three_month = df_isi[(df_isi['invoice date'] >= start_date_three) & (df_isi['invoice date'] <= end_date)]
isi_three_month = isi_three_month.groupby('part_code').agg({'qty sold': 'sum'}).reset_index()
isi_three_month['average_movement_three_month'] = (isi_three_month['qty sold'] / 12).round(2)
#Six Month
start_date_six = current_date - timedelta(weeks=24)
isi_six_month = df_isi[(df_isi['invoice date'] >= start_date_six) & (df_isi['invoice date'] <= end_date)]
isi_six_month = isi_three_month.groupby('part_code').agg({'qty sold': 'sum'}).reset_index()
isi_six_month['average_movement_six_month'] = (isi_six_month['qty sold'] / 24).round(2)
#Year
start_date_year = current_date - timedelta(weeks=52)
isi_one_year = df_isi[(df_isi['invoice date'] >= start_date_year) & (df_isi['invoice date'] <= end_date)]
isi_one_year = isi_one_year.groupby('part_code').agg({'qty sold': 'sum'}).reset_index()
isi_one_year['average_movement_one_year'] = (isi_one_year['qty sold'] / 52).round(2)


#Last Year
start_date_ly = current_date - timedelta(days=365)
end_date_ly = start_date_ly + timedelta(weeks=4)
isi_ly = df_isi[(df_isi['invoice date'] >= start_date_ly) & (df_isi['invoice date'] <= end_date_ly)]
isi_ly = isi_ly.groupby('part_code').agg({'qty sold': 'sum'}).reset_index()
isi_ly['average_movement_month_ly'] = (isi_three_month['qty sold'] / 4).round(2)


#Orders
start_date_orders = current_date - timedelta(weeks=8)
isi_orders = df_isi[(df_isi['invoice date'] >= start_date_orders) & (df_isi['invoice date'] <= end_date)]
isi_orders = isi_orders.groupby('part_code').agg({'invoice#': 'count'}).reset_index()
isi_orders['average_orders_week'] = (isi_orders['invoice#'] / 8).round(2)
print(start_date_year)
print(end_date)


# In[20]:


#Quarterly Movements


# In[21]:


avg_isi = (
    isi_one_month[['part_code', 'average_movement_month']]
    .merge(isi_two_month[['part_code', 'average_movement_two_month']], on='part_code', how='outer')
    .merge(isi_three_month[['part_code', 'average_movement_three_month']], on='part_code', how='outer')
    .merge(isi_six_month[['part_code', 'average_movement_six_month']], on='part_code', how='outer')
    .merge(isi_one_year[['part_code', 'average_movement_one_year']], on='part_code', how='outer')
    .merge(isi_ly[['part_code', 'average_movement_month_ly']], on='part_code', how='outer')
    .merge(isi_orders[['part_code', 'average_orders_week']], on='part_code', how='outer')
)
avg_isi.fillna(0, inplace=True)


# In[22]:


avg_isi['rolling_average'] = avg_isi[
    ['average_movement_month', 'average_movement_two_month', 'average_movement_three_month']
].mean(axis=1).round(2)


# In[23]:


rc_df_items = ['part_code']
rc_avg_isi = ['part_code', 'average_movement_month', 'average_movement_two_month',
       'average_movement_three_month', 'average_movement_six_month', 'average_movement_one_year', 'average_movement_month_ly', 'average_orders_week', 'rolling_average']
df_items = pd.merge(df_items, avg_isi[rc_avg_isi], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[24]:


# ACCOUNT SPRAWL


# In[25]:


end_date = current_date
start_date = current_date - timedelta(weeks=16)

accounts = df_isi[(df_isi['invoice date'] >= start_date) & (df_isi['invoice date'] <= end_date)]
accounts = accounts[~accounts['salesperson'].str.contains('Bill Back', na=False)]


# In[26]:


accounts_mode = accounts.groupby('part_code')['qty sold'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
accounts_mode.rename(columns={'qty sold': 'most_common_qty_sold'}, inplace=True)
accounts_mode.loc[accounts_mode['most_common_qty_sold'] < 0, 'most_common_qty_sold'] = 0
accounts_max = accounts.groupby('part_code').agg({'qty sold': 'max', 'bill-to': 'last'}).reset_index()
accounts_max.rename(columns={'qty sold': 'most_qty_sold', 'bill-to': 'max_qty_account'}, inplace=True)
accounts_min = accounts.groupby('part_code')['qty sold'].agg('min').reset_index()
accounts_min.rename(columns={'qty sold': 'min'}, inplace=True)
accounts_sum = accounts.groupby('part_code').agg({'qty sold': 'sum'}).reset_index()
accounts_sum.rename(columns={'qty sold': 'sum_qty_sold'}, inplace=True)
accounts_max_cust = accounts.groupby(['part_code', 'bill-to']).agg({'qty sold':'max'}).reset_index()
accounts_max_cust.rename(columns={'bill-to': 'max_qty_account',
                                 'qty sold': 'most_qty_sold'}, inplace=True)


# In[27]:


print(accounts_max.columns)
print(accounts_max_cust.columns)


# In[28]:


rc_first = ['part_code', 'most_qty_sold']
rc_other = ['part_code', 'most_qty_sold', 'max_qty_account']

accounts_max = pd.merge(accounts_max, accounts_max_cust[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
accounts_max = accounts_max.drop(columns=[col for col in accounts_max.columns if col.endswith('_new')])
accounts_max.columns = accounts_max.columns.str.replace('_old', '', regex=False)


# In[29]:


accounts_max = accounts_max.sort_values(by=['part_code', 'most_qty_sold'], ascending=[True, False])
accounts_max = accounts_max.drop_duplicates(subset='part_code')


# In[30]:


accounts = accounts.groupby('part_code').agg({
    'qty sold': 'sum',
    'bill-to': 'count',
}).reset_index()
accounts.rename(columns={'bill-to': 'account_sprawl'}, inplace=True)



dfs = [accounts_mode, accounts_max, accounts_min, accounts_sum]
accounts_merged = reduce(lambda left, right: pd.merge(left, right, on='part_code', how='left'), dfs)


# In[31]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'qty sold', 'account_sprawl']

df_items = pd.merge(df_items, accounts[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[32]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'most_common_qty_sold', 'most_qty_sold', 'max_qty_account',
       'min', 'sum_qty_sold']

df_items = pd.merge(df_items, accounts_merged[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[ ]:





# In[33]:


# KEY ACCOUNT PART CODE GROUPING


# In[34]:


end_date = current_date
start_date = current_date - timedelta(weeks=12)

key_account_isi = df_isi[(df_isi['invoice date'] >= start_date) & (df_isi['invoice date'] <= end_date)]
key_account_isi = key_account_isi[~key_account_isi['salesperson'].str.contains('House Account', na=False)]


whole_foods_isi = df_isi[(df_isi['invoice date'] >= start_date) & (df_isi['invoice date'] <= end_date)]
whole_foods_isi = whole_foods_isi[~whole_foods_isi['salesperson'].str.contains('House Account', na=False)]
whole_foods_isi = whole_foods_isi[whole_foods_isi['bill-to'].str.contains('Whole Foods', case=False, na=False)]


# In[35]:


condition = whole_foods_isi['bill-to'].str.contains('Whole Foods', case=False, na=False)
whole_foods_isi['is_whole_foods'] = np.where(condition, 'Whole Foods', 'Whole Foods')


# In[36]:


whole_foods_grouped = whole_foods_isi.groupby(['part_code']).agg({'is_whole_foods': 'count', 'invoice date': 'last'}).reset_index()
whole_foods_grouped.rename(columns={'invoice date': 'wf_last_order_date'}, inplace=True)


# In[37]:


df_wf['whole_foods'] = 'WF'


# In[38]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'whole_foods']

df_items = pd.merge(df_items, df_wf[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[39]:


df_total['total_wine'] = 'TW'


# In[40]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'total_wine']

df_items = pd.merge(df_items, df_total[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[ ]:





# In[41]:


key_account_grouped = key_account_isi.groupby(['part_code']).agg({'key_account': 'count'}).reset_index()


# In[42]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'key_account']

df_items = pd.merge(df_items, key_account_grouped[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[43]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'is_whole_foods', 'wf_last_order_date']

df_items = pd.merge(df_items, whole_foods_grouped[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[44]:


df_items.to_csv('key_account_test.csv', index=False)


# In[45]:


#Active POs


# In[46]:


df_pos.columns = df_pos.columns.str.lower()
df_pos.rename(columns={'product#': 'part_code',
                    'product name': 'product_name',
                    'p.o. reference': 'active_po',
                    'date ordered': 'date_ordered',
                    'original order': 'qty_on_order'}, inplace=True)
df_pos.fillna({'date': '01/01/01'}, inplace=True)
df_pos.dropna(subset=['part_code'], inplace=True)


# In[47]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'active_po', 'date_ordered', 'qty_on_order']

df_items = pd.merge(df_items, df_pos[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[48]:


# TIERING


# In[49]:


df_items = df_items.dropna(subset=['product_name'])
df_items = df_items.fillna(0)


# In[50]:


tier_one = (df_items['rolling_average'] >= 12.0)
tier_two = (df_items['rolling_average'] >= 6.0) & (
        df_items['rolling_average'] <= 11.99)
tier_three = (df_items['rolling_average'] >= 3.0) & (
        df_items['rolling_average'] <= 5.99)
tier_four = (df_items['rolling_average'] >= 1.0) & (
        df_items['rolling_average'] <= 3.0)
tier_five = (df_items['rolling_average'] > 0.0) & (
        df_items['rolling_average'] <= 0.99)
tier_six = np.isclose(df_items['rolling_average'], 0.0)

df_items.loc[tier_one, 'tier'] = 1
df_items.loc[tier_two, 'tier'] = 2
df_items.loc[tier_three, 'tier'] = 3
df_items.loc[tier_four, 'tier'] = 4
df_items.loc[tier_five, 'tier'] = 5
df_items.loc[tier_six, 'tier'] = 6


# In[51]:


#WEEK PROJECTION


# In[52]:


condition = (df_items['category_name'].str.contains('Beer|Cider', na=False) & (df_items['description'] == 0))

df_items['description'] = np.where(condition, 1, df_items['description'])
df_items['description'] = pd.to_numeric(df_items['description'], errors='coerce')
df_items['description'] = df_items['description'].replace(0, 12)


# In[53]:


df_items['weeks_remaining_min'] = (df_items['total_onhand'] / (df_items['most_qty_sold'] * df_items['average_orders_week'])).round(2) 
df_items['weeks_remaining'] = (df_items['total_onhand'] / (df_items['most_common_qty_sold'] * df_items['average_orders_week'])).round(2)
df_items['weeks_remaining_avg'] = (df_items['total_onhand'] / (df_items['average_movement_month'])).round(2) 
df_items['total_onhand_cost'] = (df_items['total_onhand'] * df_items['weighted_cost']).round(2)


# In[54]:


df_items.loc[df_items['weeks_remaining'].isin([np.inf]), 'weeks_remaining_avg'] = 0
df_items.loc[df_items['weeks_remaining'].isin([-np.inf]), 'weeks_remaining_avg'] = -999


# In[55]:


weeks_trig = 16.0
condition = ((df_items['weeks_remaining'].between(0.00, weeks_trig)))

df_items['purchase_qty'] = np.where(
    condition,
    np.round(
        df_items['most_common_qty_sold'] *
        df_items['average_orders_week'] *
        (weeks_trig - (df_items['weeks_remaining']))
    ),
    np.nan
)
df_items['purchase_qty'] = np.ceil(df_items['purchase_qty'] / df_items['description']) * df_items['description']


# In[56]:


def calculate_priority(row, max_inventory, weights=(0.25, 0.25, 0.25, 0.25)):
    tier = row['average_orders_week']
    key_account = row['key_account']
    account_sprawl = row['account_sprawl']
    weeks_remaining = row['weeks_remaining_avg']

    normalized_tier = 10 * (tier)
    normalized_key_account = 10 * (key_account)
    normalized_account_sprawl = 10 * (account_sprawl)
    max_weeks = 16
    normalized_weeks = 10 * ((10 + weeks_remaining) / max_weeks)

    w1, w2, w3, w4 = weights
    priority_score = w1 * normalized_tier + w2 * normalized_key_account + w3 * normalized_account_sprawl + w4 * normalized_weeks

    max_score = w1 * 10 + w2 * 10 + w3 * 10 + w4 * 10
    scaled_priority_score = (priority_score / max_score) * 15

    return scaled_priority_score


# In[57]:


max_inventory = df_items['purchase_qty'].max()
df_items['priority_score'] = df_items.apply(calculate_priority, axis=1, max_inventory=max_inventory).round()


# In[58]:


#PREVIOUS PO 


# In[59]:


df_po['po_number'] = df_po['po_number'].fillna(method='ffill')
df_po['po_date'] = df_po['po_date'].fillna(method='ffill')
df_po['shipper'] = df_po['shipper'].fillna(method='ffill')
df_po['dealer'] = df_po['dealer'].fillna(method='ffill')


# In[60]:


df_po = df_po.groupby(['part_code', 'product_name']).agg({'po_number': 'last', 
                                                          'po_date': 'last',
                                                          'qty_ordered': 'last',
                                                         'dealer':'last',
                                                         'shipper':'last'}).reset_index()


# In[61]:


df_items['date_last_po'] = 0.0
df_items['last_po'] = 0.0
df_items['last_purchase_qty'] = 0.0
df_items['dealer'] = ''
df_items['shipper'] = ''


# In[62]:


df_po.rename(columns={'qty_ordered': 'last_purchase_qty',
                     'po_date': 'date_last_po',
                     'po_number': 'last_po'}, inplace=True)


# In[63]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'date_last_po', 'last_po', 'last_purchase_qty', 'dealer', 'shipper']

df_items = pd.merge(df_items, df_po[rc_other], on=['part_code'], how='outer', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[64]:


df_items = df_items.drop_duplicates(subset='part_code')


# In[65]:


df_items.to_csv('df_items_error.csv', index=False)


# In[66]:


len(df_items)


# In[67]:


df_items['start_date'] = pd.to_datetime(df_items['start_date'], format='%m/%d/%y')
df_items['end_date'] = pd.to_datetime(df_items['end_date'], format='%m/%d/%y')
df_items['date_created'] = pd.to_datetime(df_items['date_created'], format='%m/%d/%y')


# In[68]:


#END


# In[69]:


df_items['date_last_sold'] = df_items['date_last_sold'].replace('00/00/00', '01/01/01')
df_items['date_last_sold'] = df_items['date_last_sold'].replace(0, '01/01/01')
df_items['date_last_sold'] = pd.to_datetime(df_items['date_last_sold'], format='%m/%d/%y')
df_items['date_created'] = pd.to_datetime(df_items['date_created'], format='%m/%d/%y')
df_items = df_items[df_items['discontinued'] == False]
df_items = df_items.dropna(subset=['product_name'])
df_items = df_items.dropna(subset=['vendor_name'])
df_items['margin'] = 1 - (df_items['cost'] / df_items['price']).round(2)
df_items['weighted_margin'] = 1 - (df_items['weighted_cost'] / df_items['price']).round(2)
df_items['case_count'] = df_items['qty_on_order'] / df_items['description']
df_items['po_cost'] = df_items['qty_on_order'] * df_items['cost']
df_items = df_items.replace(-np.inf, -100)
df_items = df_items.replace(np.inf, -999)
df_items.loc[df_items['weeks_remaining'] < 0, 'priority_score'] = 0
df_items = df_items.fillna(0)
df_items = df_items.drop_duplicates(subset='part_code')


# In[70]:


#VENDOR OOS


# In[71]:


rc_df_ven_oos = ['part_code']
rc_other = ['part_code', 'status']

df_ven_oos = pd.merge(df_ven_oos, df_ms[rc_other], on=rc_df_ven_oos, how='outer', suffixes=('_new', '_old'))
df_ven_oos = df_ven_oos.drop(columns=[col for col in df_ven_oos.columns if col.endswith('_old')])
df_ven_oos.columns = df_ven_oos.columns.str.replace('_new', '', regex=False)


# In[73]:


rc_df_ven_oos = ['part_code']
rc_other = ['part_code', 'product_name', 'order']

df_ven_oos = pd.merge(df_ven_oos, df_items[rc_other], on=rc_df_ven_oos, how='outer', suffixes=('_new', '_old'))
df_ven_oos = df_ven_oos.drop(columns=[col for col in df_ven_oos.columns if col.endswith('_old')])
df_ven_oos.columns = df_ven_oos.columns.str.replace('_new', '', regex=False)


# In[74]:


rc_df_items = ['part_code']
rc_other = ['part_code', 'status', 'start_date', 'end_date', 'vs_notes']

df_items= pd.merge(df_items, df_ven_oos[rc_other], on=rc_df_items, how='inner', suffixes=('_new', '_old'))
df_items = df_items.drop(columns=[col for col in df_items.columns if col.endswith('_new')])
df_items.columns = df_items.columns.str.replace('_old', '', regex=False)


# In[75]:


condition = ((df_items['weeks_remaining'] == 0) & (df_items['key_account'] > 0))
df_items['priority_score'] = np.where(condition, 100, df_items['priority_score'])


# In[76]:


df_items = df_items[~df_items['category_name'].isin(['Keg Deposits', 'Depletion Allowance'])]


# In[77]:


columns = ['tier', 'vendor_code', 'vendor_name', 'priority_score', 'part_code', 'product_name', 'description', 'category_name',
       'sub_category_name', 'location', 'total_onhand', 'cost','weighted_cost', 'total_onhand_cost', 'price', 'margin', 'weighted_margin',
       'date_last_sold', 'status', 'start_date','end_date', 'date_created', 'vs_notes', 'active_po', 'date_ordered', 'qty_on_order', 'case_count', 'purchase_qty','po_cost', 'weeks_remaining', 'weeks_remaining_avg', 'weeks_remaining_min',
       'average_movement_month', 'average_movement_two_month',
       'average_movement_three_month', 'average_movement_six_month', 'average_movement_one_year','average_movement_month_ly', 'rolling_average', 'qty sold', 'average_orders_week',
       'account_sprawl', 'most_common_qty_sold', 'most_qty_sold',
       'max_qty_account', 'min', 'sum_qty_sold', 'key_account', 'is_whole_foods', 'wf_last_order_date', 'whole_foods', 'total_wine', 'date_last_po', 'last_po', 'last_purchase_qty', 'dealer', 'shipper', 'order']


# In[78]:


df_items = df_items[columns]


# In[79]:


columns = ['part_code',	'product_name',	'status','start_date', 'end_date', 'vs_notes']
df_ven_oos = df_ven_oos[columns]
df_ven_oos = df_ven_oos.drop_duplicates(subset='part_code')
df_ven_oos = df_ven_oos.dropna(subset=['product_name'])


# In[80]:


condition_1 = (df_items['status'].isna()) & (df_items['weeks_remaining'] < df_items['weeks_remaining_avg'])
df_items.loc[condition_1, 'status'] = '-'

condition_2 = (df_items['status'].isna()) & (df_items['weeks_remaining'] > df_items['weeks_remaining_avg'])
df_items.loc[condition_2, 'status'] = '+'


# In[81]:


df_items['start_date'] = df_items['start_date'].fillna(datetime.now())
df_items['end_date'] = df_items['end_date'].fillna(datetime.now())


# In[82]:


len(df_items)


# In[85]:


df_items.to_csv('product_listing_with_variables.csv', index=False)
df_ven_oos.to_csv('vendor_sku_info.csv', index=False)


# In[86]:


# SQL


# In[87]:


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


# In[88]:


df_items_name = "purchase_report_alt"


# In[89]:


df_items.to_sql(name=df_items_name, con=engine, index=False, if_exists='replace')
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




