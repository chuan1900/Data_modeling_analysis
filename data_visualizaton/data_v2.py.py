
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt


# In[2]:

import zipfile


# In[3]:

df = pd.read_csv("../input/train.csv")


# In[4]:

df["Dates"] = pd.to_datetime(df["Dates"])


# Print to show features in the dataset

# In[5]:

print(df.columns)


# print to show the labeled categories in dataset

# In[6]:

print(df["Category"].unique())


# Then plot to show the number of each categories

# In[39]:

#group each category and then plot to show the number of each category
categories = df.groupby("Category")["Category"].count()
categories = categories.sort_values(ascending=0)
plt.figure()
categories.plot(kind='barh', title="Count of Category",
                    fontsize=8,
                    figsize=(12,8),
                    stacked=False,
                    width=1,
                    color='#468499')
plt.grid()
plt.savefig("category_count.png")
print(categories)


# From the figure shown above, number of LARCENY/THEFT is the largest. 

# In[40]:

# add time attributes to df
df['Year'] = df['Dates'].map(lambda x: x.year)
df['Week'] = df['Dates'].map(lambda x: x.week)
df['Hour'] = df['Dates'].map(lambda x: x.hour)
df['Day'] = df['Dates'].map(lambda x: x.day)


# In[41]:

print(df.head())


# plot to show relationship between crime number and sf districts

# In[36]:

df.PdDistrict.value_counts().plot(kind='barh', title="Count of District",
                                  color = '#468499')
plt.grid()
plt.savefig('district_counts.png')


# In[51]:

df['event']=1
daily_district_events = df[['PdDistrict','Day','event']].groupby(['PdDistrict','Day']).count().reset_index()
daily_district_events_pivot = daily_district_events.pivot(index='Day', columns='PdDistrict', values='event').fillna(method='ffill')
daily_district_events_pivot.interpolate().plot.area(title='number of cases daily by district', figsize=(10,6),colormap="jet")
plt.savefig('daily_events_by_district.png')


# In[50]:

hourly_district_events = df[['PdDistrict','Hour','event']].groupby(['PdDistrict','Hour']).count().reset_index()
hourly_district_events_pivot = hourly_district_events.pivot(index='Hour', columns='PdDistrict', values='event').fillna(method='ffill')
hourly_district_events_pivot.interpolate().plot.area(title='number of cases hourly by district', figsize=(10,6),colormap="jet")
plt.savefig('hourly_events_by_district.png')


# In[ ]:



