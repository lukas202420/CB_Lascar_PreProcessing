# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:33:38 2025

@author: timot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns #install using either: pip install seaborn, or: conda install seaborn
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime

#import, make sure its in the same folder
BALD_dataframe = pd.read_csv("GBEX_Raw_BALD.csv")

time = BALD_dataframe['Time']
temp=BALD_dataframe['T']

BALD_dataframe['Time_fixed'] = pd.to_datetime(BALD_dataframe['Time'])

# only take a data point every 50, to speed up plotting by a lot
downsampled_df = BALD_dataframe.iloc[::50].copy()

# creating figure
plt.figure(figsize=(12, 6))
plt.plot(downsampled_df['Time_fixed'], downsampled_df['T'])

plt.title('Temperature over Time')
plt.xlabel('Time')
plt.ylabel('Temperature values')

ax = plt.gca()

# custom tick dates
custom_tick_dates = [
    datetime(2008, 1, 1),
    datetime(2009, 1, 1),
    datetime(2010, 1, 1),
    datetime(2011, 1, 1),
    datetime(2012, 1, 1),
    datetime(2013, 1, 1),
    datetime(2014, 1, 1),
    datetime(2015, 1, 1),
    datetime(2016, 1, 1),
    datetime(2017, 1, 1),
    datetime(2018, 1, 1),
    datetime(2019, 1, 1),
    datetime(2020, 1, 1),
    datetime(2021, 1, 1),
    datetime(2022, 1, 1),
    datetime(2023, 1, 1),
    
]

ax.set_xticks(custom_tick_dates)

# format ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# show plot
plt.gcf().autofmt_xdate()
plt.show()