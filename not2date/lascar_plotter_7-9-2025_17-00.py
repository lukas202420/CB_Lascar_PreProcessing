# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:23:13 2025
Last update on Wed Jul 9 17:00 2025

@author: Timothee Briand & Lukas Lehner
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime

###############################################################################

### USER INPUTS

# Insert file path here (if '\' then [r'file_path'], if '/' then remove 'r')

file_path = r''

# Set all the variables below to match the labels of each column in the .csv
time_name = ''
temp_name = ''
rh_name = ''
td_name = ''

# Set the labels of the corrected variables that you want to create.
rh_cor_name = ''
td_cor_name = ''


###############################################################################

### INITIALIZATION

# Prints selected file name
print("Selected file:", file_path)

# Creates dataframe from .csv
dataframe = pd.read_csv(file_path)

# Skips first values as they may have been taken & tainted during installation
dataframe = dataframe.iloc[5:].reset_index(drop=True)

# Changes time string to datetime type
dataframe['Time_fixed'] = pd.to_datetime(dataframe[time_name])


## RH and Td correction

# Creating lists to insert corrected RH and Td values
RH_cor = [None] * len(dataframe)
Td_cor = [None] * len(dataframe)

# Iterating through rows to update RH values out of 0-100% range and Td values.
for i in range(0,len(dataframe)):
    if dataframe.loc[i, rh_name] > 100:
        RH_cor[i] = 100 # RH is adjusted to 100% as it must be saturated
        Td_cor[i] = dataframe.loc[i, temp_name] # Td is equal to T
    
    else: # Everything stays the same
        RH_cor[i] = dataframe.loc[i, rh_name]
        Td_cor[i] = dataframe.loc[i, td_name]
        
# Creating columns for the lists to merge into the dataframe
dataframe[rh_cor_name] = RH_cor
dataframe[td_cor_name] = Td_cor


## Initializing daily and monthly temperature averages for plotting

# Create index using the time column
dataframe = dataframe.sort_values('Time_fixed')
dataframe = dataframe.set_index('Time_fixed', drop=False)

# Resample by day and calculate daily min, max, avg for temperature
daily_summary_T = dataframe[temp_name].resample('1D')\
    .agg(['mean', 'min', 'max']).dropna().reset_index()
daily_summary_T.columns = ['Date', 'T_avg', 'T_min', 'T_max']

daily_summary_T['DateTime'] = pd.to_datetime(daily_summary_T['Date'])
daily_summary_T = daily_summary_T.set_index('Date')
monthly_summary_T = daily_summary_T.resample('ME')\
    .agg({'T_avg': 'mean','T_min': 'min','T_max': 'max'})\
        .dropna().reset_index()
        
        
## Initializing daily and monthly relative humidity averages for plotting
        
# Resample by day and calculate daily min, max, avg for RH
daily_summary_rh = dataframe[rh_cor_name].resample('1D')\
    .agg(['mean', 'min', 'max']).dropna().reset_index()
daily_summary_rh.columns = ['DateTime', 'RH_avg', 'RH_min', 'RH_max']

# Resample by day and calculate daily min, max, avg for RH
daily_summary_rh = dataframe[rh_cor_name].resample('1D')\
    .agg(['mean', 'min', 'max']).dropna().reset_index()
daily_summary_rh.columns = ['DateTime', 'RH_avg', 'RH_min', 'RH_max']

daily_summary_rh['DateTime'] = pd.to_datetime(daily_summary_rh['DateTime'])
daily_summary_rh = daily_summary_rh.set_index('DateTime', drop=False)
monthly_summary_rh = daily_summary_rh.resample('ME')\
    .agg({'RH_avg': 'mean','RH_min': 'min','RH_max': 'max'})\
        .dropna().reset_index()


###############################################################################

### FUNCTIONS

# To make all plots in one go
def plot_all():
    plot_temp()
    plot_rh()
    plot_daily_temp()
    plot_daily_rh()
    plot_monthly_temp()
    plot_monthly_rh()

## Continuous

# Continuous Temperature Plot
def plot_temp():
    # only take a data point every 50, to speed up plotting by a lot, can use different number
    downsampled_df = dataframe.iloc[::50].copy()
    
    # Creating figure
    plt.figure(figsize=(12, 6))
    plt.plot(downsampled_df['Time_fixed'], downsampled_df[temp_name])
    
    plt.title('Temperature over Time')
    plt.xlabel(time_name)
    plt.ylabel('Temperature values')
    
    ax = plt.gca()
    
    # Custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1), datetime(2007, 1, 1), datetime(2008, 1, 1),
        datetime(2009, 1, 1), datetime(2010, 1, 1), datetime(2011, 1, 1),
        datetime(2012, 1, 1), datetime(2013, 1, 1), datetime(2014, 1, 1),
        datetime(2015, 1, 1), datetime(2016, 1, 1), datetime(2017, 1, 1),
        datetime(2018, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1),
        datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Plotting
    plt.gcf().autofmt_xdate()
    plt.show()
    
# Continuous Relative Humidity Plot
def plot_rh():
    # only take a data point every 50, to speed up plotting by a lot, can use different number
    downsampled_df = dataframe.iloc[::50].copy()
    
    # Creating figure
    plt.figure(figsize=(12, 6))
    plt.plot(downsampled_df['Time_fixed'], downsampled_df[rh_cor_name])
    
    plt.title('Relative Humidity over Time')
    plt.xlabel(time_name)
    plt.ylabel('RH values')
    
    ax = plt.gca()
    
    # Custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1), datetime(2007, 1, 1), datetime(2008, 1, 1),
        datetime(2009, 1, 1), datetime(2010, 1, 1), datetime(2011, 1, 1),
        datetime(2012, 1, 1), datetime(2013, 1, 1), datetime(2014, 1, 1),
        datetime(2015, 1, 1), datetime(2016, 1, 1), datetime(2017, 1, 1),
        datetime(2018, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1),
        datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Plotting
    plt.gcf().autofmt_xdate()
    plt.show()
  
## Daily Averaged

# Daily Temperature Plot
def plot_daily_temp():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_T['DateTime'].diff()
    
    # Gap is more than 1 day
    gaps = daily_summary_T[date_diffs > pd.Timedelta(days=1)]

    # Monthly, only take a data point every 30 days/datapoints
    downsampled_daily_summary_T = daily_summary_T.iloc[::30].copy()
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(
        downsampled_daily_summary_T['DateTime'], 
        downsampled_daily_summary_T['T_avg'], 
        label='Average Temp', 
        color='black', 
        linewidth=2
        )
    plt.plot(
        downsampled_daily_summary_T['DateTime'], 
        downsampled_daily_summary_T['T_min'], 
        label='Min Temp', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
        )
    plt.plot(
        downsampled_daily_summary_T['DateTime'], 
        downsampled_daily_summary_T['T_max'], 
        label='Max Temp', 
        color='red', 
        linestyle='dashdot', 
        linewidth=0.5
        )
    
    plt.title('Daily Temperature Summary:')
    plt.xlabel('DateTime')
    plt.ylabel('Temperature')
    
    ax = plt.gca()
    
    # Custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1), datetime(2007, 1, 1), datetime(2008, 1, 1),
        datetime(2009, 1, 1), datetime(2010, 1, 1), datetime(2011, 1, 1),
        datetime(2012, 1, 1), datetime(2013, 1, 1), datetime(2014, 1, 1),
        datetime(2015, 1, 1), datetime(2016, 1, 1), datetime(2017, 1, 1),
        datetime(2018, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1),
        datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()
    
    # Shade missing data regions
    for i, row in gaps.iterrows():
        gap_idx = daily_summary_T.index.get_loc(row.name)
        gap_start = daily_summary_T.iloc[gap_idx - 1]['DateTime']
        gap_end = row['DateTime']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)
    
    plt.legend()
    plt.show()

# Daily Relative Humidity Plot
def plot_daily_rh():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_rh['DateTime'].diff()
    
    # Gap is more than 1 day
    gaps = daily_summary_rh[date_diffs > pd.Timedelta(days=1)]

    # Monthly, only take a data point every 30 days/datapoints
    downsampled_daily_summary_rh = daily_summary_rh.iloc[::30].copy()
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(
        downsampled_daily_summary_rh['DateTime'], 
        downsampled_daily_summary_rh['RH_avg'], 
        label='Average RH', 
        color='black', 
        linewidth=2.5
        )
    plt.plot(
        downsampled_daily_summary_rh['DateTime'], 
        downsampled_daily_summary_rh['RH_min'], 
        label='Min RH', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
        )
    plt.plot(downsampled_daily_summary_rh['DateTime'], 
             downsampled_daily_summary_rh['RH_max'], 
             label='Max RH', 
             color='red', 
             linestyle='dashdot', 
             linewidth=0.5
             )
    plt.scatter(downsampled_daily_summary_rh['DateTime'], 
                downsampled_daily_summary_rh['RH_avg'],  
                color='black', 
                s=10,  
                zorder=5
                )
    
    plt.title('Daily Relative Humidity Summary:')
    plt.xlabel('DateTime')
    plt.ylabel('Relative Humidity')
    
    ax = plt.gca()
    
    # Custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1), datetime(2007, 1, 1), datetime(2008, 1, 1),
        datetime(2009, 1, 1), datetime(2010, 1, 1), datetime(2011, 1, 1),
        datetime(2012, 1, 1), datetime(2013, 1, 1), datetime(2014, 1, 1),
        datetime(2015, 1, 1), datetime(2016, 1, 1), datetime(2017, 1, 1),
        datetime(2018, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1),
        datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()
    
    # Shade missing data regions    
    for i, row in gaps.iterrows():
        gap_idx = daily_summary_rh.index.get_loc(row.name)
        gap_start = daily_summary_rh.iloc[gap_idx - 1]['DateTime']
        gap_end = row['DateTime']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)
        
    plt.legend()
    plt.show()
    
## Monthly Averaged    
    
# Monthly Temperature Plot
def plot_monthly_temp():
    # Calculate time differences between consecutive months
    date_diffs = monthly_summary_T['Date'].diff()

    # Identify gaps larger than 1 month
    gaps = monthly_summary_T[date_diffs > pd.Timedelta(days=31)]

    plt.figure(figsize=(12, 6))

    # Plot monthly average, min, and max
    plt.plot(
        monthly_summary_T['Date'], 
        monthly_summary_T['T_avg'], 
        label='Average Temp', 
        color='black', 
        linewidth=2.5
    )
    plt.plot(
        monthly_summary_T['Date'], 
        monthly_summary_T['T_min'], 
        label='Min Temp', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
    )
    plt.plot(
        monthly_summary_T['Date'], 
        monthly_summary_T['T_max'], 
        label='Max Temp', 
        color='red', 
        linestyle='dashdot', 
        linewidth=0.5
    )

    plt.title('Monthly Temperature Summary:')
    plt.xlabel('Date')
    plt.ylabel('Temperature')

    ax = plt.gca()

    # Custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1), datetime(2007, 1, 1), datetime(2008, 1, 1),
        datetime(2009, 1, 1), datetime(2010, 1, 1), datetime(2011, 1, 1),
        datetime(2012, 1, 1), datetime(2013, 1, 1), datetime(2014, 1, 1),
        datetime(2015, 1, 1), datetime(2016, 1, 1), datetime(2017, 1, 1),
        datetime(2018, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1),
        datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.gcf().autofmt_xdate()

    # Shade missing data regions
    for i, row in gaps.iterrows():
        gap_start = monthly_summary_T.loc[i - 1, 'Date']
        gap_end = row['Date']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()

# Monthly Relative Humidity Plot
def plot_monthly_rh():
    # Calculate time differences between consecutive months
    date_diffs = monthly_summary_rh['DateTime'].diff()

    # Identify gaps larger than 1 month
    gaps = monthly_summary_rh[date_diffs > pd.Timedelta(days=31)]

    plt.figure(figsize=(12, 6))

    # Plot monthly average, min, and max RH
    plt.plot(
        monthly_summary_rh['DateTime'], 
        monthly_summary_rh['RH_avg'], 
        label='Average RH', 
        color='black', 
        linewidth=2.5
    )
    plt.plot(
        monthly_summary_rh['DateTime'], 
        monthly_summary_rh['RH_min'], 
        label='Min RH', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
    )
    plt.plot(
        monthly_summary_rh['DateTime'], 
        monthly_summary_rh['RH_max'], 
        label='Max RH', 
        color='red', 
        linestyle='dashdot', 
        linewidth=0.5
    )

    plt.title('Monthly Relative Humidity Summary:')
    plt.xlabel('DateTime')
    plt.ylabel('Relative Humidity')

    ax = plt.gca()

    # Custom tick dates (same as before)
    custom_tick_dates = [
        datetime(2006, 1, 1), datetime(2007, 1, 1), datetime(2008, 1, 1),
        datetime(2009, 1, 1), datetime(2010, 1, 1), datetime(2011, 1, 1),
        datetime(2012, 1, 1), datetime(2013, 1, 1), datetime(2014, 1, 1),
        datetime(2015, 1, 1), datetime(2016, 1, 1), datetime(2017, 1, 1),
        datetime(2018, 1, 1), datetime(2019, 1, 1), datetime(2020, 1, 1),
        datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    ]
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.gcf().autofmt_xdate()

    # Shade missing data regions
    for i, row in gaps.iterrows():
        gap_start = monthly_summary_rh.loc[i - 1, 'DateTime']
        gap_end = row['DateTime']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()
