# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:13:47 2025

@author: timot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import os

# create a root window and hide it
root = tk.Tk()
root.withdraw()

# open file dialog, get the selected file path
file_path1 = filedialog.askopenfilename()

name_of_file=os.path.basename(file_path1)
name_of_file_no_csv_1=os.path.splitext(name_of_file)[0]

#creates dataframe from file selection
dataframe1 = pd.read_csv(file_path1)

# adjust the number of rows to skip, so that you start at 00:00 and with believable values
dataframe1 = dataframe1.iloc[50:].reset_index(drop=True)

dataframe1['Time_fixed'] = pd.to_datetime(dataframe1['Time'])


# open file dialog, get the selected file path
file_path2 = filedialog.askopenfilename()

name_of_file=os.path.basename(file_path2)
name_of_file_no_csv_2=os.path.splitext(name_of_file)[0]

#creates dataframe from file selection
dataframe2 = pd.read_csv(file_path2)

# adjust the number of rows to skip, so that you start at 00:00 and with believable values
dataframe2 = dataframe2.iloc[50:].reset_index(drop=True)

dataframe2['Time_fixed'] = pd.to_datetime(dataframe2['Time'])


# prints selected file
print("Selected file #1 :", file_path1)
print("Selected file #2 :", file_path2)

dataframe=pd.concat([dataframe1,dataframe2])

dataframe = dataframe.sort_values('Time_fixed').reset_index(drop=True)


dataframe = dataframe.drop_duplicates()

# Keep the first occurrence, or change to 'last' if you prefer
dataframe = dataframe.drop_duplicates(subset='Time_fixed', keep='first')

def plot_temp():
    # only take a data point every 50, to speed up plotting by a lot, can use different number
    downsampled_df = dataframe.iloc[::50].copy()
    
    # creating figure
    plt.figure(figsize=(12, 6))
    plt.plot(downsampled_df['Time_fixed'], downsampled_df['T'])
    
    plt.title('Temperature over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature values')
    
    ax = plt.gca()
    
    # custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1),
        datetime(2007, 1, 1),
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
        datetime(2024, 1, 1),
        
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # format ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()
    plt.show()
    
def plot_rh():
    # only take a data point every 50, to speed up plotting by a lot, can use different number
    downsampled_df = dataframe.iloc[::50].copy()
    
    # creating figure
    plt.figure(figsize=(12, 6))
    plt.plot(downsampled_df['Time_fixed'], downsampled_df['RH'])
    
    plt.title('Relative Humidity over Time')
    plt.xlabel('Time')
    plt.ylabel('RH values')
    
    ax = plt.gca()
    
    # custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1),
        datetime(2007, 1, 1),
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
        datetime(2024, 1, 1),
        
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # format ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()
    plt.show()
  
# Calculate daily Temp avg, min,    max and create a table
dataframe = dataframe.sort_values('Time_fixed')
dataframe = dataframe.set_index('Time_fixed', drop=False)

# Resample by day and calculate daily min, max, avg for temperature
daily_summary_T = dataframe['T'].resample('1D').agg(['mean', 'min', 'max']).dropna().reset_index()
daily_summary_T.columns = ['Date', 'T_avg', 'T_min', 'T_max']

daily_summary_T['Date'] = pd.to_datetime(daily_summary_T['Date'])
monthly_summary_T = daily_summary_T.resample('ME', on='Date').agg({'T_avg': 'mean','T_min': 'min','T_max': 'max'}).dropna().reset_index()


def plot_daily_temp():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_T['Date'].diff()
    
    # gap is more than 1 day
    gaps = daily_summary_T[date_diffs > pd.Timedelta(days=1)]

    # monthly, only take a data point every 30 days/datapoints
    downsampled_daily_summary_T = daily_summary_T.iloc[::7].copy()
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(downsampled_daily_summary_T['Date'], downsampled_daily_summary_T['T_avg'], label='Average Temp', color='black', linewidth=2.5)
    plt.plot(downsampled_daily_summary_T['Date'], downsampled_daily_summary_T['T_min'], label='Min Temp', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(downsampled_daily_summary_T['Date'], downsampled_daily_summary_T['T_max'], label='Max Temp', color='red', linestyle='dashdot', linewidth=0.5)
    
    plt.title('Daily Temperature Summary \n Station: '+ name_of_file_no_csv)
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    
    ax = plt.gca()
    
    # Custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1),
        datetime(2007, 1, 1),
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
        datetime(2024, 1, 1),
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()
    
    
    # Shade missing data regions
    for i, row in gaps.iterrows():
        gap_start = daily_summary_T.loc[i - 1, 'Date']
        gap_end = row['Date']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)
    
    plt.legend()
    plt.show()

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

    plt.title('Monthly Temperature Summary \n Station: ' + name_of_file_no_csv_1)
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.gcf().autofmt_xdate()

    # Shade missing data regions
    for i, row in gaps.iterrows():
        gap_start = monthly_summary_T.loc[i - 1, 'Date']
        gap_end = row['Date']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()
    
    
# Resample by day and calculate daily min, max, avg for RH
daily_summary_rh = dataframe['RH'].resample('1D').agg(['mean', 'min', 'max']).dropna().reset_index()
daily_summary_rh.columns = ['Date', 'RH_avg', 'RH_min', 'RH_max']

daily_summary_rh['Date'] = pd.to_datetime(daily_summary_rh['Date'])
monthly_summary_rh = daily_summary_rh.resample('ME', on='Date').agg({'RH_avg': 'mean','RH_min': 'min','RH_max': 'max'}).dropna().reset_index()


def plot_daily_rh():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_rh['Date'].diff()
    
    # gap is more than 1 day
    gaps = daily_summary_rh[date_diffs > pd.Timedelta(days=1)]

    # monthly, only take a data point every 30 days/datapoints
    downsampled_daily_summary_rh = daily_summary_rh.iloc[::30].copy()
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(downsampled_daily_summary_rh['Date'], downsampled_daily_summary_rh['RH_avg'], label='Average RH', color='black', linewidth=2.5)
    plt.plot(downsampled_daily_summary_rh['Date'], downsampled_daily_summary_rh['RH_min'], label='Min RH', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(downsampled_daily_summary_rh['Date'], downsampled_daily_summary_rh['RH_max'], label='Max RH', color='red', linestyle='dashdot', linewidth=0.5)
    plt.scatter(downsampled_daily_summary_rh['Date'], downsampled_daily_summary_rh['RH_avg'],  color='black', s=10,  zorder=5)
    
    plt.title('Daily Relative Humidity Summary \n Station: '+ name_of_file_no_csv)
    plt.xlabel('Date')
    plt.ylabel('Relative Humidity')
    
    ax = plt.gca()
    
    # Custom tick dates
    custom_tick_dates = [
        datetime(2006, 1, 1),
        datetime(2007, 1, 1),
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
        datetime(2024, 1, 1),
    ]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()
    
    
    # Shade missing data regions
    for i, row in gaps.iterrows():
        gap_start = daily_summary_rh.loc[i - 1, 'Date']
        gap_end = row['Date']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)
    
    plt.legend()
    plt.show()

def plot_monthly_rh():
    # Calculate time differences between consecutive months
    date_diffs = monthly_summary_rh['Date'].diff()

    # Identify gaps larger than 1 month
    gaps = monthly_summary_rh[date_diffs > pd.Timedelta(days=31)]

    plt.figure(figsize=(12, 6))

    # Plot monthly average, min, and max RH
    plt.plot(
        monthly_summary_rh['Date'], 
        monthly_summary_rh['RH_avg'], 
        label='Average RH', 
        color='black', 
        linewidth=2.5
    )
    plt.plot(
        monthly_summary_rh['Date'], 
        monthly_summary_rh['RH_min'], 
        label='Min RH', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
    )
    plt.plot(
        monthly_summary_rh['Date'], 
        monthly_summary_rh['RH_max'], 
        label='Max RH', 
        color='red', 
        linestyle='dashdot', 
        linewidth=0.5
    )

    plt.title('Monthly Relative Humidity Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel('Date')
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.gcf().autofmt_xdate()

    # Shade missing data regions
    for i, row in gaps.iterrows():
        gap_start = monthly_summary_rh.loc[i - 1, 'Date']
        gap_end = row['Date']
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()

