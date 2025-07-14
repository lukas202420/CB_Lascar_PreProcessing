# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 11:50:37 2025

@author: timot
"""

#%% Importing the necessary modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import os

#%% How many files to merge?

files_to_merge=input("How many files do you wish to merge?")
files_to_merge=int(files_to_merge)

#%% Setting up the variables

#Change according to how these variables are called in your datafile

time_name="Datetime"
temp_name="Temperature (°C)"
rh_name="RH (%)"
td_name="Dew Point (°C)"

#%% Selecting the original data file

# create a root window and hide it
root = tk.Tk()
root.withdraw()

# open file dialog, get the selected file path
file_path = filedialog.askopenfilename()

name_of_file=os.path.basename(file_path)
name_of_file_no_csv=os.path.splitext(name_of_file)[0]

# prints selected file
print("Selected file:", file_path)

#creates dataframe from file selection
dataframe = pd.read_csv(file_path)

#%% For the remaining data files

for i in range (0,files_to_merge-1):
    
    file_path = filedialog.askopenfilename()

    name_of_file=os.path.basename(file_path)
    name_of_file_no_csv=os.path.splitext(name_of_file)[0]

    # prints selected file
    print("Selected file:", file_path)

    #creates dataframe from file selection
    temp_dataframe = pd.read_csv(file_path)
    
    dataframe= pd.concat([dataframe,temp_dataframe])
    dataframe = dataframe.sort_values(time_name).reset_index(drop=True)
    dataframe = dataframe.drop_duplicates()
    
    #%% Format the dataframe a bit better

    # adjust the number of rows to skip if needed, to avoid potential abberant data points 
    dataframe = dataframe.iloc[50:].reset_index(drop=True)

    dataframe['Time_fixed'] = pd.to_datetime(dataframe[time_name])

    
    #%% plot graphs 
  
dataframe = dataframe.sort_values('Time_fixed')
dataframe = dataframe.set_index('Time_fixed', drop=False)

# calculate daily min, max, avg for temperature
daily_summary_T = dataframe[temp_name].resample('1D').agg(['mean', 'min', 'max']).dropna().reset_index()
daily_summary_T.columns = [time_name, 'T_avg', 'T_min', 'T_max']

daily_summary_T[time_name] = pd.to_datetime(daily_summary_T[time_name])
monthly_summary_T = daily_summary_T.resample('ME', on=time_name).agg({'T_avg': 'mean','T_min': 'min','T_max': 'max'}).dropna().reset_index()


def plot_daily_temp():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_T[time_name].diff()
    
    # gap is more than 1 day
    gaps = daily_summary_T[date_diffs > pd.Timedelta(days=1)]

    # monthly, only take a data point every 30 days/datapoints, change as you wish, more points=slower graph to show, but more accuracy
    downsampled_daily_summary_T = daily_summary_T.iloc[::7].copy()
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(downsampled_daily_summary_T[time_name], downsampled_daily_summary_T['T_avg'], label='Average Temp', color='black', linewidth=2.5)
    plt.plot(downsampled_daily_summary_T[time_name], downsampled_daily_summary_T['T_min'], label='Min Temp', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(downsampled_daily_summary_T[time_name], downsampled_daily_summary_T['T_max'], label='Max Temp', color='red', linestyle='dashdot', linewidth=0.5)
    
    plt.title('Daily Temperature Summary \n Station: '+ name_of_file_no_csv)
    plt.xlabel(time_name)
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
        gap_start = daily_summary_T.loc[i - 1, time_name]
        gap_end = row[time_name]
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)
    
    plt.legend()
    plt.show()

def plot_monthly_temp():
    
    # Calculate time differences between consecutive months
    date_diffs = monthly_summary_T[time_name].diff()

    # Identify gaps larger than 1 month
    gaps = monthly_summary_T[date_diffs > pd.Timedelta(days=31)]

    plt.figure(figsize=(12, 6))

    # Plot monthly average, min, and max
    plt.plot(
        monthly_summary_T[time_name], 
        monthly_summary_T['T_avg'], 
        label='Average Temp', 
        color='black', 
        linewidth=2.5
    )
    plt.plot(
        monthly_summary_T[time_name], 
        monthly_summary_T['T_min'], 
        label='Min Temp', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
    )
    plt.plot(
        monthly_summary_T[time_name], 
        monthly_summary_T['T_max'], 
        label='Max Temp', 
        color='red', 
        linestyle='dashdot', 
        linewidth=0.5
    )

    plt.title('Monthly Temperature Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
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
        gap_start = monthly_summary_T.loc[i - 1, time_name]
        gap_end = row[time_name]
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()
    
    
# calculate daily min, max, avg for RH
daily_summary_rh = dataframe[rh_name].resample('1D').agg(['mean', 'min', 'max']).dropna().reset_index()
daily_summary_rh.columns = [time_name, 'RH_avg', 'RH_min', 'RH_max']

daily_summary_rh[time_name] = pd.to_datetime(daily_summary_rh[time_name])
monthly_summary_rh = daily_summary_rh.resample('ME', on=time_name).agg({'RH_avg': 'mean','RH_min': 'min','RH_max': 'max'}).dropna().reset_index()


def plot_daily_rh():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_rh[time_name].diff()
    
    # gap is more than 1 day, or whatever you prefer
    gaps = daily_summary_rh[date_diffs > pd.Timedelta(days=1)]

    # monthly, only take a data point every 30 days/datapoints
    downsampled_daily_summary_rh = daily_summary_rh.iloc[::30].copy()
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_avg'], label='Average RH', color='black', linewidth=2.5)
    plt.plot(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_min'], label='Min RH', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_max'], label='Max RH', color='red', linestyle='dashdot', linewidth=0.5)
    plt.scatter(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_avg'],  color='black', s=10,  zorder=5)
    
    plt.title('Daily Relative Humidity Summary \n Station: '+ name_of_file_no_csv)
    plt.xlabel(time_name)
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
        gap_start = daily_summary_rh.loc[i - 1, time_name]
        gap_end = row[time_name]
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)
    
    plt.legend()
    plt.show()

def plot_monthly_rh():
    # Calculate time differences between consecutive months
    date_diffs = monthly_summary_rh[time_name].diff()

    # Identify gaps larger than 1 month
    gaps = monthly_summary_rh[date_diffs > pd.Timedelta(days=31)]

    plt.figure(figsize=(12, 6))

    # Plot monthly average, min, and max RH
    plt.plot(
        monthly_summary_rh[time_name], 
        monthly_summary_rh['RH_avg'], 
        label='Average RH', 
        color='black', 
        linewidth=2.5
    )
    plt.plot(
        monthly_summary_rh[time_name], 
        monthly_summary_rh['RH_min'], 
        label='Min RH', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
    )
    plt.plot(
        monthly_summary_rh[time_name], 
        monthly_summary_rh['RH_max'], 
        label='Max RH', 
        color='red', 
        linestyle='dashdot', 
        linewidth=0.5
    )

    plt.title('Monthly Relative Humidity Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
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
        gap_start = monthly_summary_rh.loc[i - 1, time_name]
        gap_end = row[time_name]
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()

    
    
    