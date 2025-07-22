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
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose

#%% How many files to merge?

files_to_merge=input("How many files do you wish to merge?")
files_to_merge=int(files_to_merge)

#%% Setting up the variables

#Change according to how these variables are called in your datafile

'''
time_name="Time"
temp_name="T"
rh_name="RH"
td_name="Td"
'''

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
    

    # prints selected file
    print("Selected file:", file_path)

    #creates dataframe from file selection
    temp_dataframe = pd.read_csv(file_path)
    
    dataframe= pd.concat([dataframe,temp_dataframe])
    dataframe = dataframe.sort_values(time_name).reset_index(drop=True)
    dataframe = dataframe.drop_duplicates()
    
#%% Format the dataframe a bit better

# adjust the number of rows to skip if needed, to avoid potential abberant data points 
#dataframe = dataframe.iloc[50:].reset_index(drop=True)

dataframe['Time_fixed'] = pd.to_datetime(dataframe[time_name])
dataframe = dataframe.set_index('Time_fixed', drop=True)

# Define full time index range at hourly resolution
full_index = pd.date_range(start=dataframe.index.min(),
                           end=dataframe.index.max(),
                           freq='h')

# Create a temporary dataframe to check timestamp presence
timestamp_df = pd.DataFrame(index=full_index)
timestamp_df['present'] = timestamp_df.index.isin(dataframe.index)

#%% Only keeping months with 50%+ data
timestamp_df['month'] = timestamp_df.index.to_period('M')

# Actual present timestamps per month
actual_counts_monthly = timestamp_df.groupby('month')['present'].sum()
expected_counts_monthly = timestamp_df.groupby('month').size()

missing_ratio_monthly = 1 - (actual_counts_monthly / expected_counts_monthly)
missing_ratio_monthly = missing_ratio_monthly.to_frame(name='Missing_Ratio')

min_valid_points_per_month = 24 * 20  # 20 days

# Filter months with ≤50% missing and ≥ min points
filtered_missing_ratio_monthly = missing_ratio_monthly[
    (missing_ratio_monthly['Missing_Ratio'] <= 0.5) &
    (actual_counts_monthly >= min_valid_points_per_month)
]

valid_months = filtered_missing_ratio_monthly.index
bad_months = missing_ratio_monthly[~missing_ratio_monthly.index.isin(valid_months)]

#%% Only keeping weeks with 4+ days of data (96+ hourly points)

timestamp_df['week'] = timestamp_df.index.to_period('W')

actual_counts_weekly = timestamp_df.groupby('week')['present'].sum()
expected_counts_weekly = timestamp_df.groupby('week').size()

missing_ratio_weekly = 1 - (actual_counts_weekly / expected_counts_weekly)
missing_ratio_weekly = missing_ratio_weekly.to_frame(name='Missing_Ratio')

min_valid_points_per_week = 96  # 4 days

filtered_missing_ratio_weekly = missing_ratio_weekly[
    (missing_ratio_weekly['Missing_Ratio'] <= 0.5) &
    (actual_counts_weekly >= min_valid_points_per_week)
]

valid_weeks = filtered_missing_ratio_weekly.index
bad_weeks = missing_ratio_weekly[~missing_ratio_weekly.index.isin(valid_weeks)]

#%% plot graphs 
  
dataframe = dataframe.sort_values('Time_fixed')

# calculate daily min, max, avg for temperature
daily_summary_T = dataframe[temp_name].resample('1D').agg(['mean', 'min', 'max']).reset_index()
daily_summary_T.columns = [time_name, 'T_avg', 'T_min', 'T_max']

daily_summary_T[time_name] = pd.to_datetime(daily_summary_T[time_name])




monthly_summary_T = daily_summary_T.resample('ME', on=time_name).agg({'T_avg': 'mean','T_min': 'min','T_max': 'max'}).reset_index()
monthly_summary_T['MonthPeriod'] = monthly_summary_T[time_name].dt.to_period('M')
# filter rows to keep valid data
monthly_summary_T = monthly_summary_T[monthly_summary_T['MonthPeriod'].isin(valid_months)]




weekly_summary_T = daily_summary_T.resample('W-MON', on=time_name).agg({'T_avg': 'mean','T_min': 'min','T_max': 'max'}).reset_index()

weekly_summary_T['WeekPeriod'] = weekly_summary_T[time_name].dt.to_period('W')

# filter rows to keep valid data
weekly_summary_T = weekly_summary_T[weekly_summary_T['WeekPeriod'].isin(valid_weeks)]

def plot_daily_temp():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_T[time_name].diff()
    
    # gap is more than 1 day
    gaps = daily_summary_T[date_diffs > pd.Timedelta(days=1)]

    # monthly, only take a data point every 30 days/datapoints, change as you wish, more points=slower graph to show, but more accuracy
    downsampled_daily_summary_T = daily_summary_T.iloc[::1].copy()
    
    # Drop rows with NaNs in any of the temperature columns (only for regression)
    valid_data = downsampled_daily_summary_T.dropna(subset=['T_avg', 'T_min', 'T_max'])
    x = mdates.date2num(valid_data[time_name])
    y_avg = valid_data['T_avg']
    y_min = valid_data['T_min']
    y_max = valid_data['T_max']
    
    # Fit linear trendlines
    z_avg = np.polyfit(x, y_avg, 1)
    z_min = np.polyfit(x, y_min, 1)
    z_max = np.polyfit(x, y_max, 1)

    
    # Compute trend values
    trend_avg = np.poly1d(z_avg)(x)
    trend_min = np.poly1d(z_min)(x)
    trend_max = np.poly1d(z_max)(x)
    
    
    # T_avg trend
    result_avg = linregress(x, y_avg)
    slope_avg = result_avg.slope
    intercept_avg = result_avg.intercept
    r2_avg = result_avg.rvalue**2
    pval_avg = result_avg.pvalue
    stderr_avg = result_avg.stderr
    slope_avg_per_year = slope_avg * 365
    
    # T_min trend
    result_min = linregress(x, y_min)
    slope_min = result_min.slope
    intercept_min = result_min.intercept
    r2_min = result_min.rvalue**2
    pval_min = result_min.pvalue
    stderr_min = result_min.stderr
    slope_min_per_year = slope_min * 365
    
    # T_max trend
    result_max = linregress(x, y_max)
    slope_max = result_max.slope
    intercept_max = result_max.intercept
    r2_max = result_max.rvalue**2
    pval_max = result_max.pvalue
    stderr_max = result_max.stderr
    slope_max_per_year = slope_max * 365

    # Generate trendlines
    trend_avg = slope_avg * x + intercept_avg
    trend_min = slope_min * x + intercept_min
    trend_max = slope_max * x + intercept_max
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(downsampled_daily_summary_T[time_name], downsampled_daily_summary_T['T_avg'], label='Average Temp', color='black', linewidth=2.5)
    plt.plot(downsampled_daily_summary_T[time_name], downsampled_daily_summary_T['T_min'], label='Min Temp', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(downsampled_daily_summary_T[time_name], downsampled_daily_summary_T['T_max'], label='Max Temp', color='red', linestyle='dashdot', linewidth=0.5)
   
    # Plot trendlines
    plt.plot(valid_data[time_name], trend_avg, label=f'Trend Avg ({slope_avg_per_year:.2f}°C/yr, R²={r2_avg:.2f})', color='black', linestyle='dashed')
    plt.plot(valid_data[time_name], trend_min, label=f'Trend Min ({slope_min_per_year:.2f}°C/yr, R²={r2_min:.2f})', color='blue', linestyle='dashed')
    plt.plot(valid_data[time_name], trend_max, label=f'Trend Max ({slope_max_per_year:.2f}°C/yr, R²={r2_max:.2f})', color='red', linestyle='dashed')


    
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
        ax.axvspan(gap_start, gap_end, color='gray', alpha=0.3, label='_nolegend_')
    
    plt.legend()
    plt.show()

    print(f"Average Temp: slope={slope_avg:.6f}, R²={r2_avg:.4f}, p={pval_avg:.4g}, stderr={stderr_avg:.4f}")
    print(f"Min Temp:     slope={slope_min:.6f}, R²={r2_min:.4f}, p={pval_min:.4g}, stderr={stderr_min:.4f}")
    print(f"Max Temp:     slope={slope_max:.6f}, R²={r2_max:.4f}, p={pval_max:.4g}, stderr={stderr_max:.4f}")


def plot_weekly_temp():
    weekly_summary_T = dataframe[temp_name].resample('1W').agg(['mean', 'min', 'max']).reset_index()
    weekly_summary_T.columns = [time_name, 'T_avg', 'T_min', 'T_max']
    
    valid_data = weekly_summary_T.dropna(subset=['T_avg', 'T_min', 'T_max'])
    x = mdates.date2num(valid_data[time_name])
    y_avg = valid_data['T_avg']
    y_min = valid_data['T_min']
    y_max = valid_data['T_max']
    
    # Linear regression
    result_avg = linregress(x, y_avg)
    result_min = linregress(x, y_min)
    result_max = linregress(x, y_max)
    
    # Compute trends
    trend_avg = result_avg.slope * x + result_avg.intercept
    trend_min = result_min.slope * x + result_min.intercept
    trend_max = result_max.slope * x + result_max.intercept
    
    slope_avg_per_year = result_avg.slope * 52.1775
    slope_min_per_year = result_min.slope * 52.1775
    slope_max_per_year = result_max.slope * 52.1775
    
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_summary_T[time_name], weekly_summary_T['T_avg'], label='Average Temp', color='black', linewidth=2.5)
    plt.plot(weekly_summary_T[time_name], weekly_summary_T['T_min'], label='Min Temp', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(weekly_summary_T[time_name], weekly_summary_T['T_max'], label='Max Temp', color='red', linestyle='dashdot', linewidth=0.5)
    
    plt.plot(valid_data[time_name], trend_avg, label=f'Trend Avg ({slope_avg_per_year:.2f}°C/yr, R²={result_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
    plt.plot(valid_data[time_name], trend_min, label=f'Trend Min ({slope_min_per_year:.2f}°C/yr, R²={result_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
    plt.plot(valid_data[time_name], trend_max, label=f'Trend Max ({slope_max_per_year:.2f}°C/yr, R²={result_max.rvalue**2:.2f})', color='red', linestyle='dashed')
    
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
        datetime(2024, 1, 1),]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()

    for period in bad_weeks.index:
        start = period.start_time
        end = period.end_time
        ax.axvspan(start, end, color='gray', alpha=0.2)
        
        
    plt.title('Weekly Temperature Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Temperature')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
    
    print(f"Average Temp: slope={result_avg.slope:.6f}, R²={result_avg.rvalue**2:.4f}, p={result_avg.pvalue:.4g}, stderr={result_avg.stderr:.4f}")
    print(f"Min Temp:     slope={result_min.slope:.6f}, R²={result_min.rvalue**2:.4f}, p={result_min.pvalue:.4g}, stderr={result_min.stderr:.4f}")
    print(f"Max Temp:     slope={result_max.slope:.6f}, R²={result_max.rvalue**2:.4f}, p={result_max.pvalue:.4g}, stderr={result_max.stderr:.4f}")


def plot_monthly_temp():
    monthly_summary_T = dataframe[temp_name].resample('1M').agg(['mean', 'min', 'max']).reset_index()
    monthly_summary_T.columns = [time_name, 'T_avg', 'T_min', 'T_max']

    valid_data = monthly_summary_T.dropna(subset=['T_avg', 'T_min', 'T_max'])
    x = mdates.date2num(valid_data[time_name])
    y_avg = valid_data['T_avg']
    y_min = valid_data['T_min']
    y_max = valid_data['T_max']
    
    # Linear regression
    result_avg = linregress(x, y_avg)
    result_min = linregress(x, y_min)
    result_max = linregress(x, y_max)
    
    # Compute trends
    trend_avg = result_avg.slope * x + result_avg.intercept
    trend_min = result_min.slope * x + result_min.intercept
    trend_max = result_max.slope * x + result_max.intercept
    
    slope_avg_per_year = result_avg.slope * 12
    slope_min_per_year = result_min.slope * 12
    slope_max_per_year = result_max.slope * 12
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_summary_T[time_name], monthly_summary_T['T_avg'], label='Average Temp', color='black', linewidth=2.5)
    plt.plot(monthly_summary_T[time_name], monthly_summary_T['T_min'], label='Min Temp', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(monthly_summary_T[time_name], monthly_summary_T['T_max'], label='Max Temp', color='red', linestyle='dashdot', linewidth=0.5)
    
    plt.plot(valid_data[time_name], trend_avg, label=f'Trend Avg ({slope_avg_per_year:.2f}°C/yr, R²={result_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
    plt.plot(valid_data[time_name], trend_min, label=f'Trend Min ({slope_min_per_year:.2f}°C/yr, R²={result_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
    plt.plot(valid_data[time_name], trend_max, label=f'Trend Max ({slope_max_per_year:.2f}°C/yr, R²={result_max.rvalue**2:.2f})', color='red', linestyle='dashed')
    
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
        datetime(2024, 1, 1),]
    
    ax.set_xticks(custom_tick_dates)
    
    # Format tick labels as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()

    for period in bad_months.index:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)
        
    plt.title('Monthly Temperature Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Temperature')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
    
    print(f"Average Temp: slope={result_avg.slope:.6f}, R²={result_avg.rvalue**2:.4f}, p={result_avg.pvalue:.4g}, stderr={result_avg.stderr:.4f}")
    print(f"Min Temp:     slope={result_min.slope:.6f}, R²={result_min.rvalue**2:.4f}, p={result_min.pvalue:.4g}, stderr={result_min.stderr:.4f}")
    print(f"Max Temp:     slope={result_max.slope:.6f}, R²={result_max.rvalue**2:.4f}, p={result_max.pvalue:.4g}, stderr={result_max.stderr:.4f}")

    
# calculate min, max, avg for RH
daily_summary_rh = dataframe[rh_name].resample('1D').agg(['mean', 'min', 'max']).reset_index()
daily_summary_rh.columns = [time_name, 'RH_avg', 'RH_min', 'RH_max']

daily_summary_rh[time_name] = pd.to_datetime(daily_summary_rh[time_name])
monthly_summary_rh = daily_summary_rh.resample('ME', on=time_name).agg({'RH_avg': 'mean','RH_min': 'min','RH_max': 'max'}).reset_index()

monthly_summary_rh['MonthPeriod'] = monthly_summary_rh[time_name].dt.to_period('M')
monthly_summary_rh = monthly_summary_rh[monthly_summary_rh['MonthPeriod'].isin(valid_months)]

weekly_summary_rh = daily_summary_rh.resample('W', on=time_name).agg({'RH_avg': 'mean','RH_min': 'min','RH_max': 'max'}).reset_index()

weekly_summary_rh['WeekPeriod'] = weekly_summary_rh[time_name].dt.to_period('W')
weekly_summary_rh = weekly_summary_rh[weekly_summary_rh['WeekPeriod'].isin(valid_weeks)]

def plot_daily_rh():
    # Similar resampling and processing assumed done beforehand as with temperature
    downsampled_daily_summary_RH = daily_summary_rh.iloc[::1].copy()
    
    x = mdates.date2num(downsampled_daily_summary_RH[time_name]) 
    
    # Linear regression for RH_avg, RH_min, RH_max
    res_avg = linregress(x, downsampled_daily_summary_RH['RH_avg'])
    res_min = linregress(x, downsampled_daily_summary_RH['RH_min'])
    res_max = linregress(x, downsampled_daily_summary_RH['RH_max'])
    
    trend_avg = res_avg.slope * x + res_avg.intercept
    trend_min = res_min.slope * x + res_min.intercept
    trend_max = res_max.slope * x + res_max.intercept
    
    slope_avg_per_year = res_avg.slope * 365
    slope_min_per_year = res_min.slope * 365
    slope_max_per_year = res_max.slope * 365
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(downsampled_daily_summary_RH[time_name], downsampled_daily_summary_RH['RH_avg'], label='Average RH', color='black', linewidth=2.5)
    plt.plot(downsampled_daily_summary_RH[time_name], downsampled_daily_summary_RH['RH_min'], label='Min RH', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(downsampled_daily_summary_RH[time_name], downsampled_daily_summary_RH['RH_max'], label='Max RH', color='red', linestyle='dashdot', linewidth=0.5)
    
    plt.plot(downsampled_daily_summary_RH[time_name], trend_avg, label=f'Trend Avg ({slope_avg_per_year:.4f}%/yr, R²={res_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
    plt.plot(downsampled_daily_summary_RH[time_name], trend_min, label=f'Trend Min ({slope_min_per_year:.4f}%/yr, R²={res_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
    plt.plot(downsampled_daily_summary_RH[time_name], trend_max, label=f'Trend Max ({slope_max_per_year:.4f}%/yr, R²={res_max.rvalue**2:.2f})', color='red', linestyle='dashed')

    plt.title('Daily Relative Humidity Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Relative Humidity (%)')

    ax = plt.gca()
    # Customize ticks as you do for temperature
    custom_tick_dates = [datetime(y, 1, 1) for y in range(2006, 2025)]
    ax.set_xticks(custom_tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.legend()
    plt.show()
    
    print(f"Avg RH: slope={res_avg.slope:.6f}, R²={res_avg.rvalue**2:.4f}, p={res_avg.pvalue:.4g}, stderr={res_avg.stderr:.4f}")
    print(f"Min RH: slope={res_min.slope:.6f}, R²={res_min.rvalue**2:.4f}, p={res_min.pvalue:.4g}, stderr={res_min.stderr:.4f}")
    print(f"Max RH: slope={res_max.slope:.6f}, R²={res_max.rvalue**2:.4f}, p={res_max.pvalue:.4g}, stderr={res_max.stderr:.4f}")


def plot_weekly_rh():
    weekly_summary_RH = daily_summary_rh.resample('W', on=time_name).agg({
        'RH_avg': 'mean',
        'RH_min': 'min',
        'RH_max': 'max'
    }).reset_index()
    
    if 'valid_weeks' in globals():
        weekly_summary_RH = weekly_summary_RH[weekly_summary_RH[time_name].dt.to_period('W').isin(valid_weeks)]

    x = mdates.date2num(weekly_summary_RH[time_name])
    
    res_avg = linregress(x, weekly_summary_RH['RH_avg'])
    res_min = linregress(x, weekly_summary_RH['RH_min'])
    res_max = linregress(x, weekly_summary_RH['RH_max'])
    
    trend_avg = res_avg.slope * x + res_avg.intercept
    trend_min = res_min.slope * x + res_min.intercept
    trend_max = res_max.slope * x + res_max.intercept
    
    slope_avg_per_year = res_avg.slope * 365
    slope_min_per_year = res_min.slope * 365
    slope_max_per_year = res_max.slope * 365
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(weekly_summary_RH[time_name], weekly_summary_RH['RH_avg'], label='Average RH', color='black', linewidth=2.5)
    plt.plot(weekly_summary_RH[time_name], weekly_summary_RH['RH_min'], label='Min RH', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(weekly_summary_RH[time_name], weekly_summary_RH['RH_max'], label='Max RH', color='red', linestyle='dashdot', linewidth=0.5)
    
    plt.plot(weekly_summary_RH[time_name], trend_avg, label=f'Trend Avg ({slope_avg_per_year:.4f}%/yr, R²={res_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
    plt.plot(weekly_summary_RH[time_name], trend_min, label=f'Trend Min ({slope_min_per_year:.4f}%/yr, R²={res_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
    plt.plot(weekly_summary_RH[time_name], trend_max, label=f'Trend Max ({slope_max_per_year:.4f}%/yr, R²={res_max.rvalue**2:.2f})', color='red', linestyle='dashed')

    plt.title('Weekly Relative Humidity Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Relative Humidity (%)')

    ax = plt.gca()
    custom_tick_dates = [datetime(y, 1, 1) for y in range(2006, 2025)]
    ax.set_xticks(custom_tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    plt.legend()
    plt.show()
    
    print(f"Avg RH: slope={res_avg.slope:.6f}, R²={res_avg.rvalue**2:.4f}, p={res_avg.pvalue:.4g}, stderr={res_avg.stderr:.4f}")
    print(f"Min RH: slope={res_min.slope:.6f}, R²={res_min.rvalue**2:.4f}, p={res_min.pvalue:.4g}, stderr={res_min.stderr:.4f}")
    print(f"Max RH: slope={res_max.slope:.6f}, R²={res_max.rvalue**2:.4f}, p={res_max.pvalue:.4g}, stderr={res_max.stderr:.4f}")



def plot_monthly_rh():
    # Assume monthly_summary_RH prepared similarly to monthly_summary_T
    x = mdates.date2num(monthly_summary_rh[time_name])
    
    res_avg = linregress(x, monthly_summary_rh['RH_avg'])
    res_min = linregress(x, monthly_summary_rh['RH_min'])
    res_max = linregress(x, monthly_summary_rh['RH_max'])
    
    trend_avg = res_avg.slope * x + res_avg.intercept
    trend_min = res_min.slope * x + res_min.intercept
    trend_max = res_max.slope * x + res_max.intercept
    
    slope_avg_per_year = res_avg.slope * 365
    slope_min_per_year = res_min.slope * 365
    slope_max_per_year = res_max.slope * 365
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(monthly_summary_rh[time_name], monthly_summary_rh['RH_avg'], label='Average RH', color='black', linewidth=2.5)
    plt.plot(monthly_summary_rh[time_name], monthly_summary_rh['RH_min'], label='Min RH', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(monthly_summary_rh[time_name], monthly_summary_rh['RH_max'], label='Max RH', color='red', linestyle='dashdot', linewidth=0.5)
    
    plt.plot(monthly_summary_rh[time_name], trend_avg, label=f'Trend Avg ({slope_avg_per_year:.4f}%/yr, R²={res_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
    plt.plot(monthly_summary_rh[time_name], trend_min, label=f'Trend Min ({slope_min_per_year:.4f}%/yr, R²={res_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
    plt.plot(monthly_summary_rh[time_name], trend_max, label=f'Trend Max ({slope_max_per_year:.4f}%/yr, R²={res_max.rvalue**2:.2f})', color='red', linestyle='dashed')
    
    plt.title('Monthly Relative Humidity Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Relative Humidity (%)')

    ax = plt.gca()
    custom_tick_dates = [datetime(y, 1, 1) for y in range(2006, 2025)]
    ax.set_xticks(custom_tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    # Shade missing months if you have bad_months defined
    if 'bad_months' in globals():
        for period in bad_months.index:
            start = period.to_timestamp()
            end = (period + 1).to_timestamp()
            ax.axvspan(start, end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()
    
    print(f"Avg RH: slope={res_avg.slope:.6f}, R²={res_avg.rvalue**2:.4f}, p={res_avg.pvalue:.4g}, stderr={res_avg.stderr:.4f}")
    print(f"Min RH: slope={res_min.slope:.6f}, R²={res_min.rvalue**2:.4f}, p={res_min.pvalue:.4g}, stderr={res_min.stderr:.4f}")
    print(f"Max RH: slope={res_max.slope:.6f}, R²={res_max.rvalue**2:.4f}, p={res_max.pvalue:.4g}, stderr={res_max.stderr:.4f}")


'''
def plot_daily_rh():
    
    # Calculate time differences between consecutive dates
    date_diffs = daily_summary_rh[time_name].diff()
    
    # gap is more than 1 day, or whatever you prefer
    gaps = daily_summary_rh[date_diffs > pd.Timedelta(days=1)]

    # monthly, only take a data point every 30 days/datapoints or just everyday, change the number in .iloc[::1]
    downsampled_daily_summary_rh = daily_summary_rh.iloc[::1].copy()
    
    plt.figure(figsize=(12, 6))

    # Plot daily average, min, and max
    plt.plot(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_avg'], label='Average RH', color='black', linewidth=2.5)
    plt.plot(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_min'], label='Min RH', color='blue', linestyle='dashdot', linewidth=0.5)
    plt.plot(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_max'], label='Max RH', color='red', linestyle='dashdot', linewidth=0.5)
    #plt.scatter(downsampled_daily_summary_rh[time_name], downsampled_daily_summary_rh['RH_avg'],  color='black', s=10,  zorder=5)
    
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

def plot_weekly_rh():
    plt.figure(figsize=(12, 6))

    # Plot weekly average, min, and max RH
    plt.plot(
        weekly_summary_rh[time_name], 
        weekly_summary_rh['RH_avg'], 
        label='Average RH', 
        color='black', 
        linewidth=2.5
    )
    plt.plot(
        weekly_summary_rh[time_name], 
        weekly_summary_rh['RH_min'], 
        label='Min RH', 
        color='blue', 
        linestyle='dashdot', 
        linewidth=0.5
    )
    plt.plot(
        weekly_summary_rh[time_name], 
        weekly_summary_rh['RH_max'], 
        label='Max RH', 
        color='red', 
        linestyle='dashdot', 
        linewidth=0.5
    )

    plt.title('Weekly Relative Humidity Summary \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Relative Humidity')

    ax = plt.gca()

    # Optional: Customize x-axis ticks (year start every Jan)
    custom_tick_dates = [datetime(y, 1, 1) for y in range(2006, 2025)]
    ax.set_xticks(custom_tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.gcf().autofmt_xdate()

    # Shade missing (bad) weeks
    for period in bad_weeks.index:
        start = period.start_time
        end = period.end_time
        ax.axvspan(start, end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()

def plot_monthly_rh():
    
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
    for period in bad_months.index:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.3)

    plt.legend()
    plt.show()
'''

#%%

def plot_weekly_rh_detrend():

    # Prepare weekly RH data (make sure it's indexed by datetime)
    weekly_summary_RH = daily_summary_rh.resample('W', on=time_name).agg({
        'RH_avg': 'mean',
        'RH_min': 'min',
        'RH_max': 'max'
    }).reset_index()

    weekly_summary_RH['WeekPeriod'] = weekly_summary_RH[time_name].dt.to_period('W')

    if 'valid_weeks' in globals():
        weekly_summary_RH = weekly_summary_RH[weekly_summary_RH['WeekPeriod'].isin(valid_weeks)]

    weekly_summary_RH.set_index(time_name, inplace=True)

    # Decompose with yearly seasonality (52 weeks)
    decomposition_avg = seasonal_decompose(weekly_summary_RH['RH_avg'], model='additive', period=52)
    decomposition_min = seasonal_decompose(weekly_summary_RH['RH_min'], model='additive', period=52)
    decomposition_max = seasonal_decompose(weekly_summary_RH['RH_max'], model='additive', period=52)

    # Deseasonalized = original - seasonal component
    deseasonal_avg = weekly_summary_RH['RH_avg'] - decomposition_avg.seasonal
    deseasonal_min = weekly_summary_RH['RH_min'] - decomposition_min.seasonal
    deseasonal_max = weekly_summary_RH['RH_max'] - decomposition_max.seasonal

    # Detrended = original - trend component
    detrended_avg = weekly_summary_RH['RH_avg'] - decomposition_avg.trend
    detrended_min = weekly_summary_RH['RH_min'] - decomposition_min.trend
    detrended_max = weekly_summary_RH['RH_max'] - decomposition_max.trend

    # You can now run linregress on trend or deseasonalized series as you want

    # Example: trend linregress for average RH
    x = mdates.date2num(weekly_summary_RH.index)
    res_avg = linregress(x[~decomposition_avg.trend.isna()], decomposition_avg.trend)
    slope_per_year = res_avg.slope * 365

    plt.figure(figsize=(12, 6))

    plt.plot(weekly_summary_RH.index, weekly_summary_RH['RH_avg'], label='Original Avg RH', color='black', linewidth=2)
    plt.plot(weekly_summary_RH.index, detrended_avg, label='Detrended Avg RH', color='orange', linewidth=1.5)
    plt.plot(weekly_summary_RH.index, deseasonal_avg, label='Deseasonalized Avg RH', color='green', linewidth=1.5)

    plt.plot(weekly_summary_RH.index[~decomposition_avg.trend.isna()], decomposition_avg.trend, label=f'Trend Avg RH ({slope_per_year:.4f}/yr)', color='red', linestyle='dashed')

    plt.title('Weekly RH Summary with Decomposition \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Relative Humidity (%)')

    plt.legend()
    plt.show()

    print(f"Weekly Avg RH Trend slope: {res_avg.slope:.6f} per day, approx {slope_per_year:.4f} per year")
    print(f"R-squared: {res_avg.rvalue**2:.4f}, p-value: {res_avg.pvalue:.4g}, stderr: {res_avg.stderr:.4f}")



def plot_monthly_rh_detrend():
    monthly_summary_RH = daily_summary_rh.resample('M', on=time_name).agg({
        'RH_avg': 'mean',
        'RH_min': 'min',
        'RH_max': 'max'
    }).reset_index()

    monthly_summary_RH['MonthPeriod'] = monthly_summary_RH[time_name].dt.to_period('M')

    if 'valid_months' in globals():
        monthly_summary_RH = monthly_summary_RH[monthly_summary_RH['MonthPeriod'].isin(valid_months)]

    monthly_summary_RH.set_index(time_name, inplace=True)

    # Decompose with yearly seasonality (12 months)
    decomposition_avg = seasonal_decompose(monthly_summary_RH['RH_avg'], model='additive', period=12)
    decomposition_min = seasonal_decompose(monthly_summary_RH['RH_min'], model='additive', period=12)
    decomposition_max = seasonal_decompose(monthly_summary_RH['RH_max'], model='additive', period=12)

    deseasonal_avg = monthly_summary_RH['RH_avg'] - decomposition_avg.seasonal
    deseasonal_min = monthly_summary_RH['RH_min'] - decomposition_min.seasonal
    deseasonal_max = monthly_summary_RH['RH_max'] - decomposition_max.seasonal

    detrended_avg = monthly_summary_RH['RH_avg'] - decomposition_avg.trend
    detrended_min = monthly_summary_RH['RH_min'] - decomposition_min.trend
    detrended_max = monthly_summary_RH['RH_max'] - decomposition_max.trend

    x = mdates.date2num(monthly_summary_RH.index)
    res_avg = linregress(x[~decomposition_avg.trend.isna()], decomposition_avg.trend)
    slope_per_year = res_avg.slope * 365

    plt.figure(figsize=(12, 6))

    plt.plot(monthly_summary_RH.index, monthly_summary_RH['RH_avg'], label='Original Avg RH', color='black', linewidth=2)
    plt.plot(monthly_summary_RH.index, detrended_avg, label='Detrended Avg RH', color='orange', linewidth=1.5)
    plt.plot(monthly_summary_RH.index, deseasonal_avg, label='Deseasonalized Avg RH', color='green', linewidth=1.5)

    plt.plot(monthly_summary_RH.index[~decomposition_avg.trend.isna()], decomposition_avg.trend, label=f'Trend Avg RH ({slope_per_year:.4f}/yr)', color='red', linestyle='dashed')

    plt.title('Monthly RH Summary with Decomposition \n Station: ' + name_of_file_no_csv)
    plt.xlabel(time_name)
    plt.ylabel('Relative Humidity (%)')

    plt.legend()
    plt.show()

    print(f"Monthly Avg RH Trend slope: {res_avg.slope:.6f} per day, approx {slope_per_year:.4f} per year")
    print(f"R-squared: {res_avg.rvalue**2:.4f}, p-value: {res_avg.pvalue:.4g}, stderr: {res_avg.stderr:.4f}")










