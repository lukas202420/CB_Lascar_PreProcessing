# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:27:23 2025

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

time_name="Time"
temp_name="T"
rh_name="RH"
td_name="Td"

'''
time_name="Datetime"
temp_name="Temperature (°C)"
rh_name="RH (%)"
td_name="Dew Point (°C)"
'''

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

#%% Selecting the original data file
#creates dataframe from file selection
#dataframe = pd.read_csv(r"C:\Users\timot\OneDrive - McGill University\Peru\Lascar\RealData\PeruData\Lascars_2006_to_2021\Llan1_2006_2021.csv")
#dataframe = pd.read_csv(r"C:\Users\timot\OneDrive - McGill University\Peru\Lascar\RealData\PeruData\Lascars_2006_to_2021\Llan2_2006_2021.csv")
#dataframe = pd.read_csv(r"C:\Users\timot\OneDrive - McGill University\Peru\Lascar\RealData\PeruData\Lascars_2006_to_2021\Llan3_2006_2021.csv")
#dataframe = pd.read_csv(r"C:\Users\timot\OneDrive - McGill University\Peru\Lascar\RealData\PeruData\Lascars_2006_to_2021\Llan4_2006_2021.csv")


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

dataframe['Time_index'] = pd.to_datetime(dataframe[time_name])
dataframe = dataframe.set_index('Time_index', drop=False)

time_name='Time_index'
dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
# Define full time index range at hourly resolution
full_index = pd.date_range(start=dataframe.index.min(),
                           end=dataframe.index.max(),
                           freq='h')

dataframe=dataframe.reindex(full_index)
dataframe['present'] = ~dataframe[time_name].isna()

#%% check if valid month

dataframe['month']=dataframe.index.to_period('M')


# Actual present timestamps per month
actual_counts_monthly = dataframe.groupby('month')['present'].sum()

# Set threshold: minimum number of valid data points per month
min_valid_points = 24 * 20  # e.g. 20 days worth of hourly data

# Filter valid and bad months
valid_months = actual_counts_monthly[actual_counts_monthly >= min_valid_points].index
bad_months = actual_counts_monthly[actual_counts_monthly < min_valid_points].index

#%%% Filetring out the bad months from the dataframe

cols_to_nan = [temp_name, rh_name, td_name]

dataframe_temp=dataframe.copy()
dataframe_temp.loc[~dataframe['month'].isin(valid_months), cols_to_nan] = np.nan

temp_to_agg = temp_name
#%% Monthly temperature

monthly_temp = dataframe_temp.resample('ME', on=time_name)[temp_to_agg].agg(['mean', 'min', 'max']).reset_index()
monthly_temp['month'] = pd.to_datetime(monthly_temp[time_name]).dt.to_period('M')
monthly_temp['month'] = monthly_temp['month'].dt.to_timestamp()

# Mask of rows where all values are non-NaN
monthly_temp_no_nan = monthly_temp[['mean', 'min', 'max']].notna().all(axis=1)

# x and y values using only valid data
monthly_x = mdates.date2num(monthly_temp.loc[monthly_temp_no_nan, time_name])
monthly_y_avg = monthly_temp.loc[monthly_temp_no_nan, 'mean']
monthly_y_min = monthly_temp.loc[monthly_temp_no_nan, 'min']
monthly_y_max = monthly_temp.loc[monthly_temp_no_nan, 'max']

# Linear regression
monthly_result_avg = linregress(monthly_x, monthly_y_avg)
monthly_result_min = linregress(monthly_x, monthly_y_min)
monthly_result_max = linregress(monthly_x, monthly_y_max)

# Compute trends (on full time axis)
monthly_x_full = mdates.date2num(monthly_temp[time_name])
monthly_trend_avg = monthly_result_avg.slope * monthly_x_full + monthly_result_avg.intercept
monthly_trend_min = monthly_result_min.slope * monthly_x_full + monthly_result_min.intercept
monthly_trend_max = monthly_result_max.slope * monthly_x_full + monthly_result_max.intercept

# Optional: annualized slope
monthly_slope_avg_per_year = monthly_result_avg.slope * 365
monthly_slope_min_per_year = monthly_result_min.slope * 365
monthly_slope_max_per_year = monthly_result_max.slope * 365

# Monthly variance
monthly_variance_avg = (monthly_result_avg.stderr**2) * len(monthly_y_avg)
monthly_variance_min = (monthly_result_min.stderr**2) * len(monthly_y_min)
monthly_variance_max = (monthly_result_max.stderr**2) * len(monthly_y_max)

#%% Weekly temperature

weekly_temp = dataframe_temp.resample('W', on=time_name)[temp_to_agg].agg(['mean', 'min', 'max']).reset_index()
weekly_temp['week'] = pd.to_datetime(weekly_temp[time_name]).dt.to_period('W')
weekly_temp['week'] = weekly_temp['week'].dt.to_timestamp()

# Mask of rows where all values are non-NaN
weekly_temp_no_nan = weekly_temp[['mean', 'min', 'max']].notna().all(axis=1)

# x and y values using only valid data
weekly_x = mdates.date2num(weekly_temp.loc[weekly_temp_no_nan, time_name])
weekly_y_avg = weekly_temp.loc[weekly_temp_no_nan, 'mean']
weekly_y_min = weekly_temp.loc[weekly_temp_no_nan, 'min']
weekly_y_max = weekly_temp.loc[weekly_temp_no_nan, 'max']

# Linear regression
weekly_result_avg = linregress(weekly_x, weekly_y_avg)
weekly_result_min = linregress(weekly_x, weekly_y_min)
weekly_result_max = linregress(weekly_x, weekly_y_max)

# Compute trends (on full time axis)
weekly_x_full = mdates.date2num(weekly_temp[time_name])
weekly_trend_avg = weekly_result_avg.slope * weekly_x_full + weekly_result_avg.intercept
weekly_trend_min = weekly_result_min.slope * weekly_x_full + weekly_result_min.intercept
weekly_trend_max = weekly_result_max.slope * weekly_x_full + weekly_result_max.intercept

# Optional: annualized slope (assuming 52 weeks per year)
weekly_slope_avg_per_year = weekly_result_avg.slope * 365
weekly_slope_min_per_year = weekly_result_min.slope * 365
weekly_slope_max_per_year = weekly_result_max.slope * 365

# Weekly variance
weekly_variance_avg = (weekly_result_avg.stderr**2) * len(weekly_y_avg)
weekly_variance_min = (weekly_result_min.stderr**2) * len(weekly_y_min)
weekly_variance_max = (weekly_result_max.stderr**2) * len(weekly_y_max)

#%% Daily temperature

daily_temp = dataframe_temp.resample('D', on=time_name)[temp_to_agg].agg(['mean', 'min', 'max']).reset_index()
daily_temp['day'] = pd.to_datetime(daily_temp[time_name]).dt.to_period('D')
daily_temp['day'] = daily_temp['day'].dt.to_timestamp()

# Mask of rows where all values are non-NaN
daily_temp_no_nan = daily_temp[['mean', 'min', 'max']].notna().all(axis=1)

# x and y values using only valid data
daily_x = mdates.date2num(daily_temp.loc[daily_temp_no_nan, time_name])
daily_y_avg = daily_temp.loc[daily_temp_no_nan, 'mean']
daily_y_min = daily_temp.loc[daily_temp_no_nan, 'min']
daily_y_max = daily_temp.loc[daily_temp_no_nan, 'max']

# Linear regression
daily_result_avg = linregress(daily_x, daily_y_avg)
daily_result_min = linregress(daily_x, daily_y_min)
daily_result_max = linregress(daily_x, daily_y_max)

# Compute trends (on full time axis)
daily_x_full = mdates.date2num(daily_temp[time_name])
daily_trend_avg = daily_result_avg.slope * daily_x_full + daily_result_avg.intercept
daily_trend_min = daily_result_min.slope * daily_x_full + daily_result_min.intercept
daily_trend_max = daily_result_max.slope * daily_x_full + daily_result_max.intercept

# Annualized slope (365 days per year)
daily_slope_avg_per_year = daily_result_avg.slope * 365
daily_slope_min_per_year = daily_result_min.slope * 365
daily_slope_max_per_year = daily_result_max.slope * 365

# Daily variance
daily_variance_avg = (daily_result_avg.stderr**2) * len(daily_y_avg)
daily_variance_min = (daily_result_min.stderr**2) * len(daily_y_min)
daily_variance_max = (daily_result_max.stderr**2) * len(daily_y_max)

###############################################################################

#%% Monthly plot

plt.figure(figsize=(12,6))
plt.plot(monthly_temp['month'], monthly_temp['mean'], label='Average Temperature',color='black',linewidth=2)
plt.plot(monthly_temp['month'], monthly_temp['min'], label='Minimum Temperature', color='blue', linewidth=0.5)
plt.plot(monthly_temp['month'], monthly_temp['max'], label='Maximum Temperature', color='red', linewidth=0.5)

plt.plot(monthly_temp[time_name], monthly_trend_avg, label=f'Trend Avg ({monthly_slope_avg_per_year:.2f}°C/yr)', color='black', linestyle='dashed')
plt.plot(monthly_temp[time_name], monthly_trend_min, label=f'Trend Min ({monthly_slope_min_per_year:.2f}°C/yr)', color='blue', linestyle='dashed')
plt.plot(monthly_temp[time_name], monthly_trend_max, label=f'Trend Max ({monthly_slope_max_per_year:.2f}°C/yr)', color='red', linestyle='dashed')

ax = plt.gca()
for period in bad_months:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)

plt.title('Monthly Temperature Summary')
plt.xlabel('Time')
plt.ylabel('Temperature')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.grid('both')
plt.show()


print("\n" + " -- Monthly --")
print(f"Average Temp: slope={monthly_result_avg.slope:.6f}, R²={monthly_result_avg.rvalue**2:.4f}, p={monthly_result_avg.pvalue:.4g}, stderr={monthly_result_avg.stderr:.4f}")
print(f"Min Temp:     slope={monthly_result_min.slope:.6f}, R²={monthly_result_min.rvalue**2:.4f}, p={monthly_result_min.pvalue:.4g}, stderr={monthly_result_min.stderr:.4f}")
print(f"Max Temp:     slope={monthly_result_max.slope:.6f}, R²={monthly_result_max.rvalue**2:.4f}, p={monthly_result_max.pvalue:.4g}, stderr={monthly_result_max.stderr:.4f}")

#%% Weekly plot

plt.figure(figsize=(12,6))
plt.plot(weekly_temp['week'], weekly_temp['mean'], label='Average Temperature', color='black', linewidth=2)
plt.plot(weekly_temp['week'], weekly_temp['min'], label='Minimum Temperature', color='blue', linewidth=0.5)
plt.plot(weekly_temp['week'], weekly_temp['max'], label='Maximum Temperature', color='red', linewidth=0.5)

plt.plot(weekly_temp[time_name], weekly_trend_avg, label=f'Weekly Trend Avg ({weekly_slope_avg_per_year:.2f}°C/yr)', color='black', linestyle='dotted')
plt.plot(weekly_temp[time_name], weekly_trend_min, label=f'Weekly Trend Min ({weekly_slope_min_per_year:.2f}°C/yr)', color='blue', linestyle='dotted')
plt.plot(weekly_temp[time_name], weekly_trend_max, label=f'Weekly Trend Max ({weekly_slope_max_per_year:.2f}°C/yr)', color='red', linestyle='dotted')

ax = plt.gca()
for period in bad_months:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)

plt.title('Weekly Temperature Summary')
plt.xlabel('Time')
plt.ylabel('Temperature')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.grid('both')
plt.show()

print("\n" +" -- Weekly --")
print(f"Average Temp: slope={weekly_result_avg.slope:.6f}, R²={weekly_result_avg.rvalue**2:.4f}, p={weekly_result_avg.pvalue:.4g}, stderr={weekly_result_avg.stderr:.4f}")
print(f"Min Temp:     slope={weekly_result_min.slope:.6f}, R²={weekly_result_min.rvalue**2:.4f}, p={weekly_result_min.pvalue:.4g}, stderr={weekly_result_min.stderr:.4f}")
print(f"Max Temp:     slope={weekly_result_max.slope:.6f}, R²={weekly_result_max.rvalue**2:.4f}, p={weekly_result_max.pvalue:.4g}, stderr={weekly_result_max.stderr:.4f}")

#%% Daily plot

plt.figure(figsize=(12,6))
plt.plot(daily_temp['day'], daily_temp['mean'], label='Average Temperature',color='black',linewidth=2)
plt.plot(daily_temp['day'], daily_temp['min'], label='Minimum Temperature', color='blue', linewidth=0.5)
plt.plot(daily_temp['day'], daily_temp['max'], label='Maximum Temperature', color='red', linewidth=0.5)

plt.plot(daily_temp[time_name], daily_trend_avg, label=f'Daily Trend Avg ({daily_slope_avg_per_year:.2f}°C/yr, R²={daily_result_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
plt.plot(daily_temp[time_name], daily_trend_min, label=f'Daily Trend Min ({daily_slope_min_per_year:.2f}°C/yr, R²={daily_result_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
plt.plot(daily_temp[time_name], daily_trend_max, label=f'Daily Trend Max ({daily_slope_max_per_year:.2f}°C/yr, R²={daily_result_max.rvalue**2:.2f})', color='red', linestyle='dashed')

ax = plt.gca()

for period in bad_months:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)
    
plt.title('Daily Temperature Summary')
plt.xlabel('Time')
plt.ylabel('Temperature')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.legend()
plt.grid('both')
plt.show()    
  
print("\n" +" -- Daily --")
print(f"Average Temp: slope={daily_result_avg.slope:.6f}, R²={daily_result_avg.rvalue**2:.4f}, p={daily_result_avg.pvalue:.4g}, stderr={daily_result_avg.stderr:.4f}")
print(f"Min Temp:     slope={daily_result_min.slope:.6f}, R²={daily_result_min.rvalue**2:.4f}, p={daily_result_min.pvalue:.4g}, stderr={daily_result_min.stderr:.4f}")
print(f"Max Temp:     slope={daily_result_max.slope:.6f}, R²={daily_result_max.rvalue**2:.4f}, p={daily_result_max.pvalue:.4g}, stderr={daily_result_max.stderr:.4f}") 


#%% Remove seasonality


# DOES NOT WORK WITH GAP. DETERMINE SEASONALITY ON INTERVAL BEFORE GAP, THEN EXTRAPOLATE TO COVER THE GAP. AND THEN DESEASONALIZE
# OVER THE NOW GAP-LESS FULL INTERVAL

#components = seasonal_decompose(monthly_temp['mean'], model='additive')
#components.plot()

#monthly_summary_T['MonthPeriod'] = monthly_summary_T[time_name].dt.to_period('M')
# filter rows to keep valid data

# Join Missing_Ratio info from missing_ratio_monthly
#monthly_summary_T = monthly_summary_T.join(
    #missing_ratio_monthly,
    #on='MonthPeriod'










