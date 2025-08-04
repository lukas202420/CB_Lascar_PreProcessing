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
from matplotlib.lines import Line2D

#%% How many files to merge?

files_to_merge=input("How many files do you wish to merge? \n")
print("\n")
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

# Set threshold: minimum number of valid data points per month
month_min_points = 20 * 24  # e.g. 20 days worth of hourly data (20 days * 24 hours)
# Set threshold: minimum number of valid data points per week
week_min_points = 5 * 24  # e.g. 5 days * 24
# Set threshold: minimum number of valid data points per day
day_min_points = 20  # e.g. 20 hours a day to be valid


#%% Selecting the original data file

# create a root window and hide it
root = tk.Tk()
root.withdraw()

# open file dialog, get the selected file path
file_path = filedialog.askopenfilename()

name_of_file=os.path.basename(file_path)
name_of_file_no_csv=os.path.splitext(name_of_file)[0]

# prints selected file
print("Selected file:", file_path + "\n")

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

#%% correct the rh and td values

# Create a mask where RH > 100
over_100_rh = dataframe[rh_name] > 100

# Cap RH at 100
dataframe.loc[over_100_rh, rh_name] = 100

# Set Td = T where RH was over 100
dataframe.loc[over_100_rh, td_name] = dataframe.loc[over_100_rh, temp_name]


#%% check if valid month

dataframe['month']=dataframe.index.to_period('M')

# Actual present timestamps per month
actual_counts_monthly = dataframe.groupby('month')['present'].sum()

# Filter valid and bad months
valid_months = actual_counts_monthly[actual_counts_monthly >= month_min_points].index
bad_months = actual_counts_monthly[actual_counts_monthly < month_min_points].index

#%% check if valid week

dataframe['week']=dataframe.index.to_period('W')

# Actual present timestamps per month
actual_counts_weekly = dataframe.groupby('week')['present'].sum()

# Filter valid and bad months
valid_weeks = actual_counts_weekly[actual_counts_weekly >= week_min_points].index
bad_weeks = actual_counts_weekly[actual_counts_weekly < week_min_points].index

#%% check if valid day

dataframe['day']=dataframe.index.to_period('D')

# Actual present timestamps per month
actual_counts_daily = dataframe.groupby('day')['present'].sum()

# Filter valid and bad months
valid_days = actual_counts_daily[actual_counts_daily >= day_min_points].index
bad_days = actual_counts_daily[actual_counts_daily < day_min_points].index

#%% Filetring out the bad months from the dataframe

cols_to_nan = [temp_name, rh_name, td_name]

dataframe_temp_monthly = dataframe.copy()
dataframe_temp_monthly.loc[~dataframe['month'].isin(valid_months), cols_to_nan] = np.nan
dataframe_rh_monthly = dataframe.copy()
dataframe_rh_monthly.loc[~dataframe['month'].isin(valid_months), [rh_name]] = np.nan

#%% Filetring out the bad weeks from the dataframe

dataframe_temp_weekly = dataframe.copy()
dataframe_temp_weekly.loc[~dataframe['week'].isin(valid_weeks), cols_to_nan] = np.nan
dataframe_rh_weekly = dataframe.copy()
dataframe_rh_weekly.loc[~dataframe['week'].isin(valid_weeks), [rh_name]] = np.nan

#%% Filetring out the bad days from the dataframe

dataframe_temp_daily = dataframe.copy()
dataframe_temp_daily.loc[~dataframe['day'].isin(valid_days), cols_to_nan] = np.nan
dataframe_rh_daily = dataframe.copy()
dataframe_rh_daily.loc[~dataframe['day'].isin(valid_days), [rh_name]] = np.nan

temp_to_agg=temp_name
rh_to_agg=rh_name


#%% Select time window to plot

start_date_window=input("What is the date you want to start your plot on? (YYYY-MM-DD) \nTo select the entirety of the range, type 'all' \n")

if (start_date_window!='all'):
    end_date_window=input("What is the date you want to end your plot on? (YYYY-MM-DD) \n")
else:
    start_date_window=dataframe.index.min()
    end_date_window=dataframe.index.max()

#%% Monthly temperature

monthly_temp = dataframe_temp_monthly.resample('ME', on=time_name)[temp_to_agg].agg(['mean', 'min', 'max']).reset_index()
monthly_temp['month'] = pd.to_datetime(monthly_temp[time_name]).dt.to_period('M')
monthly_temp['month'] = monthly_temp['month'].dt.to_timestamp()

# Only keep rows where mean, min, max are all present
monthly_temp_no_nan = monthly_temp[['mean', 'min', 'max']].notna().all(axis=1)

# Select valid rows
df_valid = monthly_temp.loc[monthly_temp_no_nan].copy()

# Reference date for time zero
reference_date = df_valid[time_name].min()

# Convert datetime to days since reference
df_valid['days_since_start'] = (df_valid[time_name] - reference_date).dt.days

# Generate full trendline values using full x-axis (but same slope/intercept)
monthly_temp['days_since_start'] = (monthly_temp[time_name] - reference_date).dt.days
monthly_temp = monthly_temp[monthly_temp['days_since_start'] >= 0]

# Linear regression on average, min, max
monthly_result_avg = linregress(df_valid['days_since_start'], df_valid['mean'])
monthly_result_min = linregress(df_valid['days_since_start'], df_valid['min'])
monthly_result_max = linregress(df_valid['days_since_start'], df_valid['max'])

monthly_temp = monthly_temp[monthly_temp['days_since_start'] >= 0]

monthly_trend_avg = monthly_result_avg.slope * monthly_temp['days_since_start'] + monthly_result_avg.intercept
monthly_trend_min = monthly_result_min.slope * monthly_temp['days_since_start'] + monthly_result_min.intercept
monthly_trend_max = monthly_result_max.slope * monthly_temp['days_since_start'] + monthly_result_max.intercept


slope_avg_per_year = monthly_result_avg.slope * 365
slope_min_per_year = monthly_result_min.slope * 365
slope_max_per_year = monthly_result_max.slope * 365



#%% Weekly temperature

weekly_temp = dataframe_temp_weekly.resample('W', on=time_name)[temp_to_agg].agg(['mean', 'min', 'max']).reset_index()
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

daily_temp = dataframe_temp_daily.resample('D', on=time_name)[temp_to_agg].agg(['mean', 'min', 'max']).reset_index()
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

#%% Monthly plot temperature

#def plot_monthly_temp():
plt.figure(figsize=(12, 6))

plt.plot(monthly_temp['month'], monthly_temp['mean'], label='Avg Temp', color='black', linewidth=2)
plt.plot(monthly_temp['month'], monthly_temp['min'], label='Min Temp', color='blue', linewidth=0.5)
plt.plot(monthly_temp['month'], monthly_temp['max'], label='Max Temp', color='red', linewidth=0.5)

plt.plot(monthly_temp['month'], monthly_trend_avg, '--', label=f'Trend Avg ({slope_avg_per_year:.2f}°C/yr)', color='black')
plt.plot(monthly_temp['month'], monthly_trend_min, '--', label=f'Trend Min ({slope_min_per_year:.2f}°C/yr)', color='blue')
plt.plot(monthly_temp['month'], monthly_trend_max, '--', label=f'Trend Max ({slope_max_per_year:.2f}°C/yr)', color='red')

ax = plt.gca()
for period in bad_months:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)

plt.title('Monthly Temperature Summary')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
plt.subplots_adjust(bottom=0.3)

ax.set_xlim(pd.Timestamp(start_date_window), pd.Timestamp(end_date_window))
plt.grid(True)
plt.show()

print("\n-- Monthly Temperature Trend --")
print(f"Avg: slope = {slope_avg_per_year:.2f} °C/yr, R² = {monthly_result_avg.rvalue**2:.4f}, p = {monthly_result_avg.pvalue:.4g}")
print(f"Min: slope = {slope_min_per_year:.2f} °C/yr, R² = {monthly_result_min.rvalue**2:.4f}, p = {monthly_result_min.pvalue:.4g}")
print(f"Max: slope = {slope_max_per_year:.2f} °C/yr, R² = {monthly_result_max.rvalue**2:.4f}, p = {monthly_result_max.pvalue:.4g}")


#%% Weekly plot temperature

#def plot_weekly_temp():
plt.figure(figsize=(12,6))
plt.plot(weekly_temp['week'], weekly_temp['mean'], label='Average Temperature', color='black', linewidth=2)
plt.plot(weekly_temp['week'], weekly_temp['min'], label='Minimum Temperature', color='blue', linewidth=0.5)
plt.plot(weekly_temp['week'], weekly_temp['max'], label='Maximum Temperature', color='red', linewidth=0.5)

plt.plot(weekly_temp[time_name], weekly_trend_avg, label=f'Weekly Trend Avg ({weekly_slope_avg_per_year:.2f}°C/yr)', color='black', linestyle='dotted')
plt.plot(weekly_temp[time_name], weekly_trend_min, label=f'Weekly Trend Min ({weekly_slope_min_per_year:.2f}°C/yr)', color='blue', linestyle='dotted')
plt.plot(weekly_temp[time_name], weekly_trend_max, label=f'Weekly Trend Max ({weekly_slope_max_per_year:.2f}°C/yr)', color='red', linestyle='dotted')

ax = plt.gca()
for period in bad_weeks:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)

plt.title('Weekly Temperature Summary')
plt.xlabel('Time')
plt.ylabel('Temperature')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label='Average Temp'),
    Line2D([0], [0], color='black', linestyle='dashed', label=f'Trend Avg ({weekly_slope_avg_per_year:.2f} °C/yr)'),
    Line2D([0], [0], color='blue', linewidth=0.5, label='Minimum Temp'),
    Line2D([0], [0], color='blue', linestyle='dashed', label=f'Trend Min ({weekly_slope_min_per_year:.2f} °C/yr)'),
    Line2D([0], [0], color='red', linewidth=0.5, label='Maximum Temp'),
    Line2D([0], [0], color='red', linestyle='dashed', label=f'Trend Max ({weekly_slope_max_per_year:.2f} °C/yr)'),
]

plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=False
)
plt.subplots_adjust(bottom=0.35)

ax.set_xlim(pd.Timestamp(start_date_window), pd.Timestamp(end_date_window))

plt.grid('both')
plt.show()

print("\n" +" -- Weekly Temperature --")
print(f"Average Temp: slope={weekly_slope_avg_per_year:.6f} °C/yr, R²={weekly_result_avg.rvalue**2:.4f}, p={weekly_result_avg.pvalue:.4g}, stderr={weekly_result_avg.stderr:.4f}")
print(f"Min Temp:     slope={weekly_slope_min_per_year:.6f} °C/yr, R²={weekly_result_min.rvalue**2:.4f}, p={weekly_result_min.pvalue:.4g}, stderr={weekly_result_min.stderr:.4f}")
print(f"Max Temp:     slope={weekly_slope_max_per_year:.6f} °C/yr, R²={weekly_result_max.rvalue**2:.4f}, p={weekly_result_max.pvalue:.4g}, stderr={weekly_result_max.stderr:.4f}")

#%% Daily plot temperature

#def plot_daily_temp():
plt.figure(figsize=(12,6))
plt.plot(daily_temp['day'], daily_temp['mean'], label='Average Temperature',color='black',linewidth=2)
plt.plot(daily_temp['day'], daily_temp['min'], label='Minimum Temperature', color='blue', linewidth=0.5)
plt.plot(daily_temp['day'], daily_temp['max'], label='Maximum Temperature', color='red', linewidth=0.5)

plt.plot(daily_temp[time_name], daily_trend_avg, label=f'Daily Trend Avg ({daily_slope_avg_per_year:.2f}°C/yr, R²={daily_result_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
plt.plot(daily_temp[time_name], daily_trend_min, label=f'Daily Trend Min ({daily_slope_min_per_year:.2f}°C/yr, R²={daily_result_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
plt.plot(daily_temp[time_name], daily_trend_max, label=f'Daily Trend Max ({daily_slope_max_per_year:.2f}°C/yr, R²={daily_result_max.rvalue**2:.2f})', color='red', linestyle='dashed')

ax = plt.gca()

for period in bad_days:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)
    
plt.title('Daily Temperature Summary')
plt.xlabel('Time')
plt.ylabel('Temperature')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label='Average Temp'),
    Line2D([0], [0], color='black', linestyle='dashed', label=f'Trend Avg ({daily_slope_avg_per_year:.2f} °C/yr)'),
    Line2D([0], [0], color='blue', linewidth=0.5, label='Minimum Temp'),
    Line2D([0], [0], color='blue', linestyle='dashed', label=f'Trend Min ({daily_slope_min_per_year:.2f} °C/yr)'),
    Line2D([0], [0], color='red', linewidth=0.5, label='Maximum Temp'),
    Line2D([0], [0], color='red', linestyle='dashed', label=f'Trend Max ({daily_slope_max_per_year:.2f} °C/yr)'),
]

plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=False
)
plt.subplots_adjust(bottom=0.35)

ax.set_xlim(pd.Timestamp(start_date_window), pd.Timestamp(end_date_window))

plt.grid('both')
plt.show()    
  
print("\n" +" -- Daily Temperature --")
print(f"Average Temp: slope={daily_slope_avg_per_year:.6f} °C/yr, R²={daily_result_avg.rvalue**2:.4f}, p={daily_result_avg.pvalue:.4g}, stderr={daily_result_avg.stderr:.4f}")
print(f"Min Temp:     slope={daily_slope_min_per_year:.6f} °C/yr, R²={daily_result_min.rvalue**2:.4f}, p={daily_result_min.pvalue:.4g}, stderr={daily_result_min.stderr:.4f}")
print(f"Max Temp:     slope={daily_slope_max_per_year:.6f} °C/yr, R²={daily_result_max.rvalue**2:.4f}, p={daily_result_max.pvalue:.4g}, stderr={daily_result_max.stderr:.4f}") 



#%% Temperature histogram

def plot_hist_temp():
    plt.figure(figsize=(12,6))
    
    hist_temp_df=dataframe.loc[start_date_window : end_date_window]
    
    # Get min and max values of the temperature column
    min_val = int(np.floor(dataframe[temp_name].min()))
    max_val = int(np.ceil(dataframe[temp_name].max()))
    # Create integer bins
    bins = np.arange(min_val, max_val + 1)
    
    plt.hist(hist_temp_df[temp_name], density=False, bins=bins)
    
    plt.title('Temperature Histogram')
    plt.xlabel('Temperature')
    plt.ylabel('# occurences')
    
    plt.xticks(bins)
    plt.grid(True, which='major', axis='x')  
    plt.grid(True, which='major', axis='y')  
    
    plt.show()

#%% Night vs day temp 

night_range=[23, 0, 1, 2, 3, 4, 5]

def plot_avg_day_night_temp_all():
    df = dataframe.copy()

    # Filter to date range
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    
    # For monthly graphs
    df.loc[df.index.to_period('M').isin(bad_months), temp_name] = np.nan
    
    # For weekly graphs
    df.loc[df.index.to_period('W').isin(bad_weeks), temp_name] = np.nan
    
    # For daily graphs
    df.loc[df.index.to_period('D').isin(bad_days), temp_name] = np.nan

    # Classify as night or day
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    # ---- DAILY AVERAGE ----
    daily = df.groupby(['date', 'time_of_day'])[temp_name].mean().unstack()

    # ---- WEEKLY AVERAGE ----
    df['week'] = df.index.to_period('W').start_time
    weekly = df.groupby(['week', 'time_of_day'])[temp_name].mean().unstack()

    # ---- MONTHLY AVERAGE ----
    df['month'] = df.index.to_period('M').start_time
    monthly = df.groupby(['month', 'time_of_day'])[temp_name].mean().unstack()

    # ---- PLOTTING ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharey=True)

    # Daily
    axes[0].plot(daily.index, daily['day'], label='Day Avg', color='orange')
    axes[0].plot(daily.index, daily['night'], label='Night Avg', color='blue')
    axes[0].set_title('Daily Average Day vs Night Temperature')
    axes[0].legend()
    axes[0].grid(True)
    for period in bad_days:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        axes[0].axvspan(start, end, color='gray', alpha=0.2)
    
    # Weekly
    axes[1].plot(weekly.index, weekly['day'], label='Day Avg', color='orange')
    axes[1].plot(weekly.index, weekly['night'], label='Night Avg', color='blue')
    axes[1].set_title('Weekly Average Day vs Night Temperature')
    axes[1].legend()
    axes[1].grid(True)
    for period in bad_weeks:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        axes[1].axvspan(start, end, color='gray', alpha=0.2)
    
    # Monthly
    axes[2].plot(monthly.index, monthly['day'], label='Day Avg', color='orange')
    axes[2].plot(monthly.index, monthly['night'], label='Night Avg', color='blue')
    axes[2].set_title('Monthly Average Day vs Night Temperature')
    axes[2].legend()
    axes[2].grid(True)
    for period in bad_months:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        axes[2].axvspan(start, end, color='gray', alpha=0.2)

def plot_avg_day_night_temp_daily():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df.loc[df.index.to_period('D').isin(bad_days), temp_name] = np.nan
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    daily = df.groupby(['date', 'time_of_day'])[temp_name].mean().unstack()

    plt.figure(figsize=(14,6))
    plt.plot(daily.index, daily['day'], label='Day Avg', color='orange')
    plt.plot(daily.index, daily['night'], label='Night Avg', color='blue')
    plt.title('Daily Average Day vs Night Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    ax = plt.gca() 
    for period in bad_days:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_avg_day_night_temp_weekly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['week'] = df.index.to_period('W').start_time
    df.loc[df.index.to_period('W').isin(bad_weeks), temp_name] = np.nan
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    weekly = df.groupby(['week', 'time_of_day'])[temp_name].mean().unstack()

    plt.figure(figsize=(14,6))
    plt.plot(weekly.index, weekly['day'], label='Day Avg', color='orange')
    plt.plot(weekly.index, weekly['night'], label='Night Avg', color='blue')
    plt.title('Weekly Average Day vs Night Temperature')
    plt.xlabel('Week')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    ax = plt.gca() 
    for period in bad_weeks:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_avg_day_night_temp_monthly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['month'] = df.index.to_period('M').start_time
    df.loc[df.index.to_period('M').isin(bad_months), temp_name] = np.nan
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    monthly = df.groupby(['month', 'time_of_day'])[temp_name].mean().unstack()

    plt.figure(figsize=(14,6))
    plt.plot(monthly.index, monthly['day'], label='Day Avg', color='orange')
    plt.plot(monthly.index, monthly['night'], label='Night Avg', color='blue')
    plt.title('Monthly Average Day vs Night Temperature')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    ax = plt.gca() 
    for period in bad_months:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_daily_day_night_temp_difference():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    daily = df.groupby(['date', 'time_of_day'])[temp_name].mean().unstack()
    daily_diff = daily['day'] - daily['night']

    plt.figure(figsize=(14,6))
    plt.plot(daily_diff.index, daily_diff, color='purple')
    plt.title('Daily Average Day-Night Temperature Difference')
    plt.xlabel('Date')
    plt.ylabel('Temperature Difference (°C)')
    plt.grid(True)

    ax = plt.gca()
    for period in bad_days:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.show()

def plot_weekly_day_night_temp_difference():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['week'] = df.index.to_period('W').start_time
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    weekly = df.groupby(['week', 'time_of_day'])[temp_name].mean().unstack()
    weekly_diff = weekly['day'] - weekly['night']

    plt.figure(figsize=(14,6))
    plt.plot(weekly_diff.index, weekly_diff, color='purple')
    plt.title('Weekly Average Day-Night Temperature Difference')
    plt.xlabel('Week')
    plt.ylabel('Temperature Difference (°C)')
    plt.grid(True)

    ax = plt.gca()
    for period in bad_weeks:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.show()

def plot_monthly_day_night_temp_difference():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['month'] = df.index.to_period('M').start_time
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    monthly = df.groupby(['month', 'time_of_day'])[temp_name].mean().unstack()
    monthly_diff = monthly['day'] - monthly['night']

    plt.figure(figsize=(14,6))
    plt.plot(monthly_diff.index, monthly_diff, color='purple')
    plt.title('Monthly Average Day-Night Temperature Difference')
    plt.xlabel('Month')
    plt.ylabel('Temperature Difference (°C)')
    plt.grid(True)

    ax = plt.gca()
    for period in bad_months:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.show()

#%% max-avg, avg-min

def plot_temp_deviation_all():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]

    df['date'] = df.index.date
    df['week'] = df.index.to_period('W').start_time
    df['month'] = df.index.to_period('M').start_time
    df.loc[df.index.to_period('M').isin(bad_months), temp_name] = np.nan
    df.loc[df.index.to_period('W').isin(bad_weeks), temp_name] = np.nan
    df.loc[df.index.to_period('D').isin(bad_days), temp_name] = np.nan
    # ---- DAILY ----
    daily_stats = df.groupby('date')[temp_name].agg(['mean', 'max', 'min'])
    daily_stats['diff_max'] = daily_stats['max'] - daily_stats['mean']
    daily_stats['diff_min'] = daily_stats['mean'] - daily_stats['min']

    # ---- WEEKLY ----
    weekly_stats = df.groupby('week')[temp_name].agg(['mean', 'max', 'min'])
    weekly_stats['diff_max'] = weekly_stats['max'] - weekly_stats['mean']
    weekly_stats['diff_min'] = weekly_stats['mean'] - weekly_stats['min']

    # ---- MONTHLY ----
    monthly_stats = df.groupby('month')[temp_name].agg(['mean', 'max', 'min'])
    monthly_stats['diff_max'] = monthly_stats['max'] - monthly_stats['mean']
    monthly_stats['diff_min'] = monthly_stats['mean'] - monthly_stats['min']

    # ---- PLOTTING ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharey=False)

    # Daily
    axes[0].plot(daily_stats.index, daily_stats['diff_max'], label='Max - Avg', color='red')
    axes[0].plot(daily_stats.index, daily_stats['diff_min'], label='Avg - Min', color='blue')
    axes[0].set_title('Daily Temp Deviation (Max - Avg / Avg - Min)')
    axes[0].legend()
    axes[0].grid(True)
    if 'bad_days' in globals():
        for period in bad_days:
            start = period.to_timestamp()
            end = (period + 1).to_timestamp()
            axes[0].axvspan(start, end, color='gray', alpha=0.15)

    # Weekly
    axes[1].plot(weekly_stats.index, weekly_stats['diff_max'], label='Max - Avg', color='red')
    axes[1].plot(weekly_stats.index, weekly_stats['diff_min'], label='Avg - Min', color='blue')
    axes[1].set_title('Weekly Temp Deviation')
    axes[1].legend()
    axes[1].grid(True)
    if 'bad_weeks' in globals():
        for period in bad_weeks:
            start = period.to_timestamp()
            end = (period + 1).to_timestamp()
            axes[1].axvspan(start, end, color='gray', alpha=0.15)

    # Monthly
    axes[2].plot(monthly_stats.index, monthly_stats['diff_max'], label='Max - Avg', color='red')
    axes[2].plot(monthly_stats.index, monthly_stats['diff_min'], label='Avg - Min', color='blue')
    axes[2].set_title('Monthly Temp Deviation')
    axes[2].legend()
    axes[2].grid(True)
    if 'bad_months' in globals():
        for period in bad_months:
            start = period.to_timestamp()
            end = (period + 1).to_timestamp()
            axes[2].axvspan(start, end, color='gray', alpha=0.15)

    for ax in axes:
        ax.set_ylabel('Temperature Deviation (°C)')
        ax.set_xlabel('Date')

    plt.tight_layout()
    plt.show()

def plot_temp_deviation_daily():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['date'] = df.index.date
    df.loc[df.index.to_period('D').isin(bad_days), temp_name] = np.nan
    daily_stats = df.groupby('date')[temp_name].agg(['mean', 'max', 'min'])
    daily_stats['diff_max'] = daily_stats['max'] - daily_stats['mean']
    daily_stats['diff_min'] = daily_stats['mean'] - daily_stats['min']
    
    plt.figure(figsize=(14, 5))
    plt.plot(daily_stats.index, daily_stats['diff_max'], label='Max - Avg', color='red')
    plt.plot(daily_stats.index, daily_stats['diff_min'], label='Avg - Min', color='blue')
    plt.title('Daily Temp Deviation (Max - Avg / Avg - Min)')
    plt.xlabel('Date')
    plt.ylabel('Temp Deviation (°C)')
    plt.legend()
    plt.grid(True)
    
    if 'bad_days' in globals():
        for period in bad_days:
            start = period.to_timestamp()
            end = (period + 1).to_timestamp()
            plt.axvspan(start, end, color='gray', alpha=0.15)
    
    plt.tight_layout()
    plt.show()

def plot_temp_deviation_weekly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['week'] = df.index.to_period('W').start_time
    df.loc[df.index.to_period('W').isin(bad_weeks), temp_name] = np.nan
    weekly_stats = df.groupby('week')[temp_name].agg(['mean', 'max', 'min'])
    weekly_stats['diff_max'] = weekly_stats['max'] - weekly_stats['mean']
    weekly_stats['diff_min'] = weekly_stats['mean'] - weekly_stats['min']

    plt.figure(figsize=(14, 5))
    plt.plot(weekly_stats.index, weekly_stats['diff_max'], label='Max - Avg', color='red')
    plt.plot(weekly_stats.index, weekly_stats['diff_min'], label='Avg - Min', color='blue')
    plt.title('Weekly Temp Deviation (Max - Avg / Avg - Min)')
    plt.xlabel('Date')
    plt.ylabel('Temp Deviation (°C)')
    plt.legend()
    plt.grid(True)

    if 'bad_weeks' in globals():
        for period in bad_weeks:
            start = period.to_timestamp()
            end = (period + 1).to_timestamp()
            plt.axvspan(start, end, color='gray', alpha=0.15)

    plt.tight_layout()
    plt.show()

def plot_temp_deviation_monthly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['month'] = df.index.to_period('M').start_time
    df.loc[df.index.to_period('M').isin(bad_months), temp_name] = np.nan
    monthly_stats = df.groupby('month')[temp_name].agg(['mean', 'max', 'min'])
    monthly_stats['diff_max'] = monthly_stats['max'] - monthly_stats['mean']
    monthly_stats['diff_min'] = monthly_stats['mean'] - monthly_stats['min']

    plt.figure(figsize=(14, 5))
    plt.plot(monthly_stats.index, monthly_stats['diff_max'], label='Max - Avg', color='red')
    plt.plot(monthly_stats.index, monthly_stats['diff_min'], label='Avg - Min', color='blue')
    plt.title('Monthly Temp Deviation (Max - Avg / Avg - Min)')
    plt.xlabel('Date')
    plt.ylabel('Temp Deviation (°C)')
    plt.legend()
    plt.grid(True)

    if 'bad_months' in globals():
        for period in bad_months:
            start = period.to_timestamp()
            end = (period + 1).to_timestamp()
            plt.axvspan(start, end, color='gray', alpha=0.15)

    plt.tight_layout()
    plt.show()




#%% Monthly RH

monthly_rh = dataframe_rh_monthly.resample('ME', on=time_name)[rh_to_agg].agg(['mean', 'min', 'max']).reset_index()
monthly_rh['month'] = pd.to_datetime(monthly_rh[time_name]).dt.to_period('M')
monthly_rh['month'] = monthly_rh['month'].dt.to_timestamp()

# Mask of rows where all values are non-NaN
monthly_rh_no_nan = monthly_rh[['mean', 'min', 'max']].notna().all(axis=1)

# x and y values using only valid data
monthly_x = mdates.date2num(monthly_rh.loc[monthly_rh_no_nan, time_name])
monthly_y_avg = monthly_rh.loc[monthly_rh_no_nan, 'mean']
monthly_y_min = monthly_rh.loc[monthly_rh_no_nan, 'min']
monthly_y_max = monthly_rh.loc[monthly_rh_no_nan, 'max']

# Linear regression
monthly_result_avg = linregress(monthly_x, monthly_y_avg)
monthly_result_min = linregress(monthly_x, monthly_y_min)
monthly_result_max = linregress(monthly_x, monthly_y_max)

# Compute trends (on full time axis)
monthly_x_full = mdates.date2num(monthly_rh[time_name])
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

#%% Weekly RH

weekly_rh = dataframe_rh_weekly.resample('W', on=time_name)[rh_to_agg].agg(['mean', 'min', 'max']).reset_index()
weekly_rh['week'] = pd.to_datetime(weekly_rh[time_name]).dt.to_period('W')
weekly_rh['week'] = weekly_rh['week'].dt.to_timestamp()

# Mask of rows where all values are non-NaN
weekly_rh_no_nan = weekly_rh[['mean', 'min', 'max']].notna().all(axis=1)

# x and y values using only valid data
weekly_x = mdates.date2num(weekly_rh.loc[weekly_rh_no_nan, time_name])
weekly_y_avg = weekly_rh.loc[weekly_rh_no_nan, 'mean']
weekly_y_min = weekly_rh.loc[weekly_rh_no_nan, 'min']
weekly_y_max = weekly_rh.loc[weekly_rh_no_nan, 'max']

# Linear regression
weekly_result_avg = linregress(weekly_x, weekly_y_avg)
weekly_result_min = linregress(weekly_x, weekly_y_min)
weekly_result_max = linregress(weekly_x, weekly_y_max)

# Compute trends (on full time axis)
weekly_x_full = mdates.date2num(weekly_rh[time_name])
weekly_trend_avg = weekly_result_avg.slope * weekly_x_full + weekly_result_avg.intercept
weekly_trend_min = weekly_result_min.slope * weekly_x_full + weekly_result_min.intercept
weekly_trend_max = weekly_result_max.slope * weekly_x_full + weekly_result_max.intercept

# Optional: annualized slope
weekly_slope_avg_per_year = weekly_result_avg.slope * 365
weekly_slope_min_per_year = weekly_result_min.slope * 365
weekly_slope_max_per_year = weekly_result_max.slope * 365

# Weekly variance
weekly_variance_avg = (weekly_result_avg.stderr**2) * len(weekly_y_avg)
weekly_variance_min = (weekly_result_min.stderr**2) * len(weekly_y_min)
weekly_variance_max = (weekly_result_max.stderr**2) * len(weekly_y_max)

#%% Daily RH

daily_rh = dataframe_rh_daily.resample('D', on=time_name)[rh_to_agg].agg(['mean', 'min', 'max']).reset_index()
daily_rh['day'] = pd.to_datetime(daily_rh[time_name]).dt.to_period('D')
daily_rh['day'] = daily_rh['day'].dt.to_timestamp()

# Mask of rows where all values are non-NaN
daily_rh_no_nan = daily_rh[['mean', 'min', 'max']].notna().all(axis=1)

# x and y values using only valid data
daily_x = mdates.date2num(daily_rh.loc[daily_rh_no_nan, time_name])
daily_y_avg = daily_rh.loc[daily_rh_no_nan, 'mean']
daily_y_min = daily_rh.loc[daily_rh_no_nan, 'min']
daily_y_max = daily_rh.loc[daily_rh_no_nan, 'max']

# Linear regression
daily_result_avg = linregress(daily_x, daily_y_avg)
daily_result_min = linregress(daily_x, daily_y_min)
daily_result_max = linregress(daily_x, daily_y_max)

# Compute trends (on full time axis)
daily_x_full = mdates.date2num(daily_rh[time_name])
daily_trend_avg = daily_result_avg.slope * daily_x_full + daily_result_avg.intercept
daily_trend_min = daily_result_min.slope * daily_x_full + daily_result_min.intercept
daily_trend_max = daily_result_max.slope * daily_x_full + daily_result_max.intercept

# Annualized slope
daily_slope_avg_per_year = daily_result_avg.slope * 365
daily_slope_min_per_year = daily_result_min.slope * 365
daily_slope_max_per_year = daily_result_max.slope * 365

# Daily variance
daily_variance_avg = (daily_result_avg.stderr**2) * len(daily_y_avg)
daily_variance_min = (daily_result_min.stderr**2) * len(daily_y_min)
daily_variance_max = (daily_result_max.stderr**2) * len(daily_y_max)

###############################################################################

#%% Monthly plot rh

#def plot_monthly_rh():
plt.figure(figsize=(12,6))
plt.plot(monthly_rh['month'], monthly_rh['mean'], label='Average RH', color='black', linewidth=2)
plt.plot(monthly_rh['month'], monthly_rh['min'], label='Minimum RH', color='blue', linewidth=0.5)
plt.plot(monthly_rh['month'], monthly_rh['max'], label='Maximum RH', color='red', linewidth=0.5)

plt.plot(monthly_rh[time_name], monthly_trend_avg, label=f'Trend Avg ({monthly_slope_avg_per_year:.2f} %/yr)', color='black', linestyle='dashed')
plt.plot(monthly_rh[time_name], monthly_trend_min, label=f'Trend Min ({monthly_slope_min_per_year:.2f} %/yr)', color='blue', linestyle='dashed')
plt.plot(monthly_rh[time_name], monthly_trend_max, label=f'Trend Max ({monthly_slope_max_per_year:.2f} %/yr)', color='red', linestyle='dashed')

ax = plt.gca()
for period in bad_months:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)

plt.title('Monthly RH Summary')
plt.xlabel('Time')
plt.ylabel('Relative Humidity (%)')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label='Average RH'),
    Line2D([0], [0], color='black', linestyle='dashed', label=f'Trend Avg ({monthly_slope_avg_per_year:.2f} %/yr)'),
    Line2D([0], [0], color='blue', linewidth=0.5, label='Minimum RH'),
    Line2D([0], [0], color='blue', linestyle='dashed', label=f'Trend Min ({monthly_slope_min_per_year:.2f} %/yr)'),
    Line2D([0], [0], color='red', linewidth=0.5, label='Maximum RH'),
    Line2D([0], [0], color='red', linestyle='dashed', label=f'Trend Max ({monthly_slope_max_per_year:.2f} %/yr)'),
]

plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=False
)
plt.subplots_adjust(bottom=0.35)

ax.set_xlim(pd.Timestamp(start_date_window), pd.Timestamp(end_date_window))

plt.grid('both')
plt.show()

print("\n" + " -- Monthly Relative Humidity --")
print(f"Average RH: slope={monthly_slope_avg_per_year:.6f} %/yr, R²={monthly_result_avg.rvalue**2:.4f}, p={monthly_result_avg.pvalue:.4g}, stderr={monthly_result_avg.stderr:.4f}")
print(f"Min RH:     slope={monthly_slope_min_per_year:.6f} %/yr, R²={monthly_result_min.rvalue**2:.4f}, p={monthly_result_min.pvalue:.4g}, stderr={monthly_result_min.stderr:.4f}")
print(f"Max RH:     slope={monthly_slope_max_per_year:.6f} %/yr, R²={monthly_result_max.rvalue**2:.4f}, p={monthly_result_max.pvalue:.4g}, stderr={monthly_result_max.stderr:.4f}")

#%% Weekly plot rh

#def plot_weekly_rh():
plt.figure(figsize=(12,6))
plt.plot(weekly_rh['week'], weekly_rh['mean'], label='Average RH', color='black', linewidth=2)
plt.plot(weekly_rh['week'], weekly_rh['min'], label='Minimum RH', color='blue', linewidth=0.5)
plt.plot(weekly_rh['week'], weekly_rh['max'], label='Maximum RH', color='red', linewidth=0.5)

plt.plot(weekly_rh[time_name], weekly_trend_avg, label=f'Weekly Trend Avg ({weekly_slope_avg_per_year:.2f} %/yr)', color='black', linestyle='dotted')
plt.plot(weekly_rh[time_name], weekly_trend_min, label=f'Weekly Trend Min ({weekly_slope_min_per_year:.2f} %/yr)', color='blue', linestyle='dotted')
plt.plot(weekly_rh[time_name], weekly_trend_max, label=f'Weekly Trend Max ({weekly_slope_max_per_year:.2f} %/yr)', color='red', linestyle='dotted')

ax = plt.gca()
for period in bad_weeks:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)

plt.title('Weekly RH Summary')
plt.xlabel('Time')
plt.ylabel('Relative Humidity (%)')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label='Average RH'),
    Line2D([0], [0], color='black', linestyle='dashed', label=f'Trend Avg ({weekly_slope_avg_per_year:.2f} %/yr)'),
    Line2D([0], [0], color='blue', linewidth=0.5, label='Minimum RH'),
    Line2D([0], [0], color='blue', linestyle='dashed', label=f'Trend Min ({weekly_slope_min_per_year:.2f} %/yr)'),
    Line2D([0], [0], color='red', linewidth=0.5, label='Maximum RH'),
    Line2D([0], [0], color='red', linestyle='dashed', label=f'Trend Max ({weekly_slope_max_per_year:.2f} %/yr)'),
]

plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=False
)
plt.subplots_adjust(bottom=0.35)

ax.set_xlim(pd.Timestamp(start_date_window), pd.Timestamp(end_date_window))

plt.grid('both')
plt.show()

print("\n" + " -- Weekly Relative Humidity --")
print(f"Average RH: slope={weekly_slope_avg_per_year:.6f} %/yr, R²={weekly_result_avg.rvalue**2:.4f}, p={weekly_result_avg.pvalue:.4g}, stderr={weekly_result_avg.stderr:.4f}")
print(f"Min RH:     slope={weekly_slope_min_per_year:.6f} %/yr, R²={weekly_result_min.rvalue**2:.4f}, p={weekly_result_min.pvalue:.4g}, stderr={weekly_result_min.stderr:.4f}")
print(f"Max RH:     slope={weekly_slope_max_per_year:.6f} %/yr, R²={weekly_result_max.rvalue**2:.4f}, p={weekly_result_max.pvalue:.4g}, stderr={weekly_result_max.stderr:.4f}")

#%% Daily plot rh

#def plot_daily_rh():
plt.figure(figsize=(12,6))
plt.plot(daily_rh['day'], daily_rh['mean'], label='Average RH', color='black', linewidth=2)
plt.plot(daily_rh['day'], daily_rh['min'], label='Minimum RH', color='blue', linewidth=0.5)
plt.plot(daily_rh['day'], daily_rh['max'], label='Maximum RH', color='red', linewidth=0.5)

plt.plot(daily_rh[time_name], daily_trend_avg, label=f'Daily Trend Avg ({daily_slope_avg_per_year:.2f} %/yr, R²={daily_result_avg.rvalue**2:.2f})', color='black', linestyle='dashed')
plt.plot(daily_rh[time_name], daily_trend_min, label=f'Daily Trend Min ({daily_slope_min_per_year:.2f} %/yr, R²={daily_result_min.rvalue**2:.2f})', color='blue', linestyle='dashed')
plt.plot(daily_rh[time_name], daily_trend_max, label=f'Daily Trend Max ({daily_slope_max_per_year:.2f} %/yr, R²={daily_result_max.rvalue**2:.2f})', color='red', linestyle='dashed')

ax = plt.gca()
for period in bad_days:
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    ax.axvspan(start, end, color='gray', alpha=0.2)

plt.title('Daily RH Summary')
plt.xlabel('Time')
plt.ylabel('Relative Humidity (%)')
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

legend_elements = [
    Line2D([0], [0], color='black', linewidth=2, label='Average RH'),
    Line2D([0], [0], color='black', linestyle='dashed', label=f'Trend Avg ({daily_slope_avg_per_year:.2f} %/yr)'),
    Line2D([0], [0], color='blue', linewidth=0.5, label='Minimum RH'),
    Line2D([0], [0], color='blue', linestyle='dashed', label=f'Trend Min ({daily_slope_avg_per_year:.2f} %/yr)'),
    Line2D([0], [0], color='red', linewidth=0.5, label='Maximum RH'),
    Line2D([0], [0], color='red', linestyle='dashed', label=f'Trend Max ({daily_slope_avg_per_year:.2f} %/yr)'),
]

plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=False
)
plt.subplots_adjust(bottom=0.35)  # more space for two lines

ax.set_xlim(pd.Timestamp(start_date_window), pd.Timestamp(end_date_window))

plt.grid('both')
plt.show()

print("\n" + " -- Daily Relative Humidity --")
print(f"Average RH: slope={daily_slope_avg_per_year:.6f} %/yr, R²={daily_result_avg.rvalue**2:.4f}, p={daily_result_avg.pvalue:.4g}, stderr={daily_result_avg.stderr:.4f}")
print(f"Min RH:     slope={daily_slope_avg_per_year:.6f} %/yr, R²={daily_result_min.rvalue**2:.4f}, p={daily_result_min.pvalue:.4g}, stderr={daily_result_min.stderr:.4f}")
print(f"Max RH:     slope={daily_slope_avg_per_year:.6f} %/yr, R²={daily_result_max.rvalue**2:.4f}, p={daily_result_max.pvalue:.4g}, stderr={daily_result_max.stderr:.4f}")

#%% RH historgram

def plot_hist_rh():
    plt.figure(figsize=(12,6))
    
    hist_rh_df = dataframe.loc[start_date_window : end_date_window]
    
    # Get min and max values of the RH column
    min_val = int(np.floor(dataframe[rh_name].min()))
    max_val = int(np.ceil(dataframe[rh_name].max()))
    
    # Create integer bins
    bins = np.arange(min_val, max_val + 1)
    
    plt.hist(hist_rh_df[rh_name], density=False, bins=bins)
    
    plt.title('Relative Humidity Histogram')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('# Occurrences')
    
    xticks = np.arange(min_val, max_val + 1, 5)
    plt.xticks(xticks)
    plt.grid(True, which='major', axis='x')  
    plt.grid(True, which='major', axis='y')  
    
    plt.show()













#%% Night vs day RH

night_range = [23, 0, 1, 2, 3, 4, 5]

def plot_avg_day_night_rh_all():
    df = dataframe.copy()

    # Filter to date range
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    # Classify as night or day
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    # ---- DAILY AVERAGE ----
    daily = df.groupby(['date', 'time_of_day'])[rh_name].mean().unstack()

    # ---- WEEKLY AVERAGE ----
    df['week'] = df.index.to_period('W').start_time
    weekly = df.groupby(['week', 'time_of_day'])[rh_name].mean().unstack()

    # ---- MONTHLY AVERAGE ----
    df['month'] = df.index.to_period('M').start_time
    monthly = df.groupby(['month', 'time_of_day'])[rh_name].mean().unstack()

    # ---- PLOTTING ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharey=True)

    # Daily
    axes[0].plot(daily.index, daily['day'], label='Day Avg', color='orange')
    axes[0].plot(daily.index, daily['night'], label='Night Avg', color='blue')
    axes[0].set_title('Daily Average Day vs Night Relative Humidity')
    axes[0].legend()
    axes[0].grid(True)
    for period in bad_days:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        axes[0].axvspan(start, end, color='gray', alpha=0.2)
    
    # Weekly
    axes[1].plot(weekly.index, weekly['day'], label='Day Avg', color='orange')
    axes[1].plot(weekly.index, weekly['night'], label='Night Avg', color='blue')
    axes[1].set_title('Weekly Average Day vs Night Relative Humidity')
    axes[1].legend()
    axes[1].grid(True)
    for period in bad_weeks:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        axes[1].axvspan(start, end, color='gray', alpha=0.2)
    
    # Monthly
    axes[2].plot(monthly.index, monthly['day'], label='Day Avg', color='orange')
    axes[2].plot(monthly.index, monthly['night'], label='Night Avg', color='blue')
    axes[2].set_title('Monthly Average Day vs Night Relative Humidity')
    axes[2].legend()
    axes[2].grid(True)
    for period in bad_months:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        axes[2].axvspan(start, end, color='gray', alpha=0.2)

def plot_avg_day_night_rh_daily():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    daily = df.groupby(['date', 'time_of_day'])[rh_name].mean().unstack()

    plt.figure(figsize=(14,6))
    plt.plot(daily.index, daily['day'], label='Day Avg', color='orange')
    plt.plot(daily.index, daily['night'], label='Night Avg', color='blue')
    plt.title('Daily Average Day vs Night Relative Humidity')
    plt.xlabel('Date')
    plt.ylabel('Relative Humidity (%)')
    plt.legend()
    ax = plt.gca() 
    for period in bad_days:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_avg_day_night_rh_weekly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['week'] = df.index.to_period('W').start_time
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    weekly = df.groupby(['week', 'time_of_day'])[rh_name].mean().unstack()

    plt.figure(figsize=(14,6))
    plt.plot(weekly.index, weekly['day'], label='Day Avg', color='orange')
    plt.plot(weekly.index, weekly['night'], label='Night Avg', color='blue')
    plt.title('Weekly Average Day vs Night Relative Humidity')
    plt.xlabel('Week')
    plt.ylabel('Relative Humidity (%)')
    plt.legend()
    ax = plt.gca() 
    for period in bad_weeks:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_avg_day_night_rh_monthly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['month'] = df.index.to_period('M').start_time
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    monthly = df.groupby(['month', 'time_of_day'])[rh_name].mean().unstack()

    plt.figure(figsize=(14,6))
    plt.plot(monthly.index, monthly['day'], label='Day Avg', color='orange')
    plt.plot(monthly.index, monthly['night'], label='Night Avg', color='blue')
    plt.title('Monthly Average Day vs Night Relative Humidity')
    plt.xlabel('Month')
    plt.ylabel('Relative Humidity (%)')
    plt.legend()
    ax = plt.gca() 
    for period in bad_months:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_daily_day_night_rh_difference():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['date'] = df.index.date
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    daily = df.groupby(['date', 'time_of_day'])[rh_name].mean().unstack()
    daily_diff = daily['day'] - daily['night']

    plt.figure(figsize=(14,6))
    plt.plot(daily_diff.index, daily_diff, color='purple')
    plt.title('Daily Average Day-Night Relative Humidity Difference')
    plt.xlabel('Date')
    plt.ylabel('Relative Humidity Difference (%)')
    plt.grid(True)

    ax = plt.gca()
    for period in bad_days:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.show()

def plot_weekly_day_night_rh_difference():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['week'] = df.index.to_period('W').start_time
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    weekly = df.groupby(['week', 'time_of_day'])[rh_name].mean().unstack()
    weekly_diff = weekly['day'] - weekly['night']

    plt.figure(figsize=(14,6))
    plt.plot(weekly_diff.index, weekly_diff, color='purple')
    plt.title('Weekly Average Day-Night Relative Humidity Difference')
    plt.xlabel('Week')
    plt.ylabel('Relative Humidity Difference (%)')
    plt.grid(True)

    ax = plt.gca()
    for period in bad_weeks:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.show()

def plot_monthly_day_night_rh_difference():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['hour'] = df.index.hour
    df['month'] = df.index.to_period('M').start_time
    df['time_of_day'] = np.where(df['hour'].isin(night_range), 'night', 'day')

    monthly = df.groupby(['month', 'time_of_day'])[rh_name].mean().unstack()
    monthly_diff = monthly['day'] - monthly['night']

    plt.figure(figsize=(14,6))
    plt.plot(monthly_diff.index, monthly_diff, color='purple')
    plt.title('Monthly Average Day-Night Relative Humidity Difference')
    plt.xlabel('Month')
    plt.ylabel('Relative Humidity Difference (%)')
    plt.grid(True)

    ax = plt.gca()
    for period in bad_months:
        start = period.to_timestamp()
        end = (period + 1).to_timestamp()
        ax.axvspan(start, end, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.show()


#%% max-avg avg-min RH

def plot_rh_deviation_all():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]

    df['date'] = df.index.date
    df['week'] = df.index.to_period('W').start_time
    df['month'] = df.index.to_period('M').start_time

    df.loc[df.index.to_period('M').isin(bad_months), rh_name] = np.nan
    df.loc[df.index.to_period('W').isin(bad_weeks), rh_name] = np.nan
    df.loc[df.index.to_period('D').isin(bad_days), rh_name] = np.nan

    # Daily
    daily_stats = df.groupby('date')[rh_name].agg(['mean', 'max', 'min'])
    daily_stats['diff_max'] = daily_stats['max'] - daily_stats['mean']
    daily_stats['diff_min'] = daily_stats['mean'] - daily_stats['min']

    # Weekly
    weekly_stats = df.groupby('week')[rh_name].agg(['mean', 'max', 'min'])
    weekly_stats['diff_max'] = weekly_stats['max'] - weekly_stats['mean']
    weekly_stats['diff_min'] = weekly_stats['mean'] - weekly_stats['min']

    # Monthly
    monthly_stats = df.groupby('month')[rh_name].agg(['mean', 'max', 'min'])
    monthly_stats['diff_max'] = monthly_stats['max'] - monthly_stats['mean']
    monthly_stats['diff_min'] = monthly_stats['mean'] - monthly_stats['min']

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharey=False)

    # Daily
    axes[0].plot(daily_stats.index, daily_stats['diff_max'], label='Max - Avg', color='red')
    axes[0].plot(daily_stats.index, daily_stats['diff_min'], label='Avg - Min', color='blue')
    axes[0].set_title('Daily RH Deviation')
    axes[0].legend()
    axes[0].grid(True)
    for period in bad_days:
        axes[0].axvspan(period.to_timestamp(), (period + 1).to_timestamp(), color='gray', alpha=0.15)

    # Weekly
    axes[1].plot(weekly_stats.index, weekly_stats['diff_max'], label='Max - Avg', color='red')
    axes[1].plot(weekly_stats.index, weekly_stats['diff_min'], label='Avg - Min', color='blue')
    axes[1].set_title('Weekly RH Deviation')
    axes[1].legend()
    axes[1].grid(True)
    for period in bad_weeks:
        axes[1].axvspan(period.to_timestamp(), (period + 1).to_timestamp(), color='gray', alpha=0.15)

    # Monthly
    axes[2].plot(monthly_stats.index, monthly_stats['diff_max'], label='Max - Avg', color='red')
    axes[2].plot(monthly_stats.index, monthly_stats['diff_min'], label='Avg - Min', color='blue')
    axes[2].set_title('Monthly RH Deviation')
    axes[2].legend()
    axes[2].grid(True)
    for period in bad_months:
        axes[2].axvspan(period.to_timestamp(), (period + 1).to_timestamp(), color='gray', alpha=0.15)

    for ax in axes:
        ax.set_ylabel('RH Deviation (%)')
        ax.set_xlabel('Date')

    plt.tight_layout()
    plt.show()


def plot_rh_deviation_daily():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['date'] = df.index.date
    df.loc[df.index.to_period('D').isin(bad_days), rh_name] = np.nan

    daily_stats = df.groupby('date')[rh_name].agg(['mean', 'max', 'min'])
    daily_stats['diff_max'] = daily_stats['max'] - daily_stats['mean']
    daily_stats['diff_min'] = daily_stats['mean'] - daily_stats['min']

    plt.figure(figsize=(14, 5))
    plt.plot(daily_stats.index, daily_stats['diff_max'], label='Max - Avg', color='red')
    plt.plot(daily_stats.index, daily_stats['diff_min'], label='Avg - Min', color='blue')
    plt.title('Daily RH Deviation')
    plt.xlabel('Date')
    plt.ylabel('RH Deviation (%)')
    plt.legend()
    plt.grid(True)

    for period in bad_days:
        plt.axvspan(period.to_timestamp(), (period + 1).to_timestamp(), color='gray', alpha=0.15)

    plt.tight_layout()
    plt.show()



def plot_rh_deviation_weekly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['week'] = df.index.to_period('W').start_time
    df.loc[df.index.to_period('W').isin(bad_weeks), rh_name] = np.nan

    weekly_stats = df.groupby('week')[rh_name].agg(['mean', 'max', 'min'])
    weekly_stats['diff_max'] = weekly_stats['max'] - weekly_stats['mean']
    weekly_stats['diff_min'] = weekly_stats['mean'] - weekly_stats['min']

    plt.figure(figsize=(14, 5))
    plt.plot(weekly_stats.index, weekly_stats['diff_max'], label='Max - Avg', color='red')
    plt.plot(weekly_stats.index, weekly_stats['diff_min'], label='Avg - Min', color='blue')
    plt.title('Weekly RH Deviation')
    plt.xlabel('Date')
    plt.ylabel('RH Deviation (%)')
    plt.legend()
    plt.grid(True)

    for period in bad_weeks:
        plt.axvspan(period.to_timestamp(), (period + 1).to_timestamp(), color='gray', alpha=0.15)

    plt.tight_layout()
    plt.show()



def plot_rh_deviation_monthly():
    df = dataframe.copy()
    df = df.loc[start_date_window:end_date_window]
    df['month'] = df.index.to_period('M').start_time
    df.loc[df.index.to_period('M').isin(bad_months), rh_name] = np.nan

    monthly_stats = df.groupby('month')[rh_name].agg(['mean', 'max', 'min'])
    monthly_stats['diff_max'] = monthly_stats['max'] - monthly_stats['mean']
    monthly_stats['diff_min'] = monthly_stats['mean'] - monthly_stats['min']

    plt.figure(figsize=(14, 5))
    plt.plot(monthly_stats.index, monthly_stats['diff_max'], label='Max - Avg', color='red')
    plt.plot(monthly_stats.index, monthly_stats['diff_min'], label='Avg - Min', color='blue')
    plt.title('Monthly RH Deviation')
    plt.xlabel('Date')
    plt.ylabel('RH Deviation (%)')
    plt.legend()
    plt.grid(True)

    for period in bad_months:
        plt.axvspan(period.to_timestamp(), (period + 1).to_timestamp(), color='gray', alpha=0.15)

    plt.tight_layout()
    plt.show()

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










