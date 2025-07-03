# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:23:13 2025

@author: timot
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns #install using either: pip install seaborn, or: conda install seaborn
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# create a root window and hide it
root = tk.Tk()
root.withdraw()

# open file dialog, get the selected file path
file_path = filedialog.askopenfilename()

# prints selected file
print("Selected file:", file_path)

#creates dataframe from file selection
dataframe = pd.read_csv(file_path)

# adjust the number of rows to skip, so that you start at 00:00 and with believable values
dataframe = dataframe.iloc[5:].reset_index(drop=True)

dataframe['Time_fixed'] = pd.to_datetime(dataframe['Time'])


def plot_temperature_graph():
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
    

# Calculate daily Temp avg, min,    max and create a table

dataframe = dataframe.sort_values('Time_fixed')
dataframe = dataframe.set_index('Time_fixed', drop=False)

# Resample by day and calculate daily min, max, avg
daily_summary = dataframe['T'].resample('1D').agg(['mean', 'min', 'max']).dropna().reset_index()

daily_summary.columns = ['Date', 'T_avg', 'T_min', 'T_max']
