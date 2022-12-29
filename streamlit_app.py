import streamlit as st
import pandas as pd
import numpy as np
import psutil

import os
import gc
from numpy import repeat as rp
import scipy
from scipy import signal as signal_object
from scipy.stats import zscore
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf



@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
    
    
    
    

DATE_COLUMN = 'created_at'




#######

# Step 3. Function declaration (Definition of the FIR filtering function)
# -----------------------------------------------------------------------
def FIR(data, WinType, winWidth):
    window = np.ones(winWidth)
    if WinType == "hann":
        window = signal_object.windows.hann(winWidth)
    elif WinType == "triangle":
        window = signal_object.windows.triang(winWidth)
    elif WinType == "boxcar":
        window = signal_object.windows.boxcar(winWidth)
    elif WinType == "cosine":
        window = signal_object.windows.cosine(winWidth)
    elif WinType == "barthann":
        window = signal_object.windows.barthann(winWidth)     
    elif WinType == "bartlett":
        window = signal_object.windows.bartlett(winWidth)     
    elif WinType == "blackman":
        window = signal_object.windows.blackman(winWidth)
    elif WinType == "hamming":
        window = signal_object.windows.hamming(winWidth) 
    elif WinType == "nuttall":
        window = signal_object.windows.nuttall(winWidth)
    elif WinType == "parzen":
        window = signal_object.windows.parzen(winWidth) 
    else:
        print ("Error: Not a valid option")
    FIR = np.convolve(data, window, mode='valid')
    return(FIR)
#end

def filter_parameters(Width, date_time):
    lenDate   = len(date_time)
    halfWidth = int(np.floor(Width/2))
    st = halfWidth-1
    if Width % 2: 
        en = lenDate - (halfWidth+1)
    else:
        en = lenDate - (halfWidth)
    return st, en
#end

def PlotFormatter(PLT, AX, title, date_time):
    PLT.legend(loc="lower right", fontsize=16, frameon=False)
    PLT.ylabel('z-score', fontsize=16)
    PLT.xticks(fontsize=15)
    PLT.yticks(fontsize=15)
    PLT.title(title, fontsize=20)
    # add horizontal reference lines and text
    AX.hlines(0,min(date_time),max(date_time), linewidth=1, color='gray')
    AX.hlines(-1.96, min(date_time),max(date_time), linewidth=1, color='gray')
    AX.hlines(1.96,  min(date_time),max(date_time), linewidth=1, color='gray')
    AX.text(date_time[0], 2.05, 'z-score=1.96 (P=0.05)',  style='italic', fontsize=14)
    AX.text(date_time[0], -2.3, 'z-score=-1.96 (P=0.05)', style='italic', fontsize=14)
    # add vertical lines for the election day
    ind_start = date_time =='2020-11-03 00:00:00'
    ind_end   = date_time =='2020-11-04 00:00:00'
    AX.vlines(date_time[ind_start], -5, 5, linewidth=1, color='red')
    AX.vlines(date_time[ind_end],   -5, 5, linewidth=1, color='red')
    AX.text(date_time[ind_start], 2.5, 'Election Day', style='italic', 
            bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 0.3}, fontsize=12.5)
    PLT.ylim(-3.5, 3.5)
    return PLT
#end

def plot_filtered_data(plt, date_time, data, Widths, WinType, title):
    fig, ax = plt.subplots(figsize = (20,5) )
    plt.plot(date_time,zscore(data),linewidth = 0.5, label='Non-filtered data', alpha=0.7)
    for i in range(len(Widths)):
        Width   = Widths[i]
        st, en  = filter_parameters(Width, date_time)
        label   = 'Filtered data ' + '(width=' + str(Width) +  ')'
        plt.plot(date_time[st:en,],zscore(FIR(data, WinType, Width)),linewidth = 1, label=label)
    plt     = PlotFormatter(plt, ax, title, date_time)
    return plt, ax



#########
DATA_URL = ('https://osf.io/download/v5ad4/')
         
st.title("Time series visualization: positive and negative emotions during the 2020 Presidential election week")
st.text(psutil.virtual_memory())
data_load_state = st.text('Loading data...')

DatasetA = load_data(317861)
st.text(psutil.virtual_memory())
data_load_state.text('Loading data...done!')

VARS = ['posemo',	'negemo',	'anx',	'anger',	'sad']
Centered = DatasetA.groupby('user_id')[VARS].apply(lambda x: x - x.mean())
DatasetA_c = pd.concat([DatasetA.iloc[:,[0,1,2,9,10,12]], Centered], axis=1)
DatasetA_c['created_at'] = pd.to_datetime(DatasetA_c['created_at'])
del([[DatasetA, Centered]])
gc.collect()
# Aggregating
LIWC = DatasetA_c.drop(columns = ['user_id','status_id']).groupby(['day', 'hour', 'quarter']).agg('mean').reset_index()
#creating date_time object
date_time = list(map(lambda day, hour, quarter: '2020-11-0' + str(day) + " " + str(hour) + ":" + str((quarter-1)*15) + ":00", LIWC['day'],LIWC['hour'],LIWC['quarter']))
date_time = pd.to_datetime(date_time)

plt2 = plt
plt3 = plt
####
#data1    = LIWC['posemo']
Widths  = [12, 24]
WinType = "triangle"
title   = "A) Positive emotion: Filtered data using " +  WinType + " windows"
plt1, ax1 = plot_filtered_data(plt, date_time, LIWC['posemo'], Widths, WinType, title)

st.pyplot(plt1)
st.text(psutil.virtual_memory())

#
#data2    = LIWC['negemo']
Widths2  = [12, 24]
WinType2 = "triangle"
title   = "B) Negative emotion: Filtered data using " +  WinType + " windows"
plt2, ax2 = plot_filtered_data(plt2, date_time, LIWC['negemo'], Widths2, WinType2, title)

st.pyplot(plt2)

st.text(psutil.virtual_memory())

#data3    = LIWC['negemo']
Widths3  = [24]
WinType3 = "triangle"
title   = "C) Changepoint detection using the filtered Negative emotion"
plt3, ax3 = plot_filtered_data(plt3, date_time, LIWC['negemo'], Widths3, WinType3, title)

st.pyplot(plt3)

st.text(psutil.virtual_memory())
