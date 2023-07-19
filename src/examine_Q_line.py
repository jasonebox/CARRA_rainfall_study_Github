# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os
import xarray as xr
import glob
# import geopandas as gpd
from scipy.spatial import distance
# from shapely.geometry import Point
from datetime import datetime, timedelta

varname='t2m' ; app='PROMICE'
varname='tp' ; app='snow_pits'
varname='t2m' ; app='Q_line'

AW = 0

base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'

if AW:
    base_path = 'C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
    + 'CARRA_rain/'

os.chdir(base_path)
#--------------------------

fs=22 # font size
th=1 # line thick
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "grey"
plt.rcParams["font.size"] = fs

#--------------------------

meta = pd.read_csv('./ancil/Q_line.csv')
meta.station_name=meta.station_name.astype(str)

fn='/Users/jason/Dropbox/CARRA/CARRA_at_points_with_monthly_timeseries/output_csv/t2m_at_Q_line.csv'
t2m=pd.read_csv(fn)

fn='/Users/jason/Dropbox/CARRA/CARRA_at_points_with_monthly_timeseries/output_csv/rf_at_Q_line.csv'
rf=pd.read_csv(fn)

fn='/Users/jason/Dropbox/CARRA/CARRA_at_points_with_monthly_timeseries/output_csv/tp_at_Q_line.csv'
tp=pd.read_csv(fn)

# print(rf.columns)

v=np.where(rf.time=='2017-09-14')
# v=np.where(rf.time=='2017-09-15')
print(v[0])

fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111)
cc=0

# ax.set_title('CARRA S Greenland Q Lobe 2017-09-14')
rfx=rf.iloc[v[0][0],2:]
tpx=tp.iloc[v[0][0],2:]
t2mx=t2m.iloc[v[0][0],2:]
v2=np.where(rfx==np.max(rfx))
print(meta.elev[v2[0][0]],t2mx[v2[0][0]])

lo_elev=10
v_elev=np.where(meta.elev>lo_elev)

# ax.axvline(x=meta.elev[v2[0][0]],c='lightgrey')
ax.plot(meta.elev[v_elev[0]],rfx[v_elev[0]],'o-',color='c',label='rainfall',zorder=10)
ax.plot(meta.elev[v_elev[0]],tpx[v_elev[0]]-rfx[v_elev[0]],'s-',c='b',label='snowfall')
ax.set_ylabel('precipitation, mm per day', color='k')
ax.set_ylim(-3,290)
plt.legend(loc=3)
ax3 = ax.twinx()
ax.set_xlabel('surface elevation, m')

ax3.plot(meta.elev[v_elev[0]],t2mx[v_elev[0]],'o-',c='r',label='2 m air temperature')
ax3.set_ylabel('deg. C', color='r')
ax3.set_ylim(0,10.5)
ax3.tick_params(axis='y', colors='r')
# ax.tick_params(axis='y', colors='red')
plt.legend(loc=1)

max_rf=np.nanmax(rfx)
print(max_rf)
v=np.where(rfx==max_rf)
v=v[0][0]
print(meta.elev[v])
print(rfx[:-10])

#%%
for i in range(len(meta)):
    if meta.elev[i]>lo_elev: print(meta.elev[i],rfx[i])