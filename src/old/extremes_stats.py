#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:53:12 2022

@author: jason
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
from glob import glob
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 
import numpy as np

# CARRA grid info
# Lambert_Conformal()
#     grid_mapping_name: lambert_conformal_conic
#     standard_parallel: 72.0
#     longitude_of_central_meridian: -36.0
#     latitude_of_projection_origin: 72.0
#     earth_radius: 6367470.0
#     false_easting: 1334211.3405653758
#     false_northing: 1584010.8994621644
#     longitudeOfFirstGridPointInDegrees: 302.903
#     latitudeOfFirstGridPointInDegrees: 55.81

AW=0
path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
if AW:path='/Users/jason/Dropbox/CARRA/prog/map_CARRA_west/'
os.chdir(path)

df=pd.read_csv('/Users/jason/Dropbox/CARRA/CARRA_rain/stats/rf_extremes.csv')
df['date2']=pd.to_datetime(df['date'])
df['year']=pd.DatetimeIndex(df['date2']).year
print(df.columns)

# s=reversed(sorted(df.maxlocalrate))

#%%
for year in np.arange(1991,2022):
    print(year,sum(df.year==year))
#%%

var=df.maxlocalrate.values
# var=df.Gt_overall.values

s=reversed(np.argsort(var))
for sort_index,i in enumerate(s):
    if sort_index<15:
        print(sort_index,df['date2'][i].strftime('%Y-%m-%d'),df.Gt_overall[i],df.maxlocalrate[i],df.lat[i],df.lon[i],df.elev[i])