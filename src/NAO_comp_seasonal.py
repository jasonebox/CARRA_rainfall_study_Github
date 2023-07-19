#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:04:59 2023

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
# from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 
import numpy as np
import datetime


path='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/'
os.chdir(path)

opath='/Users/jason/0_dat/CARRA/output/seasonal/'

ni=1269 ; nj=1069

NAO=pd.read_csv('/Users/jason/Dropbox/NAO/NCDC monthly NAO 1950-2021/output/NAO_seasonal_1950_20222.csv')
NAO=NAO[NAO.year>1990]
NAO=NAO[NAO.year<2022]
NAO.reset_index(drop=True, inplace=True)

print(NAO)

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
mask=np.rot90(mask.T)

fn='/Users/jason/Dropbox/CARRA/CARRA_rain/ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='/Users/jason/Dropbox/CARRA/CARRA_rain/ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)


mask_iceland=1
mask_svalbard=1

if mask_iceland:
    mask[((lon-360>-30)&(lat<66.6))]=0
if mask_svalbard:
    mask[((lon-360>-20)&(lat>70))]=0
# plt.imshow(mask)

varnams=['rf','tp','t2m']
varnams=['tp','t2m']
varnams=['t2m']
# varnams=['tp']

years=np.arange(1991,2022).astype('str')

seasons=['annual','JJAS','JJA','DJFM']

for ss,season in enumerate(seasons):
    totalx=[]
    if ss==1:
        for i,varnam in enumerate(varnams):
            for yy,year in enumerate(years):
                fn=opath+varnams[i]+'_'+year+'_'+seasons[ss]+'_'+str(ni)+'x'+str(nj)+'_float16.npy'
                CARRA=np.fromfile(fn, dtype=np.float16)
                CARRA=CARRA.reshape(ni, nj)
                # CARRA=np.rot90(CARRA.T)
                # plt.imshow(CARRA*mask)
                if varnam=='t2m':
                    mass=np.nanmean(CARRA[mask>0]*mask[mask>0])
                if varnam=='tp':
                    areax=2.5e3**2
                    mass=np.nansum(CARRA[mask>0]*mask[mask>0])/1e9*areax/1000
                print(year,mass)
                totalx.append(mass)

        plt.scatter(NAO[season],totalx)
                