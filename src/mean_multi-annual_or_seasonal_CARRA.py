#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

still fine for this study but further developed in /Users/jason/Dropbox/CARRA/prog/mean_multi-annual_or_seasonal_or_monthly_CARRA.py

preceded by ./CARRA_rain/map_annual_CARRA.py

updated Nov 2022
@author: Jason Box, GEUS, jeb@geus.dk
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob
# from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 
import xarray as xr

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
os.chdir(path)

# # global plot settings
# th=1
# font_size=18
# plt.rcParams['axes.facecolor'] = 'k'
# plt.rcParams['axes.edgecolor'] = 'k'
# plt.rcParams["font.size"] = font_size

ni=1269 ; nj=1069
fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = xr.open_dataset(fn)

mask = np.array(nc2.z)
# mask=np.rot90(mask.T)
# plt.imshow(mask)
# plt.colorbar()


# mask = np.rot90(mask.T)
# plt.imshow(mask)
# fn='./ancil/CARRA_W_domain_ice_mask.nc'
# ds=xr.open_dataset(fn)
# print(ds)
# mask = np.rot90((ds.z).T)
mask_iceland=1
mask_svalbard=1
mask_jan_mayen=1
       

if mask_jan_mayen:
    mask[((lon-360>-15)&(lat>66.6)&(lat<75))]=0
if mask_iceland:
    mask[((lon-360>-30)&(lat<66.6))]=0
if mask_svalbard:
    mask[0:300,800:]=0

# fn='/Users/jason/Dropbox/CARRA/CARRA_rain/ancil/mask_peri_glaciers_ice_caps_features_1269x1069.npy'
fn='/Users/jason/Dropbox/CARRA/CARRA_rain/ancil/mask_peri_glaciers_ice_caps_1269x1069.npy'
msk_pg=np.load(fn)
# plt.imshow(msk_pg)
msk_pg=np.rot90(msk_pg.T)

mask_fuzzy_PG=np.array(mask)
mask_fuzzy_PG[msk_pg==0]=0

# plt.imshow(mask)
# plt.colorbar()
# #%%
# x=mask
# x=lat
years=np.arange(1991,2022).astype('str')

n_years=len(years)

wo=1

seasons=['annual','JJAS']
season_name=['_','_JJAS_']

for ss,season in enumerate(seasons):
    if ss==0:
        inpath='/Users/jason/Dropbox/CARRA/CARRA_rain/output_annual/'
        if ss:inpath='/Users/jason/0_dat/CARRA/output_seasonal/'
    
        tp_stats=np.zeros((ni,nj))
        rf_stats=np.zeros((ni,nj))
        t2m_stats=np.zeros((ni,nj))
        
        cc=0
        
        for yy,year in enumerate(years):
        # for year in years[1:2]:
            if yy>=0:
                print(year)
            # if year=='2017':
            # if yy==2:
            # if ((yy>0=)&(yy<=3)):
                fn=inpath+'tp_'+year+season_name[ss]+str(ni)+'x'+str(nj)+'_float16.npy'
                os.system('ls -lF '+fn)
                tp=np.fromfile(fn, dtype=np.float16)
                tp=tp.reshape(ni, nj)
        
                fn=inpath+'rf_'+year+season_name[ss]+str(ni)+'x'+str(nj)+'_float16.npy'
                os.system('ls -lF '+fn)
                rf=np.fromfile(fn, dtype=np.float16)
                rf=rf.reshape(ni, nj)
                
        
                fn=inpath+'t2m_'+year+season_name[ss]+str(ni)+'x'+str(nj)+'_float16.npy'
                os.system('ls -lF '+fn)
                t2m=np.fromfile(fn, dtype=np.float16)
                t2m=t2m.reshape(ni, nj)
        
                t2m_stats+=t2m
                rf_stats+=rf
                tp_stats+=tp
                cc+=1

        ofile='./output_annual/rf_1991-2021_1269x1069_float16.npy'
        temp=rf_stats/cc
        temp=np.rot90(temp.T)
        temp.astype('float16').tofile(ofile)     

        do_plot=1
        
        if do_plot:
            ly='x'
            plt.close()
            plt.imshow(temp)
            plt.axis(False)
            plt.colorbar()
            plt.title(year)
            if ly == 'x':
                plt.show()

            DPIs=[150]
            # DPIs=[150]

            if ly =='p':
                for DPI in DPIs:
                    figpath='/Users/jason/0_dat/CARRA_temp/extremes/'
                    # os.system('mkdir -p '+figpath)
                    plt.savefig(figpath+date.strftime('%Y-%m-%d')+'.png', bbox_inches='tight', dpi=DPI)

        ofile='./output_annual/tp_1991-2021_1269x1069_float16.npy'
        temp=tp_stats/cc
        temp=np.rot90(temp.T)
        temp.astype('float16').tofile(ofile)    
        
        ofile='./output_annual/t2m_1991-2021_1269x1069_float16.npy'
        temp=t2m_stats/cc
        temp=np.rot90(temp.T)
        temp.astype('float16').tofile(ofile)
        
        # np.nanmax(temp)
        # plt.imshow(temp)
        # plt.colorbar()