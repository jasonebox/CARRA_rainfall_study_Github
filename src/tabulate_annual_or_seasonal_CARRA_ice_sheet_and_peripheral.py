#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outputs .csv of ice sheet and peripheral glacier rf, tp and t2m

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
        inpath='./output_annual/'
        if ss:inpath='./output_seasonal/'
    
        tp_stats=np.zeros(n_years)
        rf_stats=np.zeros(n_years)
        fr_stats=np.zeros(n_years)
        
        tp_stats_PG=np.zeros(n_years)
        rf_stats_PG=np.zeros(n_years)
        fr_stats_PG=np.zeros(n_years)
        
        t2m_stats=np.zeros(n_years)
        t2m_stats_PG=np.zeros(n_years)
        
        for yy,year in enumerate(years):
        # for year in years[1:2]:
            if yy>=0:
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
        
                fr=np.zeros((ni,nj))
                v=np.where(tp>0)
                fr[v]=rf[v]/tp[v]
        
                t2m_stats[yy]=np.mean(t2m[mask>0])
                t2m_stats_PG[yy]=np.mean(t2m[mask_fuzzy_PG>0])
    
                fr_stats[yy]=np.mean(fr[mask>0])
                fr_stats_PG[yy]=np.mean(fr[mask_fuzzy_PG>0])
        
                # x=tp
                # tp[mask<=0]=np.nan
                # plt.imshow(tp)
                # assaas
                areax=2.5e3**2
                not_fuz=np.sum(tp[mask>0])/1e9*areax/1000
                tp_stats[yy]=np.sum(tp[mask>0]*mask[mask>0])/1e9*areax/1000
                rf_stats[yy]=np.sum(rf[mask>0]*mask[mask>0])/1e9*areax/1000
        
                tp_stats_PG[yy]=np.sum(tp[mask_fuzzy_PG>0]*mask_fuzzy_PG[mask_fuzzy_PG>0])/1e9*areax/1000
                rf_stats_PG[yy]=np.sum(rf[mask_fuzzy_PG>0]*mask_fuzzy_PG[mask_fuzzy_PG>0])/1e9*areax/1000
        
                print(year,tp_stats[yy],not_fuz)#,t2m_stats[yy])
                # print(year,rf_stats[yy],tp_stats[yy],fr_stats[yy])#,t2m_stats[yy])
            
        # data=[[years],[means],[stds]]
        df = pd.DataFrame(columns = ['year', 'rf','tp','fr','rf_PG','tp_PG','fr_PG','t2m','t2m_PG']) 
        df.index.name = 'index'
        df["year"]=pd.Series(years)
        df["rf"]=pd.Series(rf_stats)
        df["tp"]=pd.Series(tp_stats)
        df["fr"]=pd.Series(fr_stats)
        
        df["rf_PG"]=pd.Series(rf_stats_PG)
        df["tp_PG"]=pd.Series(tp_stats_PG)
        df["fr_PG"]=pd.Series(fr_stats_PG)
        
        df["t2m"]=pd.Series(t2m_stats)
        df["t2m_PG"]=pd.Series(t2m_stats_PG)
        
        opath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/stats/'
        ofile=opath+'tabulate_'+season+'_CARRA.csv'
        df.to_csv(ofile)
