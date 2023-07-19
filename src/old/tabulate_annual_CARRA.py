#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:04:02 2021

@author: jeb and AW

preceded by ./CARRA_rain/map_annual_CARRA.py

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
       
fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

# plt.imshow(lat)
# plt.colorbar()
# #%%
fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)

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
years=np.arange(1991,2021).astype('str')

n_years=len(years)

wo=1


tp_stats=np.zeros(n_years)
rf_stats=np.zeros(n_years)
fr_stats=np.zeros(n_years)

tp_stats_PG=np.zeros(n_years)
rf_stats_PG=np.zeros(n_years)
fr_stats_PG=np.zeros(n_years)

# t2m_stats=np.zeros(n_years)

for yy,year in enumerate(years):
# for year in years[1:2]:
    if yy>=0:
    # if year=='2017':
    # if yy==2:
    # if ((yy>0=)&(yy<=3)):
        fn='./output_annual/tp_'+year+'_'+str(ni)+'x'+str(nj)+'_float32.npy'
        tp=np.fromfile(fn, dtype=np.float32)
        tp=tp.reshape(ni, nj)

        fn='./output_annual/rf_'+year+'_'+str(ni)+'x'+str(nj)+'_float32.npy'
        rf=np.fromfile(fn, dtype=np.float32)
        rf=rf.reshape(ni, nj)

        # fn='./output_annual/t2m_'+year+'_'+str(ni)+'x'+str(nj)+'_float32.npy'
        # t2m=np.fromfile(fn, dtype=np.float32)
        # t2m=t2m.reshape(ni, nj)

        fr=np.zeros((ni,nj))
        v=np.where(tp>0)
        fr[v]=rf[v]/tp[v]

        
        # t2m_stats[yy]=np.mean(t2m[mask>0])
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
df = pd.DataFrame(columns = ['year', 'rf','tp','fr','rf_PG','tp_PG','fr_PG']) 
df.index.name = 'index'
df["year"]=pd.Series(years)
df["rf"]=pd.Series(rf_stats)
df["tp"]=pd.Series(tp_stats)
df["fr"]=pd.Series(fr_stats)

df["rf_PG"]=pd.Series(rf_stats_PG)
df["tp_PG"]=pd.Series(tp_stats_PG)
df["fr_PG"]=pd.Series(fr_stats_PG)

# df["t2m"]=pd.Series(t2m_stats)

# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
ofile='./output_annual/tabulate_annual_CARRA.csv'
df.to_csv(ofile)
