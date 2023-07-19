#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:46:51 2020

code to map SICE rasters on a single Arctic map

@author: jeb
"""

# import geopandas
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

from pathlib import Path


# fn='/Users/jason/Dropbox/Greenland map/coastline/Greenland_coast/Greenland_coast.shp' 
# coastline = geopandas.read_file(fn)

# --------------------------------- guess this not needed
# reproj_file='/tmp/x.tif' 
# dst_crs = 'EPSG:4326' #WGS84

# with rasterio.open(fn,mode='r') as src:
#     transform, width, height = calculate_default_transform(src.crs, dst_crs,src.width,src.height,*src.bounds)
#     kwargs = src.meta.copy()  #create features for dst
#     kwargs.update({'crs': dst_crs,'transform': transform, 'width': width,'height': height}) #update dst features
    
#     #write new file: new extension, projection, compression
#     with rasterio.open(reproj_file, 'w', **kwargs,compress='deflate') as dst:
#             reproject(source=rasterio.band(src, 1),destination=rasterio.band(dst, 1),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.nearest)
# raster_file=reproj_file



fn='/Users/jason/Dropbox/500m_grid/BedMachineGreenland-2017-09-20_surface_500m.tif'
fn='/Users/jason/Dropbox/1km_grid2/elev_1km_1487x2687.tif'
fn='/Users/jason/Dropbox/1km_grid2/elev_1km_land_only_1487x2687.tif'
r400x = rasterio.open(fn)
profile_S3=r400x.profile
elev=r400x.read(1)
elev=np.array(elev).astype(float)
elev[elev<0]=np.nan

fn='/Users/jason/Dropbox/1km_grid2/lon_1km_1487x2687.tif'
r400x = rasterio.open(fn)
profile_S3=r400x.profile
lon=r400x.read(1)
lon=np.array(lon).astype(float)
lon[lon<-180]=np.nan

fn='/Users/jason/Dropbox/1km_grid2/lat_1km_1487x2687.tif'
r400x = rasterio.open(fn)
profile_S3=r400x.profile
lat=r400x.read(1)
lat=np.array(lat).astype(float)
lat[lat<60]=np.nan

np.shape(elev)
#%%
# 500 m data
y0=5099
x0=890 ; x1=1200
thickness=35

# 1000 m data
y0=2550
x0=445 ; x1=630
thickness=15

temp=elev.copy()
fig, ax = plt.subplots(figsize=(10, 10))


temp[y0-thickness:y0+thickness,x0:x1]=3000
plt.imshow(temp)
# rasterio.plot.show(NDSI, ax=ax)
# coastline.plot(ax=ax, facecolor='none', edgecolor='red')
plt.axis('off')
plt.colorbar()
# plt.title(datex+' '+band)

ly='x'
if ly == 'x':plt.show()
 
if ly == 'p':            
    figname='a.png'
    plt.savefig(figname, bbox_inches='tight', dpi=150)
#%%    
    
sample=np.nanmean(elev[y0-thickness:y0+thickness,x0:x1],axis=0)
maxx=np.nanmax(elev[y0-thickness:y0+thickness,x0:x1],axis=0)
stds=np.nanstd(elev[y0-thickness:y0+thickness,x0:x1],axis=0)*1.96

sample_lon=np.nanmean(lon[y0-thickness:y0+thickness,x0:x1],axis=0)
sample_lat=np.nanmean(lat[y0-thickness:y0+thickness,x0:x1],axis=0)
# sample_lon=lon[y0,x0:x1]
# sample_lon=sample_lon[0,:]
np.shape(sample_lon)
np.shape(maxx)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(sample,label='mean')
ax.plot(maxx,label='max')
# ax.fill_between(np.arange(0,np.shape(sample)[0]),sample+stds, sample-stds, color='k',alpha=0.2,interpolate=True,
#                               label='stdev. smoothed')
ax.set_xlim(0,np.shape(sample)[0])
# ax.set_xlim(np.nanmin(sample_lon),np.nanmax(sample_lon))
ax.set_ylim(np.nanmin(sample),np.nanmax(maxx))

xx0=47 ; xx1=119 # gap
# xx0=0 ; xx1=46 # west of gap
# xx0=120 ; xx1=166 # west of gap
ax.vlines(xx0,0,np.nanmax(maxx))
ax.vlines(xx1,0,np.nanmax(maxx))
print('mean',np.mean(sample[xx0:xx1]))
print('max lat',np.nanmax(sample_lat[xx0:xx1]))
print('min lat',np.nanmin(sample_lat[xx0:xx1]))
print('max',np.mean(maxx[xx0:xx1]))
print(xx1-xx0)
plt.show()

