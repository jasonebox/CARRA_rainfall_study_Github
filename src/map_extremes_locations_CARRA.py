#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:04:02 2021

@author: jeb and AW
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

ly='p'

res='l'
if ly=='p':res='h'

# global plot settings
th=1
font_size=18
# plt.rcParams['axes.facecolor'] = 'k'
# plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
# mask = np.rot90(mask.T)
# plt.imshow(mask)

ni=1269 ; nj=1069

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)

# latx=np.rot90(lat.T)
# lonx=np.rot90(lon.T)
offset=0
lon=lon[offset:ni-offset,offset:nj-offset]
lat=lat[offset:ni-offset,offset:nj-offset]
ni-=offset*2
nj-=offset*2
# print(ni,nj)
LLlat=lat[0,0]
LLlon=lon[0,0]-360
# print("LL",LLlat,LLlon)
# print("UL",lat[ni-1,0],lon[ni-1,0]-360)
lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
lat0=lat[int(round(ni/2)),int(round(nj/2))]
# print("mid lat lon",lat0,lon0)

URlat=lat[ni-1,nj-1]
URlon=lon[ni-1,nj-1]
# print("LR",lat[0,nj-1],lon[0,nj-1]-360)
# print("UR",URlat,URlon)

# m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=72, lon_0=-36, resolution='l', projection='lcc')
m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=lat0, lon_0=lon0, resolution=res, projection='lcc')
x, y = m(lat, lon)

lon-=360
lons, lats = m(lon, lat)

#%%

plt.close()
fig, ax = plt.subplots(figsize=(12, 12))

ly='x'

# plt.imshow(lon)
ax = plt.subplot(111)
tit='CARRA extremesover Greenland ice'
# ax.set_title(tit)


m.drawcoastlines(color='k',linewidth=0.5)
# m.drawparallels([66.6],color='gray')
# m.drawparallels([60,70,80],dashes=[2,4],color='k')
# m.drawmeridians(np.arange(0.,420.,10.))
# m.drawmapboundary(fill_color='aqua')
ax = plt.gca()     
# plt.title("Lambert Conformal Projection")
# plt.show()


fn='/Users/jason/Dropbox/CARRA/CARRA_rain/stats/rf_extremes.csv'
df=pd.read_csv(fn)

print(df.columns)

minval=300 ;scaling_factor=3000

v=np.where(df.maxlocalrate>minval)

v=np.where(df.Gt_overall>2.2)
# v=v[0]
print(v[0])

# print()
minval=np.min(df.maxlocalrate[v[0]]) ; scaling_factor=1500
for i in v[0]:
    # co_index=(df.maxlocalrate[i]-200)/(np.max(df.maxlocalrate)-200)*255
    print(i,df.date[i],df.lon[i],df.lat[i],df.Gt_overall[i],df.maxlocalrate[i])
    xx,yy=m(df.lon[i],df.lat[i])
    m.scatter(xx,yy,s=(df.maxlocalrate[i]-minval)/465*scaling_factor,facecolors='none', edgecolors='m',linewidths=th*2)#cmap='jet')
    # plt.text(xx,yy,str(df.maxlocalrate[i]))#cmap='jet')

print('n cases',len(v[0]))

# m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=380, facecolors='none', edgecolors='m',linewidths=th*1)
# m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='m')

# if i<2: # rf or tp
#     cbar_min=min_value[i]
#     cbar_max=max_value[i]
#     cbar_step=max_value[i]/24
#     cbar_num_format = "%d"

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
    
#     # plt.colorbar(im)            
#     cbar = plt.colorbar(pp,
#                         orientation='vertical',
#                         ticks=np.arange(cbar_min,
#                         cbar_max+cbar_step, cbar_step),format=cbar_num_format, cax=cax)
#     cbar.ax.set_ylabel(units[i], fontsize = font_size)
#     tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step).astype(int)
#     # print(tickranges)
#     cbar.ax.set_yticklabels(tickranges, fontsize=font_size)

# if i==2: # t2m
#     cbar_min=min_value[i]
#     cbar_max=max_value[i]
#     cbar_step=max_value[i]/24
#     cbar_num_format = "%d"

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
    
#     # plt.colorbar(im)            
#     cbar = plt.colorbar(pp,orientation='vertical',format=cbar_num_format, cax=cax)
#     cbar.ax.set_ylabel(units[i], fontsize = font_size)
#     # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
#     # # print(tickranges)
#     # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)

# cc=0
# xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
# mult=0.7
# color_code='grey'
# plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
#   transform=ax.transAxes,color=color_code) ; cc+=1. 


# if i<2:
#     cc=0
#     xx0=0.44 ; yy0=0.17 ; dy2=-0.028
#     mult=0.8
#     color_code='k'
#     print()
#     print(year)
#     plt.text(xx0, yy0+cc*dy2,year, fontsize=font_size*mult,
#       transform=ax.transAxes,color=color_code) ; cc+=1.

#     msg="{:.0f}".format(mass)+" Gt / y "+varnam2[i]+" total mass flux"
#     print(msg)
#     plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
#       transform=ax.transAxes,color=color_code) ; cc+=1. 

#     msg="{:.0f}".format(np.max(plotvar))+" mm / y "+"max "+varnam2[i]+""
#     print(msg)
#     plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
#       transform=ax.transAxes,color=color_code) ; cc+=1. 

#     msg="{:.4f}".format(alllat)+" N, "+"{:.4f}".format(abs(alllon))+" W"
#     print(msg)
#     plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
#       transform=ax.transAxes,color=color_code) ; cc+=1. 
    
#     # msg="{:.0f}".format(np.max(plotvar[v]))+" mm / y "+"max "+varnam2[i]+" SSW"
#     # print(msg)
#     # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

    # msg="{:.4f}".format(SSWlat)+" N, "+"{:.4f}".format(abs(SSWlon))+" W"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

ly='x'
if ly == 'x':
    plt.show()


if ly =='p':
    figpath='/Users/jason/Dropbox/CARRA/CARRA rainfall study/Figs/'
    # os.system('mkdir -p '+figpath)
    figname='map_extremes_locations_CARRA_rf_extremes'
    plt.savefig(figpath+figname+'.pdf', bbox_inches='tight')
            #%%
make_gif=1

if make_gif:
    animpath='/Users/jason/Dropbox/CARRA/anim/'
    # inpath='/Users/jason/Dropbox/CARRA/Figs/'
    inpath=figpath
    msg='convert  -delay 70  -loop 0   '+inpath+'*'+varnams[i]+'*'+'.png  '+animpath+varnams[i]+'_1991-2021.gif'
    os.system(msg)