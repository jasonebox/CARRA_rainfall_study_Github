#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
for ERA5 upstream is /Users/jason/Dropbox/CARRA/CARRA_ERA5_events/src/ERA5/Resampling_ERA5_to_CARRA_code.py

carried forward in /Users/jason/Dropbox/CARRA/CARRA_ERA5_events/

updated Nov 2022
@author: Jason Box, GEUS, jeb@geus.dk
"""

import os
from glob import glob
from netCDF4 import Dataset
import pandas as pd
from datetime import datetime 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Polygon 

# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams['axes.facecolor']='w'
plt.rcParams['savefig.facecolor']='w'

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

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
os.chdir(path)

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
# mask = np.rot90(mask.T)
# plt.imshow(mask)

# rf is raingall, tp is total precipitation
varnams=['rf','tp','t2m']

wo=1

ni=1269 ; nj=1069

rf_cum=np.zeros((ni,nj))

casex='2021-08-14' ; casex2='14 August, 2021'
# casex='2017-09-14'; casex2='14 September, 2017'

do_ERA5=0
CARRA_or_ERA5='CARRA'
if do_ERA5:
    CARRA_or_ERA5='ERA5'
    
for hh in range(8):            
    print(hh)
    fn='/Users/jason/0_dat/CARRA/output/event/rf/'+casex+'-'+str((hh+1)*3)+'_1269x1069.npy'
    if do_ERA5:
        fn='/Users/jason/0_dat/ERA5/events/resampled/rf/'+casex+'-'+str((hh)*3).zfill(2)+'_1269x1069.npy'
        
    x=np.fromfile(fn, dtype=np.float16)
    x=x.reshape(ni, nj)
    rf_cum+=x
    plt.close()
    # plt.imshow(rf_cum)
    plt.imshow(x)
    plt.title(str((hh+1)*3))
    plt.show()

rf_cum=np.rot90(rf_cum.T)
outpath='/Users/jason/Dropbox/CARRA/CARRA_ERA5_events/data_raw/'+CARRA_or_ERA5+'/event/'
rf_cum.astype('float16').tofile(outpath+'rf_'+casex+'_1269x1069.npy')
    #%%

ly='x'

res='l'
if ly=='p':res='h'


# ly='x'
map_version=1
if map_version:
    fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
    lat=np.fromfile(fn, dtype=np.float32)
    lat=lat.reshape(ni, nj)

    fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
    lon=np.fromfile(fn, dtype=np.float32)
    lon=lon.reshape(ni, nj)

    fn='./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
    elev=np.fromfile(fn, dtype=np.float32)
    elev=elev.reshape(ni, nj)
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
#%%

# def loadct(i):
#     loval=188
#     if i==0:loval=255
#     print("loval",loval)
#     r=[188,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156, 255, 255]
#     g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28, 255, 255 ]
#     b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196, 0, 255 ]
#     r=[loval,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156]
#     g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28 ]
#     b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196]
#     colors = np.array([r, g, b]).T / 255
#     n_bin = 24
#     cmap_name = 'my_list'
#     # Create the colormap
#     cm = LinearSegmentedColormap.from_list(
#         cmap_name, colors, N=n_bin)
#     cm.set_under('w') #  brown, land
#     # my_cmap.set_over('#E4EEF8')
#     return cm



#%%
plotvar=rf_cum
plotvar_non_fuzzy=rf_cum

fn='./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
elevx=np.fromfile(fn, dtype=np.float32)
elevx=elevx.reshape(ni, nj)
elevx=np.rot90(elevx.T)

varnam2=['rainfall','precip.',r'$(t2m_{max} + t2m_{min})/2$']
min_value=[0,0,-30]
max_value=[72,6000,4]
units=['mm','mm','deg C']
 


# plotvar*=mask
plotvar[mask<=0]=np.nan
plotvar_non_fuzzy[mask<0]=0

mask_svalbard=1;mask_iceland=1;mask_jan_mayen=1

i=0

if i<3:
    if mask_iceland:
        mask[((lon-360>-30)&(lat<66.6))]=0
    if mask_svalbard:
        mask[((lon-360>-20)&(lat>70))]=0
					# if mask_jan_mayen:
					#     mask[((lon-360>-15)&(lat>66.6)&(lat<75))]=0
					# if mask_iceland:
					#     mask[((lon-360>-30)&(lat<66.6))]=0
					# if mask_svalbard:
					#     mask[0:300,800:]=0
					#                     plotvar[mask==0]=0


plotvar_masked=plotvar.copy()
# plotvar_masked[elevx<800]=np.nan
plotvar_masked[mask<1]=np.nan

if i==2:
    plotvar_non_fuzzy[mask==0]=np.nan
    plotvar[mask==0]=np.nan

areax=2.5e3**2
mass=np.sum(plotvar[mask>0]*mask[mask>0])/1e9*areax/1000

if i<2:plotvar_non_fuzzy[mask==0]=-1


def loadct(i):
    col_bins=4
    bin_bins=0
    off=0
    # colors0 = plt.cm.binary(np.linspace(0.9, 0, bin_bins)) #binary
    colors7 = plt.cm.Blues(np.linspace(0.1, 0.9, col_bins))
    colors6 = plt.cm.Greens(np.linspace(0.1, 0.9, col_bins))
    colors5 = plt.cm.BrBG(np.linspace(0.4, 0.1, col_bins)) #browns
    colors4= plt.cm.Reds(np.linspace(0.1, 0.9, col_bins))
    colors3 = plt.cm.Purples(np.linspace(0.1, 0.9, col_bins)) 
    colors2 = plt.cm.RdPu(np.linspace(0.7, 0.8, 1)) #magenta
    colors1 = plt.cm.autumn(np.linspace(0.9, 1, 1)) #yellow
    colors = np.vstack((colors7, colors6, colors5, colors4, colors3, colors2, colors1))
    colors=colors[0:int(len(colors))-off+2]
    n_bin = bin_bins + col_bins*5 +2 - off
    colors=colors[0:int(len(colors))-bin_bins] 
    n_bin=n_bin-bin_bins
    
    bounds = [0,1,2,3,4,6,8,10,14,24,30,40,55,70,90,120,160,200,265,350]
    
    max_value=np.max(bounds)
    min_value=0

    #create colormap
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    cm.set_bad(color='white') #set color of zeros white
    norm = BoundaryNorm(bounds, cm.N)
    cbar_num_format = "%d"
    return cm,bounds

plt.close()
fig, ax = plt.subplots(figsize=(12, 12))
# plt.imshow(lon)
ax = plt.subplot(111)
# tit=year+' CARRA '+varnam2[i]+' over Greenland ice'
# ax.set_title(tit)

if i<2:cm,bounds=loadct(i)

if map_version==0:
    pp=plt.imshow(plotvar_non_fuzzy, interpolation='nearest', origin='lower', cmap=cm,vmin=min_value[i],vmax=max_value[i]) ; plt.axis('off') 

lon-=360
if i<2:
    v=np.where((lon<-45)&(lat<62))
    maxval=np.max(plotvar_non_fuzzy)
    minval=np.min(plotvar_non_fuzzy[plotvar_non_fuzzy>1])
    maxval2=np.max(plotvar_non_fuzzy[v])
    # print("position of extremes")
    # print(np.max(plotvar_non_fuzzy))
    # print(maxval)
    # print(lat[plotvar==maxval2])
    # print(lon[plotvar==maxval2])
    SSWlat=lat[plotvar_non_fuzzy==maxval2][0]
    SSWlon=lon[plotvar_non_fuzzy==maxval2][0]
    elev_atSSW=elev[plotvar_non_fuzzy==maxval2][0]
    # print(lat[plotvar==maxval2])
    # print(lon[plotvar==maxval2])
    alllat=lat[plotvar_non_fuzzy==maxval][0]
    alllon=lon[plotvar_non_fuzzy==maxval][0]
    minlat=lat[plotvar_non_fuzzy==minval][0]
    minlon=lon[plotvar_non_fuzzy==minval][0]

if map_version:
    pp=m.imshow(plotvar, cmap = cm,vmin=min_value[i],vmax=max_value[i]) 
    # m.axis('off')
    m.drawcoastlines(color='k',linewidth=0.5)
    # m.drawparallels([66.6],color='gray')
    # m.drawparallels([60,70,80],dashes=[2,4],color='k')
    # m.drawmeridians(np.arange(0.,420.,10.))
    # m.drawmapboundary(fill_color='aqua')
    ax = plt.gca()     
    # plt.title("Lambert Conformal Projection")
    # plt.show()
    lons, lats = m(lon, lat)
    # m.scatter(lons[plotvar_non_fuzzy==maxval2],lats[plotvar_non_fuzzy==maxval2], s=380, facecolors='none', edgecolors='m')
    m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=400, facecolors='none', edgecolors='k',linewidths=th*2)
    m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=380, facecolors='none', edgecolors='m',linewidths=th*1)
    # m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='m')
    m.contour(lons, lats,plotvar_masked,[0.1],linewidths=2,colors='grey')
              
if i<2: # rf or tp
    # cbar_min=min_value[i]
    # cbar_max=max_value[i]
    # cbar_step=max_value[i]/24
    # cbar_num_format = "%d"

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    # # plt.colorbar(im)            
    # cbar = plt.colorbar(pp,
    #                     orientation='vertical',
    #                     ticks=np.arange(cbar_min,
    #                     cbar_max+cbar_step, cbar_step),format=cbar_num_format, cax=cax)
    # cbar.ax.set_ylabel(units[i], fontsize = font_size)
    # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step).astype(int)
    # # print(tickranges)
    # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)

    # --------------------- mask Iceland Sval with a polygon
    xx0=0.7
    X = np.array([[xx0,0], [xx0,1], [1, 1], [1, 0],[xx0,0]])
    ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
            transform=ax.transAxes,zorder=9)) 
    yy0x=0.35
    # --------------------- colorbar location
    cbaxes = fig.add_axes([xx0-0.04, 0.16, 0.01, yy0x]) 
    cbar = plt.colorbar(pp,orientation='vertical',format="%d",cax=cbaxes, ticks=bounds)
    cbar.ax.minorticks_off()

    # cbar.ax.tick_params(length=0)

    # from matplotlib.ticker import LogFormatter 
    # formatter = LogFormatter(10, labelOnlyBase=False) 

    # cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes, ticks=bounds, format=formatter)
    # ticks=[1,5,10,20,50]
    # cbar.outline.set_linewidth(.4)  
    # cbar.ax.tick_params(width=.4)
    mult=0.6
    cbar.ax.set_yticklabels(bounds, fontsize=font_size*mult)
    
    #units
    mult=0.7
    yy0=yy0x+0.18 ; dy2=-0.03 ; cc=0
    plt.text(xx0+0.02, yy0+cc*dy2,'mm', fontsize=font_size*mult,
             transform=ax.transAxes, color='k') ; cc+=1. 

if i==2: # t2m
    cbar_min=min_value[i]
    cbar_max=max_value[i]
    cbar_step=max_value[i]/24
    cbar_num_format = "%d"

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    # plt.colorbar(im)            
    cbar = plt.colorbar(pp,orientation='vertical',format=cbar_num_format, cax=cax)
    cbar.ax.set_ylabel(units[i], fontsize = font_size)
    # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
    # # print(tickranges)
    # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)

# cc=0
# xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
# mult=0.7
# color_code='grey'
# plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
#   transform=ax.transAxes,color=color_code) ; cc+=1. 


if i<2:
    cc=0
    xx0=0.44 ; yy0=0.17 ; dy2=-0.028
    mult=0.8
    color_code='k'
    print()
    plt.text(xx0, yy0+cc*dy2,casex2, fontsize=font_size*mult,
      transform=ax.transAxes,color=color_code) ; cc+=1.

    msg="{:.1f}".format(mass)+" Gt "+varnam2[i]+" total mass flux"
    print(msg)
    plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
      transform=ax.transAxes,color=color_code) ; cc+=1. 

    msg="{:.0f}".format(np.max(plotvar))+" mm "+"max "+varnam2[i]+""
    print(msg)
    plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
      transform=ax.transAxes,color=color_code) ; cc+=1. 

    msg="{:.4f}".format(alllat)+" N, "+"{:.4f}".format(abs(alllon))+" W"
    print(msg)
    plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
      transform=ax.transAxes,color=color_code) ; cc+=1. 
    
    # msg="{:.0f}".format(np.max(plotvar[v]))+" mm / y "+"max "+varnam2[i]+" SSW"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

    # msg="{:.4f}".format(SSWlat)+" N, "+"{:.4f}".format(abs(SSWlon))+" W"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

if ly == 'x':
    plt.show()


if ly =='p':
    figpath='/Users/jason/Dropbox/CARRA/CARRA rainfall study/Figs/'
    # os.system('mkdir -p '+figpath)
    # figpath='./Figs/annual/'+varnams[i]+'/'+str(DPI)+'/'
    # os.system('mkdir -p '+figpath)
    figname=figpath+'rf_'+CARRA_or_ERA5+'_'+casex
    # if i<2:
    plt.savefig(figname+'.pdf', bbox_inches='tight')
    # else:
    #     plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)#, facecolor=fig.get_facecolor(), edgecolor='none')
    #%%

#%%
make_gif=0

if make_gif:
    animpath='/Users/jason/Dropbox/CARRA/anim/'
    # inpath='/Users/jason/Dropbox/CARRA/Figs/'
    inpath=figpath
    msg='convert  -delay 70  -loop 0   '+inpath+'*'+varnams[i]+'*'+'.png  '+animpath+varnams[i]+'_1991-2021.gif'
    os.system(msg)