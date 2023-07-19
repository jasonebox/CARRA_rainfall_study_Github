#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:12:42 2023

@author: jason
"""


import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import calendar
from datetime import timedelta
# from mpl_toolkits.basemap import Basemap

os.chdir('/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/')

# ------------------------------------------- global plot settings
th=1  # line thickness
font_size=16
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'

##%%
# ly='p' # x for console, p for out to png file
# res='h' # low res for console, high res for png
# if ly=='p':res='h'
subset_it=0 # subset south Greenland Q transect

ni=1269 ; nj=1069

fn='/Users/jason/Dropbox/CARRA/ancil/CARRA_W_domain_elev.nc'
ds=xr.open_dataset(fn)
# print(ds)
elev=ds.z
# print(elev.shape)
# elev[elev < 1] = np.nan


fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)

subset_name=''; lonrot=0

max_value=12
max_value=36

elev0=50;elev1=3000;delev=500

if subset_it:
    elev0=50;elev1=3000;delev=200
    # 50,3000,200
    max_value=24
    max_value=36
    subset_name='subset_'
    lon=np.rot90(lon.T)
    lat=np.rot90(lat.T)
    xc0=190 ; xc1=400
    yc0=950 ; yc1=1210
    xc0=190 ; xc1=420
    yc0=890 ; yc1=1210
    xc0=150 ; xc1=520 # wider and taller
    yc0=550 ; yc1=1210
    
    lat=lat[yc0:yc1,xc0:xc1]
    lon=lon[yc0:yc1,xc0:xc1]
    # elev=elev[yc0:yc1,xc0:xc1]
    # print(lat.shape)
    ni=lat.shape[0]
    nj=lat.shape[1]
    lon=np.rot90(lon.T)
    lat=np.rot90(lat.T)
    
    elev=np.rot90(elev.T)
    elev=elev[yc0:yc1,xc0:xc1]
    elev=np.rot90(elev.T)
    lonrot=11.25
    lonrot=11.5
    # plt.imshow(elev)
    # saas

# # --------------------------------------------------------- set up projection
# LLlat=lat[0,0]
# LLlon=lon[0,0]-360
# # LLlat=55.81
# # LLlon=-57.09698486328125
# # print("LL",LLlat,LLlon)
# # print("UL",lat[ni-1,0],lon[ni-1,0]-360)
# lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
# lat0=lat[int(round(ni/2)),int(round(nj/2))]
# # print("mid lat lon",lat0,lon0)
# # lon0=-35.999969482421875
# # lat0=72.00049
# URlat=lat[ni-1,nj-1]
# URlon=lon[ni-1,nj-1]
# # URlat=77.83195
# # URlon=37.63491
# # print("LR",lat[0,nj-1],lon[0,nj-1]-360)
# # print("UR",URlat,URlon)

# m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat,\
#             lat_0=lat0, lon_0=lon0+lonrot, \
#             resolution=res, projection='lcc')

# x, y = m(lon, lat)

#%%
#functions to read in files
def gett2m(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    t2m=ds.variables['t2m'].values-273.15
    # print(np.shape(t2m))
    # asas
    T=np.mean(t2m[:,:,:],axis=0)
    time=ds.variables['time'].values
    return T,time

def get_vel(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    # print(ds.variables)
    P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u'].values
    V=ds.variables['v'].values
    W=ds.variables['wz'].values
    time=ds.variables['time'].values
    # P=P[0]
    return U,V,W,P,time


varx=['t2m']

for var in varx:
    # fn='/Users/jason/0_dat/CARRA/202209/no-ar-cw_vel_an_pl_202209'+day.zfill(2)+'.grib'
    # ds=xr.open_dataset(fn,engine='cfgrib')
    # # 
    # Ux,Vx,Wx,P,timex=get_vel(fn)
    # print(np.shape(Ux))
    
    fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_t2m.grib'
    t2m,timex=gett2m(fn)
    
    time=pd.to_datetime(timex)
    
    
    # print(np.shape(t2m))

    ihour=0 ; fhour=len(time)
    n_hours=fhour-ihour+1
    # import datetime
    
    plt.close()
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
    # print(str(time.strftime('%Y %b %d')))
               
    skip_interval=30 # how often to skip vector quiver arrows... CARRA data is so dense
    
    if var=='t2m':
        plotvar = t2m
        lo=-12; hi=-lo
        dx=(hi-lo)/40
        units='°C'
        # units=str(int(P))+'hPa\n°C'

        # quiver_color='k'
        # skip_interval=8
        # varnam='CARRA\nnear surface\nair temperature'
        cm = plt.cm.bwr
        # cm.set_under('purple')
        # cm.set_over('orange')
        quiver_color='k'
        clevs = np.arange(lo,hi+dx,dx)

    # if subset_it:
    #     U=np.rot90(Ux[i,:,:].T)
    #     U=U[yc0:yc1,xc0:xc1]
    #     V=np.rot90(Vx[i,:,:].T)
    #     V=V[yc0:yc1,xc0:xc1]
    #     # V=np.rot90(V.T)        
    #     plotvar=np.rot90(plotvar.T)
    #     plotvar=plotvar[yc0:yc1,xc0:xc1]
    #     plotvar=np.rot90(plotvar.T)
    
    # if subset_it==0:skip_interval=30

    # mcmap = plt.get_cmap(cmap_choice)
    # mcmap.set_over('orange')
    
    plotvar[plotvar>=np.max(clevs)]=np.max(clevs)-0.01
    plotvar[plotvar<=lo]=lo

    # plt.title(str(time[i].strftime('%Y %b %d %H')),c='k')
    # print(dates[i])
    
    # inv=np.where(lats<50)
    # plotvar[inv[0]]=np.nan
    # print("min",np.nanmin(plotvar),"max",np.nanmax(plotvar))

    # m.drawcoastlines()
    # m.drawparallels(np.arange(-90.,120.,30.))
    # m.drawmeridians(np.arange(0.,420.,50.))
    
    mine=plt.contourf(plotvar,clevs,cmap=cm,extend='both')
    clevs=np.arange(elev0,elev1,delev)

    plt.clim(lo,hi)

    # du_color_bar=1
    # if du_color_bar:
    #     clb = plt.colorbar(fraction=0.046/1.5, pad=0.05)
    #     clb.ax.set_title(units,fontsize=font_size,c='k',ha='left')

    # plt.contour(plotvar,[0],linewidths=th,colors='black',zorder=15)
    ax.axis('off')

    # skip=(slice(None,None,skip_interval),slice(None,None,skip_interval))

    # du_vect=0 # quiver map
    # if du_vect:
    #     v=np.where(elev>8800)
    #     xx=x.copy(); xx[v]=np.nan
    #     yy=y.copy(); yy[v]=np.nan
    #     U[v]=np.nan
    #     V[v]=np.nan
    #     m.quiver(xx[skip],yy[skip], U[skip], V[skip], scale_units='inches', scale=80, color=quiver_color,width=0.0015,zorder=16)
    #     # speed = np.sqrt(U*U + V*V)
    #     # xx, yy = map.makegrid(V.shape[1], U.shape[0], returnxy=True)[2:4]
    #     # m.streamplot(y, x, U, V, color=speed, cmap=plt.cm.autumn, linewidth=0.5*speed)
    #     # np.shape(x)
    #     # np.shape(y)
    #     # m.streamplot(x, y, U, V, broken_streamlines=False)


    # ----------- annotation
    
    
    du_color_bar=1
    if du_color_bar:
        # units=str(int(P))+'hPa\n°C'
        width=0.025
        cbax = ax.inset_axes([1.02, 0.005, width, 0.41], transform=ax.transAxes)
        cbax.set_title(units,fontsize=font_size*0.9,c='k',ha='left')
        cbax.tick_params(labelsize=font_size*0.9)
        fig.colorbar(mine, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')
       
    # xx0=1.2 ; yy0=0.8
    # mult=1.1
    # color_code='k'
    # props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
    # plt.text(xx0, yy0, varnam,#+'\nand\n900hPa winds',
    #         fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes,zorder=20,ha='center')
    
    # xx0=1.03 ; yy0=0.97
    # mult=1.1
    # color_code='k'
    # plt.text(xx0, yy0, time[i].strftime('%Y %b %d %H'),
    #         fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes,zorder=20)

    # xx0=1.03 ; yy0=0.01
    # mult=0.9
    # color_code='grey'
    # plt.text(xx0, yy0, 'quiver each '+str(skip_interval)+'\n2.5 km grid points',
    #         fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
    
    # cc=0
    # xx0=1.02 ; yy0=0.02 ; dy2=-0.04
    # mult=1
    # color_code='grey'
    
    # plt.text(xx0, yy0+cc*dy2, '@climate_ice',
    #          fontsize=font_size*mult,color=color_code,rotation=90,transform=ax.transAxes) ; cc+=1.5
    
    # xx0=0.5 ; yy0=0.5 ; dy=0.1 ; dx=0.2
    # currentAxis = plt.gca()
    # currentAxis.add_patch(Rectangle((0. , .0), cc/n_hours, 0.01,alpha=1,transform=ax.transAxes))
    # cc+=1
    
    ly='x'

    if ly == 'x':
        plt.show() 

    DPI=300
    
    if ly == 'p':
        # fig_basepath='./Figs/'
        figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/streamline/'
        # figpath=fig_basepath+var+'/'
        # os.system('mkdir -p '+figpath)
        # figpath=fig_basepath+var+'/'+event+'/'
        # os.system('mkdir -p '+figpath)
        figname=figpath+'t2m_'+time[0].strftime('%Y_%m_%d')
        # figpath='./Figs/'+var+'/'
        # os.system('mkdir -p '+figpath)
        # figpath='./Figs/'+var+'/'+event+'/'
        # os.system('mkdir -p '+figpath)
        # figname=figpath+time[i].strftime('%Y %m %d %H')
        plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

    make_gif=0
    
    if make_gif:
        print("making gif")
        animpath='./anim/'
        os.system('mkdir -p '+'./anim/')
        inpath=figpath
        msg='convert  -delay 20  -loop 0   '+figpath+'*.png  '+animpath+var+'_'+event+'_'+str(DPI)+'DPI_wider.gif'
        os.system(msg)