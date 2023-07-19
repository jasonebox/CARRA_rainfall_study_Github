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
from mpl_toolkits.basemap import Basemap

os.chdir('/Users/jason/Dropbox/CARRA/CARRA_202209/')


#%%
# ------------------------------------------- global plot settings
th=1  # line thickness
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'

ly='p' # x for console, p for out to png file
res='h' # low res for console, high res for png
if ly=='p':res='h'

ni=1269 ; nj=1069

# fn='/Users/jason/Dropbox/CARRA/ancil/CARRA_W_domain_elev.nc'
# ds=xr.open_dataset(fn)
# # print(ds)
# elev=ds.z
# # print(elev.shape)
# # elev[elev < 1] = np.nan


# fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
# lat=np.fromfile(fn, dtype=np.float32)
# lat=lat.reshape(ni, nj)

# fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
# lon=np.fromfile(fn, dtype=np.float32)
# lon=lon.reshape(ni, nj)

# subset_name=''; lonrot=0


# elev0=50;elev1=3000;delev=500

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

# #%%

# ly='p'
# fig, ax = plt.subplots(figsize=(10, 10*ni/nj))

# m.drawcoastlines()

# if ly == 'x':
#     plt.show() 

# DPI=100

# if ly == 'p':
#     figpath='./Figs/'
#     os.system('mkdir -p '+figpath)
#     figname=figpath+'coastline'
#     plt.savefig(figname+'.svg', bbox_inches='tight', facecolor=None, edgecolor=None)
#%%
#functions to read in files
# def gett2m(fn):
#     print(fn)
#     ds=xr.open_dataset(fn,engine='cfgrib')
#     t2m=ds.variables['t2m'].values-273.15
#     time=ds.variables['time'].values
#     return t2m,time

def get_UVW(fn):
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

# def get_UV(fn):
#     print(fn)
#     ds=xr.open_dataset(fn,engine='cfgrib')
    
#     # print(ds.variables)
#     # P=np.array(ds.variables['isobaricInhPa'].values)
#     U=ds.variables['u'].values
#     V=ds.variables['v'].values
#     # W=ds.variables['wz'].values
#     time=ds.variables['time'].values
#     # P=P[0]
#     return U,V,time

def get_UV(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    
    # print(ds.variables)
    # P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u10'].values
    V=ds.variables['v10'].values
    # T=ds.variables['t2m'].values-273.15
    # W=ds.variables['wz'].values
    time=ds.variables['time'].values
    # P=P[0]
    return U,V,time

def get_t2m(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    T=ds.variables['t2m'].values-273.15
    # W=ds.variables['wz'].values
    time=ds.variables['time'].values
    # P=P[0]
    return T,time

days=np.arange(14,15).astype(str) ; event='2017-09-14'

ly='p'
density=2
hour0=0 ; hour1=7

test=0
subset_it=0 # subset south Greenland 

if test:
    density=1
    ly='x'
    hour0=0 ; hour1=1
    days=np.arange(2,3).astype(str) ; event='test'


for i,day in enumerate(days):
    print(i,day)
    # if int(day)>=0:
    # if int(day)==2:
    if i>=0:
        # fn=
        # ds=xr.open_dataset(fn,engine='cfgrib')
        # ds.variables
        # list(ds.keys())

        # Ux,Vx,Wx,P,timex=get_UVW(fn)
        Ux,Vx,timex=get_UV('/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_u10v10.grib')
        Tx,timex=get_t2m('/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_t2m.grib')
        # print(np.shape(Ux))
        # units=str(int(P))+'hPa\nm s$^{-1}$'

        # fn='/Users/jason/0_dat/CARRA/202209/no-ar-cw_t2m_fc_sfc_202209'+day.zfill(2)+'.grib'
        # t2m,timex=gett2m(fn) 
        
        time=pd.to_datetime(timex)


        units='10 m winds,\nm s$^{-1}$'

        # for i in range(hour0+1,hour1):
        # for i in range(hour0,hour0+1):
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
        print(str(time[i].strftime('%Y %b %d %H')))
        
        if subset_it:
            subset_name='subset_'
            xc0=190 ; xc1=400
            yc0=950 ; yc1=1210
            xc0=190 ; xc1=420
            yc0=890 ; yc1=1210
            xc0=150 ; xc1=520 # wider and taller
            yc0=550 ; yc1=1210
            U=Ux[i,yc0:yc1,xc0:xc1]
            V=Vx[i,yc0:yc1,xc0:xc1]
            ni=yc1-yc0 ; nj=xc1-xc0
            Y, X = np.mgrid[0:ni,0:nj]
            density=1
        else:
            U=np.mean(Ux[:,:,:],axis=0)
            V=np.mean(Vx[:,:,:],axis=0)
            T=np.mean(Tx[:,:,:],axis=0)
            Y, X = np.mgrid[0:ni,0:nj]

        speed = np.sqrt(U**2 + V**2)

        lw = 1*speed / speed.max()

        color_by_speed=0
        
        if color_by_speed:
            color_var=speed
            cm_name='viridis'
            cmap = plt.cm.get_cmap(cm_name)
        else:
            cm_name='bwr'
            cmap = plt.cm.get_cmap(cm_name)
            extreme=10.
            lo=-extreme; hi=-lo
            color_var=T
            # hi=np.min(color_var)
            color_var[color_var<-extreme]=-extreme
            color_var[color_var>extreme]=extreme
            # color_var[0:4,0:4]=15.; color_var[ni-5:ni-1,nj-5:nj-1]=-15.
            
        strm = ax.streamplot(X, Y, U, V,density=density,color=color_var,cmap=cm_name,linewidth=lw,arrowstyle='->',
                     broken_streamlines=False)
        # ax.set_clim(vmin=-his[lev_index], vmax=his[lev_index])

        # ax.grid(off)
        ax.axis('off')

        # xx0=0.65 ; yy0=0.02
        # mult=0.9
        # color_code='k'
        # props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')

        # ax.text(xx0, yy0, time[i].strftime('%d %b %Y %HUTC'),
        #         fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes,zorder=20)

        # cbar=plt.colorbar(strm.lines,fraction=0.03, pad=0.04)
        # # plt.clim
        # # cbar.set_label(units, rotation=0, labelpad=8)
        # fs=15
        # cbar.ax.set_title(units, rotation=0,ha='center',fontsize=fs)#, labelpad=8)
        # cbar.ax.tick_params(labelsize=fs)
        
        du_color_bar=0
        with_colorbarname=''
        if du_color_bar:
            with_colorbarname='_w_colorbar'
            width=0.03
            cbax = ax.inset_axes([1.02, 0.5, width, 0.5], transform=ax.transAxes)
            cbax.set_title(units,fontsize=font_size,c='k',ha='left')
            fig.colorbar(strm.lines, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')


        # clb.set_clim(0,50)
        # cbar.draw_all()
        if ly == 'x':
            plt.show() 
    
        DPI=300
        
        if ly == 'p':
            var='streamline'
            fig_basepath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'
            figpath=fig_basepath+var+'/'
            os.system('mkdir -p '+figpath)
            figpath=fig_basepath+var+'/'+event+'/'
            os.system('mkdir -p '+figpath)
            figpath=fig_basepath+var+'/'+event+'/'+cm_name+'/'
            os.system('mkdir -p '+figpath)
            figname=figpath+time[i].strftime('%Y %m %d')+'_surface'
            plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, transparent=True)#, facecolor='w', edgecolor='k')
            # figname=figpath+'case'
            # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

    # make_gif=0
    
    # if make_gif:
    #     print("making gif")
    #     animpath='./anim/'
    #     os.system('mkdir -p '+'./anim/')
    #     inpath=figpath
    #     msg='convert  -delay 20  -loop 0   '+figpath+'*.png  '+animpath+var+'_'+event+'_'+str(DPI)+'DPI_wider.gif'
    #     os.system(msg)