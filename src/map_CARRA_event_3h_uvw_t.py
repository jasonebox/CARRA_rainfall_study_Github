#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:52:21 2021

@author: jeb
"""
import numpy as np
import os
import xarray as xr
import datetime 
# from matplotlib.patches import Rectangle 
import pandas as pd
import datetime 
import matplotlib.pyplot as plt
import matplotlib as mpl
# from matplotlib import cm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
wo=0

# ------------------------------------------- set paths

working_path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
inpath='/Users/jason/0_dat/CARRA/' # location of large .nc files input to this baby
outpath='/Users/jason/0_dat/CARRA/output/'

script_path = '/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/'
    
os.chdir(working_path)
# ------------------------------------------- set paths




# ------------------------------------------- global plot settings
th=1  # line thickness
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'



def get_CARRA(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    list(ds.keys())
    print(ds.variables)
    U=ds.variables['u10'].values
    V=ds.variables['v10'].values
    SHF=ds.variables['sshf'].values/(3*3600)
    LHF=ds.variables['slhf'].values/(3*3600)
    STR=ds.variables['str'].values/(3*3600)
    times=ds.variables['time'].values
    return U,V,SHF,LHF,STR,times


ni=1269 ; nj=1069


#%% Load elevation and prepare map   

fn= script_path+ '/ancil/CARRA_W_domain_elev.nc'
ds_elev = xr.open_dataset(fn)
# print(ds)
elev=ds_elev.z
# print(elev.shape)
# elev[elev < 1] = np.nan


fn=script_path + '/ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn= script_path + '/ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)

subset_name=''; lonrot=0

max_value=12
max_value=36

# elev0=0;elev1=1000;delev=50
elev0=50;elev1=3000;delev=500
p0=950;p1=1050;dp=10
subset_it=0
if subset_it:
    elev0=50;elev1=3000;delev=200
    # 50,3000,200
    max_value=24
    max_value=36
    subset_name='subset_'
    lon=np.rot90(lon.T)
    lat=np.rot90(lat.T)
    xc0=200 ; xc1=450 # 250
    yc0=950 ; yc1=1200 # 250
    xc0=210 ; xc1=400
    yc0=1000 ; yc1=1210

    lat=lat[yc0:yc1,xc0:xc1]
    lon=lon[yc0:yc1,xc0:xc1]
    # print(lat.shape)
    ni=lat.shape[0]
    nj=lat.shape[1]
    lon=np.rot90(lon.T)
    lat=np.rot90(lat.T)
    
    elev=np.rot90(elev.T)
    elev=elev[yc0:yc1,xc0:xc1]
    elev=np.rot90(elev.T)
    lonrot=11.25
    # plt.imshow(elev)
    # saas

# --------------------------------------------------------- set up projection
LLlat=lat[0,0]
LLlon=lon[0,0]-360
# LLlat=55.81
# LLlon=-57.09698486328125
# print("LL",LLlat,LLlon)
# print("UL",lat[ni-1,0],lon[ni-1,0]-360)
lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
lat0=lat[int(round(ni/2)),int(round(nj/2))]
# print("mid lat lon",lat0,lon0)
# lon0=-35.999969482421875
# lat0=72.00049
URlat=lat[ni-1,nj-1]
URlon=lon[ni-1,nj-1]
# URlat=77.83195
# URlon=37.63491
# print("LR",lat[0,nj-1],lon[0,nj-1]-360)
# print("UR",URlat,URlon)
# if zoom == 'QASL':
#     LLlat=60.8
#     LLlon=-47.2
#     URlat=61.2
#     URlon=-46.2
    
#     lon0 = -47.6
#     lat0 = 61

# if zoom == 'South_GRL':
#     LLlat=59.5
#     LLlon=-50.5
#     URlat=62.5
#     URlon=-40.5
    
#     lon0 = -45
#     lat0 = 61

#%%
# ------------------------------------------- parameters

year='2017' ; month='09' ; day='13' ; casex=year+month+day
year='2017' ; month='09' ; day='14' ; casex=year+month+day
year='2017' ; month='09' ; day='15' ; casex=year+month+day

# year='2012' ; month='07' ; day='10' ; casex=year+month+day
# year='2012' ; month='07' ; day='09' ; casex=year+month+day
# year='2012' ; month='07' ; day='08' ; casex=year+month+day
# year='2012' ; month='07' ; day='12' ; casex=year+month+day

# year='2021' ; month='08' ; day='14' ; casex=year+month+day
# year='2021' ; month='08' ; day='15' ; casex=year+month+day
# year='2021' ; month='08' ; day='16' ; casex=year+month+day

# years=['2012','2017','2021','2022']

# years=['2022']

years=['2017']

for year in years:
    if year=='2012':
        days=np.arange(8,13).astype(str) # 2012
        month='07'

    if year=='2017':
        days=np.arange(13,16).astype(str) # 2012
        month='09'
        
    if year=='2021':
        days=np.arange(13,16).astype(str) # 2012
        month='08'

    if year=='2022':
        days=np.arange(1,6).astype(str) # 2012
        month='09'
    
    for dd,day in enumerate(days):
        # if dd==0:
        if dd>=0:
            casex=year+month+day.zfill(2)
            fn='/Users/jason/0_dat/CARRA/202209_CDS/'+casex+'_CARRA_u10v10t2mLHF_SHF_LWNet.grib'
            figpath='/Users/jason/0_dat/CARRA/Figs_SEB/'
            
            Ux,Vx,SHF,LHF,STR,times=get_CARRA(fn)
            times=pd.to_datetime(times)
            
            var='SHF' # surface sensible heat flux
            var='LHF' # surface latent heat flux
            # var='STR' # surface longwave
            
            var='THF' # surface latent heat flux
            
            # var='wind'
            # var='w' # vertical velocity
            skip_interval=30 # how often to skip vector quiver arrows... CARRA data is so dense
            ly='p' # x for console, p for out to png file
            res='l' # low res for console, high res for png
            subset_it=0 # subset south Greenland Q transect
            du_vect=1 # quiver map
            du_surf=0 # surface contours 
            du_color_bar=1 # add colorbar
            zoom = ''
            
            if ly=='p':
                res='h'
            else:
                res='l'
            
            m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat,\
                        lat_0=lat0, lon_0=lon0+lonrot, \
                        resolution=res, projection='lcc')
            
            x, y = m(lon, lat)
            # Plot data on map
            ihour=0 ; fhour=len(times)
            n_hours=fhour-ihour+1
            
            cc=0
            for i in range(ihour,fhour):
                if i==0:
                # if i>=0:
                    plt.close()
                    plt.clf()
                    fig, ax = plt.subplots(figsize=(8, 14))
                    
                    print(str(times[i]))
                       
                    if du_vect:
                        U = Ux[i,:,:]
                        V = Vx[i,:,:]
            
                    if var=='THF':
                        plotvar = SHF[i,:,:] + LHF[i,:,:]
                        lo=-300; hi=300
                        # lo=-150; hi=150
                        # lo=-100; hi=100
                        dx=(hi-lo)/20
                        units='W m$^{-2}$\n'
                        prefix="CARRA surface turbulent heat flux, "
                        cm=plt.cm.seismic
                        cm.set_under('purple')
                        cm.set_over('orange')
                        quiver_color='k'
                        skip_interval=5
                
                    if var=='SHF':
                        plotvar = SHF[i,:,:]
                        lo=-200; hi=200
                        # lo=-150; hi=150
                        # lo=-100; hi=100
                        dx=(hi-lo)/20
                        units='W m$^{-2}$\n'
                        prefix="CARRA surface sensible heat flux, "
                        cm=plt.cm.seismic
                        cm.set_under('purple')
                        cm.set_over('orange')
                        quiver_color='k'
                        skip_interval=5
            
                    if var=='LHF':
                        plotvar = LHF[i,:,:]
                        # lo=-200; hi=200
                        lo=-150; hi=150
                        # lo=-100; hi=100
                        dx=(hi-lo)/20
                        units='W m$^{-2}$\n'
                        prefix="CARRA surface latent heat flux, "
                        cm=plt.cm.seismic
                        cm.set_under('purple')
                        cm.set_over('orange')
                        quiver_color='k'
                        skip_interval=5
                        
                    if var=='STR':
                        plotvar = STR[i,:,:]
                        lo=-200; hi=200
                        # lo=-150; hi=150
                        # lo=-100; hi=100
                        dx=(hi-lo)/20
                        units='W m$^{-2}$\n'
                        prefix="CARRA surface infrared flux, "
                        cm=plt.cm.seismic
                        cm.set_under('purple')
                        cm.set_over('orange')
                        quiver_color='k'
                        skip_interval=5
            
                        
                    # if var=='LHF':
                    #     # if event=='20170913-15':
                    #     #     U = ds['u'].values[i,0,:,:]
                    #     #     V = ds['v'].values[i,0,:,:]
                    #     #     plotvar = nds['wz'].values[i,0,:,:]
                    #     # else:
                    #     #     U = ds['u'].values[i,:,:]
                    #     #     V = ds['v'].values[i,:,:]
                    #         # ds['v'].values.shape            
                    #     plotvar = ds['slhf'].values[i,:,:]
                    #     # plt.imshow(plotvar)
                    #     # plt.colorbar()
                    #     lo=-150; hi=150
                    #     dx=(hi-lo)/20
                    #     units='W m$^{-2}$'
                    #     prefix="CARRA surface latent heat flux, "
                    #     cm=plt.cm.seismic
                    #     cm.set_over('r')
                    #     cm.set_over('orange')
            
                    #     quiver_color='k'
                    #     skip_interval=5
                    if subset_it:
                        if event!='2010-10_02-06':
                            U=np.rot90(U.T)
                            U=U[yc0:yc1,xc0:xc1]
                            U=np.rot90(U.T)
                            V=np.rot90(V.T)
                            V=V[yc0:yc1,xc0:xc1]
                            V=np.rot90(V.T)        
                        plotvar=np.rot90(plotvar.T)
                        plotvar=plotvar[yc0:yc1,xc0:xc1]
                        plotvar=np.rot90(plotvar.T)
                
                    if subset_it==0:  skip_interval=30
                    clevs = np.arange(lo,hi+dx,dx)
                
                    # mcmap = plt.get_cmap(cmap_choice)
                    # mcmap.set_over('orange')
                    
                    # plotvar[plotvar>=np.max(clevs)]=np.max(clevs)
                    # plotvar[plotvar<=lo]=lo
                
                    # plt.title(prefix+'\n'+times[i].strftime('%Y %b %d %H')+' UTC',c='k')
                    # print(dates[i])
                    
                    # inv=np.where(lats<50)
                    # plotvar[inv[0]]=np.nan
                    # print("min",np.nanmin(plotvar),"max",np.nanmax(plotvar))
                
                    m.drawcoastlines(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
                    # m.drawparallels(np.arange(-90.,120.,30.))
                    # m.drawmeridians(np.arange(0.,420.,50.))
                    
                    mine=m.contourf(x,y,plotvar,clevs,cmap=cm,linewidths=th/1.)#,extend='both')
                
                        
                    clevs_elev =np.arange(elev0,elev1,delev)
                    clevs_ps =np.arange(p0,p1,dp)
                    
                    plt.clim(lo,hi)
                
                
                    m.contour(x,y,elev,clevs_elev,linewidths=th/2,colors='black',zorder=15)
            
                    cc=0
                    xx0=0.012 ; yy0=0.967
                    mult=0.8
                    color_code='k'
                    props = dict(facecolor='w', alpha=1,edgecolor='w')
                    plt.text(xx0, yy0, prefix+' '+times[i].strftime('%Y %b %d %H')+'UTC',c='k',
                            fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes,zorder=20) ; cc+=1.5
            
                
                    skip=(slice(None,None,skip_interval),slice(None,None,skip_interval))
                
                    if du_vect:
                        m.quiver(x[skip],y[skip], U[skip], V[skip], scale_units='inches', scale=80, color=quiver_color,width=0.002,zorder=16)
                    # if du_surf:
                    #     m.contour(x,y,Ps,clevs_ps,linewidths=th/2,colors='red',zorder=14)
                    # if plotAWS:
                    #     m.scatter(xAWS, yAWS, s = 100, marker='^',color='black', edgecolors='white')
                    if du_color_bar:
                        clb = plt.colorbar(mine,fraction=0.03, pad=0.04,ticks=clevs[0::2])
                        clb.ax.set_title(units,fontsize=font_size,c='k',loc='left',pad = 0.1)
            
                    do_colorbar=0
                    
                    if do_colorbar:
                        # clb = plt.colorbar(fraction=0.035, pad=0.05)
                        # clb.ax.tick_params(labelsize=font_size,labelrotation = 'auto')
                        cbax = ax.inset_axes([1.07, 0.0, 0.03, 0.5], transform=ax.transAxes)
                        clb=fig.colorbar(mine, ax=ax, cax=cbax, shrink=0.7,orientation='vertical')
                        clb.ax.set_title(units,fontsize=font_size,c='k')
                    # ----------- annotation
                    cc=0
                    xx0=1.01 ; yy0=0.01
                    mult=0.6
                    color_code='grey'
                    
                    plt.text(xx0, yy0+cc, '10 m wind vectors\neach 30 CARRA\ngrid points',
                              fontsize=font_size*mult,color='k',rotation=0,transform=ax.transAxes) ; cc+=1.5
                    
                    # xx0=0.5 ; yy0=0.5 ; dy=0.1 ; dx=0.2
                    # currentAxis = plt.gca()
                    # currentAxis.add_patch(Rectangle((0. , .0), cc/n_hours, 0.01,alpha=1,transform=ax.transAxes))
                    # cc+=1
                    
                    if ly == 'x':
                        plt.show() 
                    
                    DPI=150
                                
                    if ly == 'p':
                        figpath='/Users/jason/0_dat/CARRA/Figs_SEB/'+var+'/'
                        os.system('mkdir -p '+figpath)
                        figname=figpath+times[i].strftime('%Y%m%d%H')
                        plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                        # plt.savefig(figname+'.svg', bbox_inches='tight')

