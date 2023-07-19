#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:52:21 2021

@author: jeb
"""
import numpy as np
import os
import datetime 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray as xr
import geopandas as gpd
from pyproj import Proj, transform
from netCDF4 import Dataset

path='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/'
os.chdir(path)

ni=1269 ; nj=1069

fn='./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
elev=np.fromfile(fn, dtype=np.float32)
elev=elev.reshape(ni, nj)
elev=np.rot90(elev.T)
# plt.imshow(elev)

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
mask=np.rot90(mask.T)

# plt.imshow(mask)

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)
lat=np.rot90(lat.T)

def lon360_to_lon180(lon360):

    # reduce the angle  
    lon180 =  lon360 % 360 
    
    # force it to be the positive remainder, so that 0 <= angle < 360  
    lon180 = (lon180 + 360) % 360;  
    
    # force into the minimum absolute value residue class, so that -180 < angle <= 180  
    lon180[lon180 > 180] -= 360
    
    return lon180

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)
lon = lon360_to_lon180(lon)
lon=np.rot90(lon.T)

inpath='/Users/jason/0_dat/CARRA/'
# inpath='/Volumes/LaCie/0_dat/CARRA/'
outpath='/Users/jason/0_dat/CARRA/output/'

# # ------------------------------------------- rain fraction
# rainos=0.
# x0=0.5 ; x1=2.5
# x0-=rainos
# x1-=rainos
# y0=0 ; y1=1
# a1=(y1-y0)/(x1-x0)
# a0=y0-a1*x0

year='2017' ; month='09' ; day='14' ; casex=year+month+day+'_18'
year='2017' ; month='09' ; day='14' ; casex=year+month+day+'_15'
# year='2012' ; month='07' ; day='10' ; casex=year+month+day
fn='/Users/jason/0_dat/CARRA/202209_CDS/'+casex+'UTC_CARRA_multivars_1000-300hPa_all_levsx.grib'
# fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_15UTC_CARRA_multivars_1000-300hPa_all_levsx.grib'

def get_wc(fn):
    #'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 
    #        'specific_cloud_rain_water_content',
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    print(ds.variables)
    list(ds.keys())
    # P=np.array(ds.variables['isobaricInhPa'].values)
    cswc=ds.variables['cswc'].values
    crwc=ds.variables['crwc'].values
    clwc=ds.variables['clwc'].values
    ciwc=ds.variables['ciwc'].values

    # time=ds.variables['time'].values
    # P=P[0]
    return cswc,crwc,clwc,ciwc

# fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_rh_1000-300hPa_14_levs.grib'
cswcx,crwcx,clwcx,ciwcx=get_wc(fn)

var='w'

# U V W wind components
def get_UVW(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    list(ds.keys())
    print(ds.variables)
    P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u'].values
    V=ds.variables['v'].values
    W=ds.variables['wz'].values
    time=ds.variables['time'].values
    return U,V,W,P,time

Ux,Vx,Wx,Px,timesx=get_UVW(fn)

# total precipitation
def get_TP(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    # print(ds.variables)
    P=np.array(ds.variables['isobaricInhPa'].values)
    T=ds.variables['t'].values#-273.15
    papt=ds.variables['papt'].values
    time=ds.variables['time'].values
    return T,papt,P,time

Tx,paptx,P,times=get_TP(fn)
time=pd.to_datetime(times)
levs=np.array(P)
print(levs)
print(len(levs))

print(np.shape(paptx))
# fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvw_1000-300hPa_14_levs.grib'

# relative humidity
def get_rh(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    # print(ds.variables)
    # list(ds.keys())
    # P=np.array(ds.variables['isobaricInhPa'].values)
    rh=ds.variables['r'].values
    # T=ds.variables['t'].values-273.15
    time=ds.variables['time'].values
    # P=P[0]
    return rh

# fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_rh_1000-300hPa_14_levs.grib'
rhx=get_rh(fn)

#%% plot

# contour line labeling
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"
    # return rf"{s} \K" if plt.rcParams["text.usetex"] else f"{s}K"

plt_map=1 # if set to 1 draws a map to the right of the graphic
ly='p'
by_date=1

# global plot settings
th=1
font_size=20
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['axes.grid'] = False
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.alpha'] = 0.8
# plt.rcParams['grid.color'] = "k"

x0=302
xs=np.arange(214,380) # range of x values on CARRA grid
xs=np.arange(253,283)
# xs=np.arange(283,450)

xs=np.arange(313,314) # figure candidate
# xs=np.arange(300,313) # figure candidate
xs=np.arange(300,301) # figure candidate
xs=np.arange(310,311) # submitted figure 
# xs=np.arange(311,312) # figure candidate
# xs=np.arange(335,336) # figure candidate
# xs=np.arange(360,367) # choice
# xs=np.arange(280,323) # range of x values on CARRA grid for Qagssimiut
# xs=np.arange(322,323) # range of x values on CARRA grid for Qagssimiut
xs=np.arange(210,431) # range of x values on CARRA grid for Qagssimiut
# xs=np.arange(367,431) # range of x values on CARRA grid for Qagssimiut
xs=np.arange(312,313) # figure candidate

p0=1013.25
p0=1000

do_width=0
width=15

var_choice='w'
# var_choice='speed'
# var_choice='V'
# var_choice='U'
# var_choice='rh'
# var_choice='clwc'
# var_choice='ciwc'
# var_choice='crwc'
# var_choice='cswc'

do_pseudo=1
do_isentropes=1

n_levs=16 # until 300 hPa

print(levs[0:n_levs-1])
# levx=n_levs

for x0 in xs:
    print(x0)
    y0=990 ; y1=ni-60
    
    levs=np.array(P[0:n_levs])
    # print(levs)
    theta=np.zeros((n_levs,y1-y0))
    T=np.zeros((n_levs,y1-y0))
    theta_pseudo=np.zeros((n_levs,y1-y0))
    W=np.zeros((n_levs,y1-y0))
    U=np.zeros((n_levs,y1-y0))
    V=np.zeros((n_levs,y1-y0))
    rh=np.zeros((n_levs,y1-y0))
    clwc=np.zeros((n_levs,y1-y0))
    ciwc=np.zeros((n_levs,y1-y0))
    crwc=np.zeros((n_levs,y1-y0))
    cswc=np.zeros((n_levs,y1-y0))
    
    lonx=lon[y0:y1,x0][::-1]
    latx=lat[y0:y1,x0][::-1]
    maskx=mask[y0:y1,x0][::-1]
    # plt.plot(elevx)
    
    n_hours=1
    for hh in range(n_hours):
        if hh==0: # 15 UTC
        # if hh==6: # 18 UTC
            # print(hh,time[hh])
        # if hh==6: # 18 UTC
        # if hh>=0:
            for lev_index,lev in enumerate(levs):
                temp=np.rot90(Tx[lev_index,:,:].T)
                temp_pseudo=np.rot90(paptx[lev_index,:,:].T)
                # T[lev_index,:]=temp[y0:y1,x0]
                tempx=np.rot90(Wx[lev_index,:,:].T).copy()
                tempU=np.rot90(Ux[lev_index,:,:].T).copy()
                tempV=np.rot90(Vx[lev_index,:,:].T).copy()
                temprh=np.rot90(rhx[lev_index,:,:].T).copy()
                tempclwc=np.rot90(clwcx[lev_index,:,:].T).copy()*1000
                tempciwc=np.rot90(ciwcx[lev_index,:,:].T).copy()*1000
                tempcrwc=np.rot90(crwcx[lev_index,:,:].T).copy()*1000
                tempcswc=np.rot90(cswcx[lev_index,:,:].T).copy()*1000
                np.shape(tempclwc)
                # if do_width:
                #     # theta[lev_index,:]=np.mean(temp[y0:y1,x0-width:x0+width],axis=1)*(p0/lev)**0.286
                #     theta[lev_index,:]=np.mean(temp[y0:y1,x0-width:x0+width],axis=1)*(p0/lev)**0.286
                #     elevx=np.mean(elev[y0:y1,x0-width:x0+width],axis=1)/1000               
                #     W[lev_index,:]=np.mean(tempx[y0:y1,x0-width:x0+width],axis=1)
                # else:
                theta[lev_index,:]=temp[y0:y1,x0][::-1]*(p0/lev)**0.286
                theta_pseudo[lev_index,:]=temp_pseudo[y0:y1,x0][::-1]#*(p0/lev)**0.286
                elevx=elev[y0:y1,x0][::-1]/1000               
                W[lev_index,:]=tempx[y0:y1,x0][::-1]
                U[lev_index,:]=tempU[y0:y1,x0][::-1]
                V[lev_index,:]=tempV[y0:y1,x0][::-1]
                rh[lev_index,:]=temprh[y0:y1,x0][::-1]
                clwc[lev_index,:]=tempclwc[y0:y1,x0][::-1]
                ciwc[lev_index,:]=tempciwc[y0:y1,x0][::-1]
                crwc[lev_index,:]=tempcrwc[y0:y1,x0][::-1]
                cswc[lev_index,:]=tempcswc[y0:y1,x0][::-1]

            # plt.imshow(theta)
            # plt.colorbar()
            x = np.linspace(0,1,np.shape(theta)[1])*2.5*np.shape(theta)[1]
            # y = np.linspace(0,1,np.shape(theta)[0])
            # y = np.linspace(0,1,np.shape(theta)[0])
            
            # international barometric formula
            y = 44330 * (1 - (levs/p0)**(1/5.255)) /1000
            
            # print(,levs[levx])

            # plt.plot(y,levs)
            
            # H = altitude (m)
            # P = measured pressure (Pa) from the sensor
            # p0 = reference pressure at sea level (e.g. 1013.25hPa)
            
            X, Y = np.meshgrid(x, y)
            # np.shape(X)
            
            plt.close()
            plt.clf()
            if plt_map==0:
                fig, ax = plt.subplots(figsize=(10,10))
            else:
                # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16,14),gridspec_kw={'width_ratios': [0.8, 0.4]})
                fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20,14),gridspec_kw={'width_ratios': [0.8, 0.4]})
            isentropes_name=''
            if do_isentropes:
                T0=280 ; T1=340
                levs_theta=np.arange(T0,T1,2)
                levs_theta2=np.arange(T0,T1,4)
                CS=ax.contour(X,Y,theta,levels=levs_theta,linewidths=th,colors='k') #darkgrey
                ax.clabel(CS, levs_theta2, inline=True, fmt=fmt, fontsize=14)
                isentropes_name='_isentropes'

            pseudo_name=''
            if do_pseudo:
                T0_psuedo=280 ; T1_psuedo=310
                levs_theta_pseudo=np.arange(T0_psuedo,T1_psuedo,1)    
                levs_theta_pseudo=[287,289,291,291.5,292,292.5,293.5]
                levs_theta_pseudo=[280,281,282,283,284,285,286,287,288,289,290,291,291.5,292,293,294]
                CS_pseudo=ax.contour(X,Y,theta_pseudo,levels=levs_theta_pseudo,linewidths=th*3,colors='#565656') #darkgrey
                ax.clabel(CS_pseudo, CS_pseudo.levels, inline=True, fmt=fmt, fontsize=14)
                pseudo_name='_pseudo'

            # ax.set_title('CARRA vertical velocity & potential temperature')

            print("kilometers from %.3f" %latx[0]+"°N, %.3f" %abs(lonx[0])+"°W to %.3f" %latx[-1]+"°N, %.3f" %abs(lonx[-1])+"°W")
            ax.set_xlabel("kilometers from %.2f" %latx[0]+"°N, %.2f" %abs(lonx[0])+"°W to %.2f" %latx[-1]+"°N, %.2f" %abs(lonx[-1])+"°W")
            ax.spines['bottom'].set_color('red')
            ax.spines['bottom'].set_color('m')
            ax.xaxis.label.set_color('m')
            ax.tick_params(axis='x', colors='m')
            
            ax.plot(x,elevx,c='k')
            v=np.where(maskx>=0.5)
            v=v[0]
            ax.fill_between(x[v],elevx[v],0, color='#60A7D2',alpha=1,interpolate=True,zorder=300,label='stdev. smoothed')
            v=np.where(maskx<=0.5)
            v=v[0]
            ax.fill_between(x[v],elevx[v],0, color='#795548',alpha=1,interpolate=True,zorder=300,label='stdev. smoothed')

            # ------------------------- y axis
            ax.set_ylabel('height, km')
            
            ax3 = ax.twinx()
            
            # ax3.plot(meta.elev[v_elev[0]],t2mx[v_elev[0]],'o-',c='r',label='2 m air temperature')
            # ax3.set_ylabel('deg. C', color='r')
            ax3.set_ylim(1000,200)
            # ax3.set_xticklabels(levs.astype(int).astype(str))
            ax3.set_yticks(levs[0:n_levs-1],labels=levs[0:n_levs-1].astype(int).astype(str))

            # ax3.tick_params(axis='y', colors='r')
            
            max_height=5
            max_height=25
            # max_height=np.max(y)
            max_height=y[n_levs-1]
            max_height=10.279 # 250 hPa
            max_height=11.2 # 250 hPa
            max_height=8 # 250 hPa
            ax.set_ylim(0,max_height)

            if var_choice=='clwc':
                units='sp. cld. liq. w.c. g kg$\mathregular{^{-1}}$'
                cm=plt.cm.viridis
                cm.set_over('orange')
                # cm.set_under('purple')
                lo=0 ; hi=0.8 ; dx=0.05
                levs_W=np.arange(lo,hi+dx,dx)
                cntr=ax.contourf(X,Y,clwc,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both') 

            if var_choice=='ciwc':
                units='sp. cld. ice w.c. g kg$\mathregular{^{-1}}$'
                cm=plt.cm.viridis
                cm.set_over('orange')
                # cm.set_under('purple')
                lo=0 ; hi=0.03 ; dx=0.01
                levs_W=np.arange(lo,hi+dx,dx)
                cntr=ax.contourf(X,Y,ciwc,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='max') 

            if var_choice=='crwc':
                units='sp. cld. rain w.c. g kg$\mathregular{^{-1}}$'
                cm=plt.cm.viridis
                cm.set_over('orange')
                cm.set_under('purple')
                lo=0 ; hi=0.03 ; dx=0.01
                lo=0 ; hi=0.8 ; dx=0.05
                levs_W=np.arange(lo,hi+dx,dx)
                cntr=ax.contourf(X,Y,crwc,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both') 

            if var_choice=='cswc':
                units='sp. cld. snow w.c. g kg$\mathregular{^{-1}}$'
                cm=plt.cm.viridis
                cm.set_over('orange')
                cm.set_under('purple')
                lo=0 ; hi=0.8 ; dx=0.05
                # lo=0 ; hi=0.8 ; dx=0.05

                levs_W=np.arange(lo,hi+dx,dx)
                cntr=ax.contourf(X,Y,cswc,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both') 
                
            if var_choice=='w':
                units='vertical winds, m s$\mathregular{^{-1}}$'
                cm=plt.cm.bwr
                cm.set_over('orange')
                cm.set_under('purple')
                lo=-1.5 ; hi=1.5 ; dx=0.25
                levs_W=np.arange(lo,hi+dx,dx)
                # print(levs_W)
                # levs_W=[-1.5,1.5]
                # mine=plt.contourf(X,Y,W,levels=levs_W)
                # linestyles=['dashed','dashed','dashed','dotted','solid','solid','solid']
                cntr=ax.contourf(X,Y,W,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both')

            if var_choice=='speed':
                units='winds, m s$\mathregular{^{-1}}$'
                cm=plt.cm.cividis
                cm.set_over('orange')
                # cm.set_under('purple')
                lo=0 ; hi=36 ; dx=4
               
                levs_W=np.arange(lo,hi+dx,dx)
                speed = np.sqrt(U**2 + V**2)

                cntr=ax.contourf(X,Y,speed,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='max')

            if var_choice=='V':
                units='meridional winds, m s$\mathregular{^{-1}}$'
                cm=plt.cm.bwr
                cm.set_over('orange')
                cm.set_under('purple')
                lo=-36 ; hi=36 ; dx=4
               
                levs_W=np.arange(lo,hi+dx,dx)

                cntr=ax.contourf(X,Y,V,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='max')

            if var_choice=='U':
                units='zonal winds, m s$\mathregular{^{-1}}$'
                cm=plt.cm.bwr
                cm.set_over('orange')
                cm.set_under('purple')
                lo=-22 ; hi=22 ; dx=4
               
                levs_W=np.arange(lo,hi+dx,dx)

                cntr=ax.contourf(X,Y,U,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both')

            if var_choice=='rh':
                units='relative humidity, %'
                cm=plt.cm.viridis
                cm.set_over('orange')
                cm.set_under('k')
                lo=78 ; hi=100 ; dx=2
               
                levs_W=np.arange(lo,hi+dx,dx)

                cntr=ax.contourf(X,Y,rh,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both')

            annotatex=1
            
            if annotatex:
                props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')

                # xx0=0.02 ; yy0=0.98
                # mult=1.
                # ax.text(xx0, yy0, time[hh].strftime('- Θ'),
                #         fontsize=font_size*mult,color='k',bbox=props,rotation=0,
                #         va='top',ha='left',transform=ax.transAxes,zorder=20)

                xx0=1.09 ; yy0=0.55
                mult=1.
                ax.text(xx0, yy0,'pressure, hPa',
                        fontsize=font_size*mult,color='k',rotation=90,
                        va='top',ha='left',transform=ax.transAxes,zorder=20)
                
                xx0=1.15 ; yy0=1
                mult=1.
                ax.text(xx0, yy0, time.strftime('%Y %b %d %HUTC'),
                        fontsize=font_size*mult,color='k',rotation=0,
                        va='top',ha='left',transform=ax.transAxes,zorder=20)

                # if ((x0>=252) & (x0<=281)):
                #     # props = dict(boxstyle='round', facecolor='grey', alpha=1,edgecolor='w')
                #     xx0=0.02 ; yy0=0.02
                #     mult=1.
                #     ax.text(xx0, yy0, 'Greenland terrain',
                #             fontsize=font_size*mult,color='k',rotation=0,
                #             va='bottom',ha='left',transform=ax.transAxes,zorder=20)

                # if ((x0>=282) & (x0<=379)):
                #     xx0=0.09 ; yy0=0.09
                #     mult=1.
                #     ax.text(xx0, yy0, 'Greenland ice sheet',
                #             fontsize=font_size*mult,color='k',rotation=0,
                #             va='bottom',ha='left',transform=ax.transAxes,zorder=301)


                xx0=1.15 ; yy0=-0.02
                mult=1.
                ax.text(xx0, yy0, units,
                        fontsize=font_size*mult,color='k',rotation=0,
                        va='bottom',ha='left',transform=ax.transAxes,zorder=20)
            
            du_color_bar=1
            
            with_colorbarname=''
            if du_color_bar:
                with_colorbarname='_w_colorbar'
                cbax = ax.inset_axes([1.03, -0.05, 0.55, 0.025], transform=ax.transAxes)
                # cbax.set_title('vertical\nwinds,\nm/s',fontsize=font_size,c='k',ha='center')
                fig.colorbar(cntr, ax=ax, cax=cbax, shrink=0.7,orientation='horizontal')
                    
            if plt_map:
                # -----------------------------------------------------------------------------  
                fn='/Users/jason/Dropbox/AWS/GEUS_AWS_NRT_alerts/ancil/Greenland_Coastline/GRL_adm0.shp'
                # fn='/Users/jason/Dropbox/Greenland map/coastline/Greenland_coast/Greenland_coast.shp'
                coastline = gpd.read_file(fn)
                coastline.crs = {'init' :'epsg:4326'}
                coastline = coastline.to_crs({'init': 'EPSG:3413'})
            
                coastline.plot(facecolor='none', edgecolor='k',linewidth=0.5,ax=ax2,figure=fig)
                ylims=ax2.get_ylim()
                xlims=ax2.get_xlim()
                
                fn='/Users/jason/Dropbox/AWS/GEUS_AWS_NRT_alerts/ancil/500 m contours/500 m contours.shp'
                contours = gpd.read_file(fn)
                contours.plot(facecolor='none', edgecolor='grey',linewidth=0.7,ax=ax2,figure=fig)

                x0x=-6.6e5 ; y0x=-3.4e6
                x1x=8.7e5 ; y1x=1221304.695
                
                ax2.set_xlim([x0x,x1x])
                # ax2.set_ylim([y0,y1])
                ax2.axis('off')
                inProj = Proj(init='epsg:4326')
                outProj = Proj(init='epsg:3413')
                x1xx, y1xx = lon, lat
                lon0_3413, lat0_3413 = transform(inProj,outProj,lonx[0],latx[0])
                lon1_3413, lat1_3413 = transform(inProj,outProj,lonx[-1],latx[-1])
                ax2.plot([lon0_3413,lon1_3413], [lat0_3413,lat1_3413],'-',c='m',linewidth=th*4,zorder=30)
                # if ((x0>=0) & (x0<=361)):
                #     ax2.text(lon1_3413, lat1_3413,' vertical profile',c='m',zorder=30,ha='left',va='bottom')
                # ax2.plot(lon_3413, lat_3413,'*',c=anom_color,markersize=25,zorder=30)
                mult=0.9
                # txt=meta.ASSET[i]+"\n%.3f" % lat+"°N %.3f" % -lon+"°W\n"+" %.0f" % elev+" m elevation"
                # plt.text(lon_3413+1, lat_3413,txt,color='w', fontsize=fs*mult*1.2,verticalalignment='center')
                # plt.text(lon_3413+1, lat_3413,txt,color='b', fontsize=fs*mult,verticalalignment='center')
                plt.subplots_adjust(wspace=0.11)
            
                mult=1

            
            if ly == 'x':
                plt.show() 
            
            DPI=150
                        
            if ly == 'p':
                if by_date:
                    figpath='/Users/jason/0_dat/CARRA/Figs_w_vs_elev/'+time.strftime('%Y%m%d%H')+'/'
                else:
                    figpath='/Users/jason/0_dat/CARRA/Figs_w_vs_elev/'#+str(x0).zfill(3)+'/'
                    # figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/N-S_profiles/'
                os.system('mkdir -p '+figpath)
                figname=figpath+str(x0).zfill(3)+'_'+time.strftime('%Y%m%d%H')+'_'+var_choice+pseudo_name+isentropes_name
                plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                # plt.savefig(figname+'.svg', bbox_inches='tight')