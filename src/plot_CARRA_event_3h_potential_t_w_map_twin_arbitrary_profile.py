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
# from datetime import datetime
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

plt.imshow(mask)

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

fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_t_1000-300hPa_14_levs.grib'

var='w'

def get_UVW(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    # print(ds.variables)
    P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u'].values
    V=ds.variables['v'].values
    W=ds.variables['wz'].values
    # T=ds.variables['t'].values-273.15
    time=ds.variables['time'].values
    # P=P[0]
    return U,V,W,P,time


def get_TP(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    
    # print(ds.variables)
    P=np.array(ds.variables['isobaricInhPa'].values)
    T=ds.variables['t'].values#-273.15
    # W=ds.variables['wz'].values
    time=ds.variables['time'].values
    # P=P[0]
    return T,P,time

Tx,P,times=get_TP(fn)
time=pd.to_datetime(times)
levs=np.array(P)
print(levs)

print(np.shape(Tx))
fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvw_1000-300hPa_14_levs.grib'
Ux,Vx,Wx,Px,timesx=get_UVW(fn)

#%%
def get_rh(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    # print(ds.variables)
    # list(ds.keys())
    P=np.array(ds.variables['isobaricInhPa'].values)
    rh=ds.variables['r'].values
    # T=ds.variables['t'].values-273.15
    time=ds.variables['time'].values
    # P=P[0]
    return rh,time

fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_rh_1000-300hPa_14_levs.grib'
rhx,timesx=get_rh(fn)

#%% plot

plt_map=1 # if set to 1 draws a map to the right of the graphic
ly='p'
by_date=0

# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['axes.grid'] = False
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.alpha'] = 0.5
# plt.rcParams['grid.color'] = "k"

x0=302
xs=np.arange(214,380) # range of x values on CARRA grid
xs=np.arange(253,283)
xs=np.arange(280,323) # range of x values on CARRA grid for Qagssimiut
# xs=np.arange(283,450)

xs=np.arange(313,314) # figure candidate
# xs=np.arange(300,313) # figure candidate
xs=np.arange(300,301) # figure candidate
xs=np.arange(310,311) # figure candidate
# xs=np.arange(360,367) # choice

def find_nearest(array, array2, value, value2):
    Y = np.abs(array2 - value2)
    X = np.abs(array - value)
    diff = X+Y
    idy, idx=np.where(diff==np.min(diff)) #location of val
    return idy, idx #array.flat[idx]

point_south=[61.24979325, -50.16499025] # lat lon
point_north=[60.77824406, -41.01194347] # lat lon

point_north=[63.11611121, -47.97473123] # lat lon 
point_south=[59.79145265, -46.71302681] # lat lon

point_north=[63., -46.40008102] # lat lon 
point_south=[59., -45.825000] # lat lon

point_north=[65.03701995, -53.46092367] # lat lon 
point_south=[64.41483728, -38.18607612] # lat lon


point_south=point_south[::-1] ; point_north=point_north[::-1] # flip so lon lat


dist_deg=0
point_south=[point_south[0]+(dist_deg), point_south[1]+(dist_deg)]
point_north=[point_north[0]+(dist_deg), point_north[1]+(dist_deg)]

# #run loop
# for i in range(1,scans+1):
#     point_south=[point_south[0]-(dist_deg), point_south[1]-(dist_deg)]
#     point_north=[point_north[0]-(dist_deg), point_north[1]-(dist_deg)]

# find location
y0, x0=find_nearest(lon, lat, point_south[0], point_south[1])
y1, x1=find_nearest(lon, lat, point_north[0], point_north[1])

# print(y0, x0)
# print(y1, x1)

# x0, y0 = 115, 575 # These are in _pixel_ coordinates!!
# x1, y1 = 1000, 750
num = int(np.round(np.sqrt(np.abs(x0-x1)**2 + np.abs(y0-y1)**2))) *2 #multiplying by 2 shows the step
xx, yy = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

n_x=len(xx)

x = xx[:, 0].astype(int)
y = yy[:, 0].astype(int)
# print(x)

# print(n_x)
# print(n_y)
# print(y)

#in meters
grid_length=2500 #m
x_m = np.abs(x0-x1)*grid_length
y_m = np.abs(y0-y1)*grid_length
len_m=np.sqrt(x_m**2+y_m**2)/1000 #km
profile_axis=np.linspace(0,len_m, num)


p0=1013.25
p0=1000

var_choice='w'
# var_choice='speed'
# var_choice='V'
# var_choice='rh'

    
n_levs=11 # until 600 hPa
n_levs=12 # until 500 hPa
n_levs=14 # until 300 hPa

levs=np.array(P[0:n_levs])
# print(levs)

theta=np.zeros((n_levs,n_x))
T=np.zeros((n_levs,n_x))
W=np.zeros((n_levs,n_x))
U=np.zeros((n_levs,n_x))
V=np.zeros((n_levs,n_x))
rh=np.zeros((n_levs,n_x))

lonx=lon[y,x]
latx=lat[y,x]
maskx=mask[y,x]
# plt.plot(elevx)

n_hours=8

for hh in range(n_hours):
    # if hh>=0: 
    if hh==5: # 15 UTC
    # if hh==6: # 18 UTC
        print(hh,time[hh])
    # if hh==6: # 18 UTC
    # if hh>=0:
        for lev_index,lev in enumerate(levs):
            temp=np.rot90(Tx[hh,lev_index,:,:].T)
            tempx=np.rot90(Wx[hh,lev_index,:,:].T).copy()
            tempU=np.rot90(Ux[hh,lev_index,:,:].T).copy()
            tempV=np.rot90(Vx[hh,lev_index,:,:].T).copy()
            temprh=np.rot90(rhx[hh,lev_index,:,:].T).copy()

            theta[lev_index,:]=temp[y,x]*(p0/lev)**0.286
            elevx=elev[y,x]/1000               
            W[lev_index,:]=tempx[y,x]
            U[lev_index,:]=tempU[y,x]
            V[lev_index,:]=tempV[y,x]
            # rh[lev_index,:]=temprh[y,x]

        x = np.linspace(0,1,np.shape(theta)[1])*2.5*np.shape(theta)[1]
        # y = np.linspace(0,1,np.shape(theta)[0])
        # y = np.linspace(0,1,np.shape(theta)[0])
        
        # international barometric formula
        y = 44330 * (1 - (levs/p0)**(1/5.255)) /1000
        
        # plt.plot(y,levs)
        
        # H = altitude (m)
        # P = measured pressure (Pa) from the sensor
        # p0 = reference pressure at sea level (e.g. 1013.25hPa)
        
        X, Y = np.meshgrid(x, y)
        # X=X-np.min(X)
        # Y=Y-np.min(Y)
        np.shape(X)
        
        plt.close()
        plt.clf()
        if plt_map==0:
            fig, ax = plt.subplots(figsize=(10,10))
        else:
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16,10),gridspec_kw={'width_ratios': [0.8, 0.4]})

        levs_theta=np.arange(280,360,1)
        # mine=plt.contourf(X,Y,theta,levels=levs_theta)
        # c0=0.4
        # (c0,c0,c0)
        
        # contour line labeling
        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} \K" if plt.rcParams["text.usetex"] else f"{s}K"

        CS=ax.contour(X,Y,theta,levels=levs_theta,linewidths=th,colors='k') #darkgrey
        # ax.legend(loc=0)
        # plt.clabel(CS, inline=1, fontsize=10)
        # ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
        ax.clabel(CS, np.arange(280,360,10), inline=True, fmt=fmt, fontsize=14)

        # ax.set_title('CARRA vertical velocity & potential temperature')
        ax.set_ylabel('height, km')
        
        ax3 = ax.twinx()
        
        # ax3.plot(meta.elev[v_elev[0]],t2mx[v_elev[0]],'o-',c='r',label='2 m air temperature')
        # ax3.set_ylabel('deg. C', color='r')
        ax3.set_ylim(1000,300)
        # ax3.set_xticklabels(levs.astype(int).astype(str))
        ax3.set_yticks(levs,labels=levs.astype(int).astype(str))

        # ax3.tick_params(axis='y', colors='r')

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

        max_height=5
        max_height=9.
        # max_height=np.max(y)
        ax.set_ylim(0,max_height)
        
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
            units='vertical winds, m s$\mathregular{^{-1}}$'
            cm=plt.cm.cividis
            cm.set_over('orange')
            # cm.set_under('purple')
            lo=0 ; hi=36 ; dx=4
           
            levs_W=np.arange(lo,hi+dx,dx)
            speed = np.sqrt(U**2 + V**2)

            cntr=ax.contourf(X,Y,speed,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='max')

        if var_choice=='V':
            units='vertical winds, m s$\mathregular{^{-1}}$'
            cm=plt.cm.bwr
            cm.set_over('orange')
            cm.set_under('purple')
            lo=-30 ; hi=30 ; dx=4
           
            levs_W=np.arange(lo,hi+dx,dx)

            cntr=ax.contourf(X,Y,V,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='max')

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

            xx0=1.075 ; yy0=0.55
            mult=1.
            ax.text(xx0, yy0,'pressure, hPa',
                    fontsize=font_size*mult,color='k',rotation=90,
                    va='top',ha='left',transform=ax.transAxes,zorder=20)
            
            xx0=1.15 ; yy0=1
            mult=1.
            ax.text(xx0, yy0, time[hh].strftime('%Y %b %d %HUTC'),
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
            cbax = ax.inset_axes([1.01, -0.05, 0.5, 0.025], transform=ax.transAxes)
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
            plt.subplots_adjust(wspace=0.06)
        
            mult=1

        
        if ly == 'x':
            plt.show() 
        
        DPI=200
        DPI=300
        import os
        if ly == 'p':
            if by_date:
                figpath='/Users/jason/0_dat/CARRA/Figs_w_vs_elev/'+time[hh].strftime('%Y%m%d%H')+'/'
            else:
                figpath='/Users/jason/0_dat/CARRA/Figs_w_vs_elev/'+str(x0).zfill(3)+'/'
                figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/N-S_profiles/'
                figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/arbitrary_profiles/'
            os.system('mkdir -p '+figpath)
            # figname=figpath+str(x0).zfill(3)+'_'+time[hh].strftime('%Y%m%d%H')
            figname=figpath+"%.3f" %latx[0]+"°N, %.3f" %abs(lonx[0])+"°W to %.3f" %latx[-1]+"°N, %.3f" %abs(lonx[-1])+"°W"+'_'+time[hh].strftime('%Y%m%d%H')
            plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')