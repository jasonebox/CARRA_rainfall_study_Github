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

path='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/'
os.chdir(path)

ni=1269 ; nj=1069

fn='./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
elev=np.fromfile(fn, dtype=np.float32)
elev=elev.reshape(ni, nj)
elev=np.rot90(elev.T)

# plt.imshow(elev)

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

do_plot=1
# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['axes.grid'] = False


inpath='/Users/jason/0_dat/CARRA/'
# inpath='/Volumes/LaCie/0_dat/CARRA/'
outpath='/Users/jason/0_dat/CARRA/output/'

# ------------------------------------------- rain fraction
rainos=0.
x0=0.5 ; x1=2.5
x0-=rainos
x1-=rainos
y0=0 ; y1=1
a1=(y1-y0)/(x1-x0)
a0=y0-a1*x0

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
x0=302
xs=np.arange(180,330)
# xs=np.arange(300,301)

p0=1013.25
p0=1000

for x0 in xs:
    print(x0)
    y0=950 ; y1=ni-40
    
    n_levs=11 # until 600 hPa
    n_levs=12 # until 500 hPa
    n_levs=14 # until 300 hPa
    
    levs=np.array(P[0:n_levs])
    print(levs)
    theta=np.zeros((n_levs,y1-y0))
    T=np.zeros((n_levs,y1-y0))
    W=np.zeros((n_levs,y1-y0))
    
    elevx=elev[y0:y1,x0]
    lonx=lon[y0:y1,x0]
    latx=lat[y0:y1,x0]
    # plt.plot(elevx)
    
    n_hours=8
    for hh in range(n_hours):
        if hh==5: # 15 UTC
        # if hh==6: # 18 UTC
            print(hh,time[hh])
        # if hh==6: # 18 UTC
        # if hh>=0:
            for lev_index,lev in enumerate(levs):
                temp=np.rot90(Tx[hh,lev_index,:,:].T).copy()
                T[lev_index,:]=temp[y0:y1,x0]
                theta[lev_index,:]=temp[y0:y1,x0]*(lev/p0)**0.286
                # theta[lev_index,10:20]=280
                tempx=np.rot90(Wx[hh,lev_index,:,:].T).copy()
                W[lev_index,:]=tempx[y0:y1,x0]
                
    
                if lev_index==0:
                    plt.imshow(temp)
                    plt.colorbar()
    
            x = np.linspace(0,1,np.shape(theta)[1])*2.5*np.shape(theta)[1]
            # y = np.linspace(0,1,np.shape(theta)[0])
            # y = np.linspace(0,1,np.shape(theta)[0])
            y = 44330 * (1 - (levs/p0)**(1/5.255)) # international barometric formula: 
            # H = altitude (m)
            # P = measured pressure (Pa) from the sensor
            # p0 = reference pressure at sea level (e.g. 1013.25hPa)
            
            X, Y = np.meshgrid(x, y)
            np.shape(X)
            
            plt.close()
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,10))
            # plt.imshow(theta)
            # plt.colorbar()
            levs_theta=np.arange(165,290,4)
            # mine=plt.contourf(X,Y,theta,levels=levs_theta)
            # c0=0.4
            # (c0,c0,c0)
            plt.contour(X,Y,theta,levels=levs_theta,linewidths=th/2,colors='k') #darkgrey
            
            ax.set_title('CARRA vertical velocity & potential temperature ')
            ax.set_ylabel('height, m')
            ax.set_xlabel("distance, km from %.2f" %latx[0]+"°N to %.2f" %latx[-1]+'°N\n'+"longitude line %.2f" %lonx[0]+"°W to %.2f" %lonx[-1]+'°W')
            ax.plot(x,elevx,c='k')
            ax.fill_between(x,elevx,0, color='grey',alpha=1,interpolate=True,zorder=10,label='stdev. smoothed')
            max_height=5000
            max_height=8000
            ax.set_ylim(0,max_height)
            
            cm=plt.cm.bwr
            cm.set_over('orange')
            cm.set_under('purple')
            plt_W=1
            if plt_W:
                lo=-1.5 ; hi=1.5 ; dx=0.25
                levs_W=np.arange(lo,hi+dx,dx)
                # print(levs_W)
                # levs_W=[-1.5,1.5]
                # mine=plt.contourf(X,Y,W,levels=levs_W)
                # linestyles=['dashed','dashed','dashed','dotted','solid','solid','solid']
                cntr=plt.contourf(X,Y,W,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both')
                # cntr.monochrome = True
                # for col, ls in zip(cntr.collections, cntr._process_linestyles()):
                #     col.set_linestyle(ls)
                
                # Override the linestyles based on the levels.
                for line, lvl in zip(cntr.collections, cntr.levels):
                    if lvl < 0:
                        line.set_linestyle('--')
                    elif lvl == 0:
                        line.set_linestyle(':')
                    else:
                        # Optional; this is the default.
                        line.set_linestyle('-')
                        
                # plt.show()
                # for line, lvl in zip(cntr.collections, cntr.levels):
                #     if lvl < 0:
                #         line.set_linestyle('--')
                #     elif lvl == 0:
                #         line.set_linestyle(':')
                #     else:
                #         # Optional; this is the def
                #         line.set_linestyle('-')
            
            # ax.axis('off')
            
            # cbar = fig.colorbar(mine)
            
            
            props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
            
            # xx0=0.5 ; yy0=0.95
            # mult=1.1
            
            # ax.text(xx0, yy0, 'Greenland CARRA rain flux',
            #         fontsize=font_size*mult,color='k',ha='center',
            #         bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
            
            annotatex=1
            
            if annotatex:
                xx0=0.015 ; yy0=0.015
                mult=1.
                ax.text(xx0, yy0, time[hh].strftime('%Y %b %d %HUTC'),
                        fontsize=font_size*mult,color='k',bbox=props,rotation=0,
                        va='bottom',ha='left',transform=ax.transAxes,zorder=20)

                xx0=0.17 ; yy0=0.12
                mult=1.
                ax.text(xx0, yy0, 'Greenland',
                        fontsize=font_size*mult,color='k',rotation=0,
                        va='bottom',ha='left',transform=ax.transAxes,zorder=20)

            
            du_color_bar=1
            
            with_colorbarname=''
            if du_color_bar:
                with_colorbarname='_w_colorbar'
                width=0.03
                cbax = ax.inset_axes([1.01, 0.2, width, 0.5], transform=ax.transAxes)
                cbax.set_title('m/s',fontsize=font_size,c='k',ha='left')
                fig.colorbar(cntr, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')
        
            ly='p'
            
            if ly == 'x':
                plt.show() 
            
            DPI=200
            
            if ly == 'p':
                figpath='/Users/jason/0_dat/CARRA/Figs_w_vs_elev/'+time[hh].strftime('%Y%m%d%H')+'/'
                os.system('mkdir -p '+figpath)
                figname=figpath+str(x0).zfill(3)+'_'+time[hh].strftime('%Y%m%d%H')
                plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')