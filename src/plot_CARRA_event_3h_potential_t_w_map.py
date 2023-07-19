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

#%% plot

plt_map=1 # if set to 1 draws a map to the right of the graphic

x0=302
xs=np.arange(214,380) # range of x values on CARRA grid
xs=np.arange(300,301)
# xs=np.arange(284,285)
# xs=np.arange(279,322) # range of x values on CARRA grid for Qagssimiut

p0=1013.25
p0=1000

for x0 in xs:
    print(x0)
    y0=950 ; y1=ni-40
    
    n_levs=11 # until 600 hPa
    n_levs=12 # until 500 hPa
    n_levs=14 # until 300 hPa
    
    levs=np.array(P[0:n_levs])
    # print(levs)
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
    
            x = np.linspace(0,1,np.shape(theta)[1])*2.5*np.shape(theta)[1]
            # y = np.linspace(0,1,np.shape(theta)[0])
            # y = np.linspace(0,1,np.shape(theta)[0])
            
            # international barometric formula
            y = 44330 * (1 - (levs/p0)**(1/5.255)) 
            
            # plt.plot(y,levs)
            
            # H = altitude (m)
            # P = measured pressure (Pa) from the sensor
            # p0 = reference pressure at sea level (e.g. 1013.25hPa)
            
            X, Y = np.meshgrid(x, y)
            np.shape(X)
            

            plt.close()
            plt.clf()
            if plt_map==0:
                fig, ax = plt.subplots(figsize=(10,10))
            else:
                fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16,10),gridspec_kw={'width_ratios': [0.8, 0.4]})

            levs_theta=np.arange(165,290,1)
            # mine=plt.contourf(X,Y,theta,levels=levs_theta)
            # c0=0.4
            # (c0,c0,c0)
            
            # contour line labeling
            def fmt(x):
                s = f"{x:.1f}"
                if s.endswith("0"):
                    s = f"{x:.0f}"
                return rf"{s} \K" if plt.rcParams["text.usetex"] else f"{s}K"

            CS=ax.contour(X,Y,theta,levels=levs_theta,linewidths=th/2,colors='k') #darkgrey
            # ax.legend(loc=0)
            # plt.clabel(CS, inline=1, fontsize=10)
            # ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)

            # ax.set_title('CARRA vertical velocity & potential temperature ')
            ax.set_ylabel('height, m')
            
            ax3 = ax.twinx()
            ax3.set_ylabel('pressure, hPa')
            
            # ax3.plot(meta.elev[v_elev[0]],t2mx[v_elev[0]],'o-',c='r',label='2 m air temperature')
            # ax3.set_ylabel('deg. C', color='r')
            ax3.set_ylim(1000,300)
            # ax3.set_xticklabels(levs.astype(int).astype(str))
            ax3.set_yticks(levs,labels=['1000', '950', '925', '900', '875', '850', '825', '800', '750',
                   '700', '600', '500', '400', '300'])

            # ax3.tick_params(axis='y', colors='r')

            ax.set_xlabel("kilometers from %.2f" %latx[0]+"°N to %.2f" %abs(lonx[0])+"°W to %.2f" %latx[-1]+"°N to %.2f" %abs(lonx[-1])+"°W")
            ax.spines['bottom'].set_color('red')
            ax.spines['bottom'].set_color('m')
            ax.xaxis.label.set_color('m')
            ax.tick_params(axis='x', colors='m')
            
            ax.plot(x,elevx,c='k')
            ax.fill_between(x,elevx,0, color='grey',alpha=1,interpolate=True,zorder=10,label='stdev. smoothed')
            max_height=5000
            max_height=8000
            max_height=np.max(y)
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
                cntr=ax.contourf(X,Y,W,cmap=cm,levels=levs_W,vmin=lo,vmax=hi,extend='both')
            
            annotatex=1
            
            if annotatex:
                props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')

                # xx0=0.02 ; yy0=0.98
                # mult=1.
                # ax.text(xx0, yy0, time[hh].strftime('- Θ'),
                #         fontsize=font_size*mult,color='k',bbox=props,rotation=0,
                #         va='top',ha='left',transform=ax.transAxes,zorder=20)
                
                xx0=1.03 ; yy0=1
                mult=1.
                ax.text(xx0, yy0, time[hh].strftime('%Y %b %d %HUTC'),
                        fontsize=font_size*mult,color='k',rotation=0,
                        va='top',ha='left',transform=ax.transAxes,zorder=20)

                if ((x0>=231) & (x0<=283)):
                    props = dict(boxstyle='round', facecolor='grey', alpha=1,edgecolor='w')
                    xx0=1.01 ; yy0=-0.06
                    mult=1.
                    ax.text(xx0, yy0, 'Greenland terrain',
                            fontsize=font_size*mult,color='w',bbox=props,rotation=0,
                            va='bottom',ha='left',transform=ax.transAxes,zorder=20)

                if x0>=284:
                    xx0=0.12 ; yy0=0.09
                    mult=1.
                    ax.text(xx0, yy0, 'Greenland terrain',
                            fontsize=font_size*mult,color='k',rotation=0,
                            va='bottom',ha='left',transform=ax.transAxes,zorder=20)


                xx0=1.07 ; yy0=0.51
                mult=1.
                ax.text(xx0, yy0, 'vertical\nwinds,\n m s$\mathregular{^{-1}}$',
                        fontsize=font_size*mult,color='k',rotation=0,
                        va='bottom',ha='center',transform=ax.transAxes,zorder=20)
            
            du_color_bar=1
            
            with_colorbarname=''
            if du_color_bar:
                with_colorbarname='_w_colorbar'
                width=0.03
                cbax = ax.inset_axes([1.03, 0., width, 0.5], transform=ax.transAxes)
                # cbax.set_title('vertical\nwinds,\nm/s',fontsize=font_size,c='k',ha='center')
                fig.colorbar(cntr, ax=ax, cax=cbax, shrink=0.7,orientation='vertical')
        
            ly='x'
            
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
                ax2.text(lon1_3413, lat1_3413,' transect',c='m',zorder=30,ha='left',va='top')
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
            
            if ly == 'p':
                figpath='/Users/jason/0_dat/CARRA/Figs_w_vs_elev/'+time[hh].strftime('%Y%m%d%H')+'/'
                os.system('mkdir -p '+figpath)
                figname=figpath+str(x0).zfill(3)+'_'+time[hh].strftime('%Y%m%d%H')
                plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')