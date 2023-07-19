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

do_plot=1
# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'


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

fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvwt_3_levs.grib'
fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvw_1000-300hPa_14_levs.grib'

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

Ux,Vx,Wx,P,times=get_UVW(fn)
time=pd.to_datetime(times)
print(P)
levs=np.array(P)

def get_UVT(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    
    # print(ds.variables)
    # P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u'].values
    V=ds.variables['v'].values
    T=ds.variables['t'].values-273.15
    # W=ds.variables['wz'].values
    time=ds.variables['time'].values
    # P=P[0]
    return U,V,T,time

subset_it=0

# fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
# lat=np.fromfile(fn, dtype=np.float32)
# lat=lat.reshape(ni, nj)

# fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
# lon=np.fromfile(fn, dtype=np.float32)
# lon=lon.reshape(ni, nj)

max_value=36



if subset_it:
    elev0=50;elev1=3000;delev=200
    # 50,3000,200
    max_value=24
    max_value=36
    subset_name='_subset_name'
    # lon=np.rot90(lon.T)
    # lat=np.rot90(lat.T)
    xc0=200 ; xc1=450 # 250
    yc0=950 ; yc1=1200 # 250
    xc0=210 ; xc1=400
    yc0=1000 ; yc1=1210

    # lat=lat[yc0:yc1,xc0:xc1]
    # lon=lon[yc0:yc1,xc0:xc1]
    # print(lat.shape)
    # ni=lat.shape[0]
    # nj=lat.shape[1]
    # lon=np.rot90(lon.T)
    # lat=np.rot90(lat.T)
    
    # elev=np.rot90(elev.T)
    # elev=elev[yc0:yc1,xc0:xc1]
    # elev=np.rot90(elev.T)
    # plt.imshow(elev)
    # saas

# mask subset

# fn='./ancil/CARRA_W_domain_ice_mask.nc'
# nc2 = xr.open_dataset(fn)
# # print(nc2.variables)
# mask = nc2.variables['z']

# if subset_it:
#     # if event!='2010-10_02-06':
#     #     U=np.rot90(U.T)
#     #     U=U[yc0:yc1,xc0:xc1]
#     #     U=np.rot90(U.T)
#     #     V=np.rot90(V.T)
#     #     V=V[yc0:yc1,xc0:xc1]
#     #     V=np.rot90(V.T)
    
#     mask=np.rot90(mask.T)
#     mask=mask[yc0:yc1,xc0:xc1]
#     # mask=np.rot90(mask.T)


# plt.close()
# plt.clf()

# fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
# plt.imshow(mask)
# ax.axis('off')

# ly='p'

# if ly == 'x':
#     plt.show() 

# DPI=200

# if ly == 'p':
#     figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/subset/'
#     # os.system('mkdir -p '+figpath)
#     figname=figpath+'mask'+subset_name
#     plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

#%% do hourly
n_hours=8
hh=2
cc=0

for hh in range(n_hours):
    print(hh)
    # if hh==5: # 15 UTC
    # if hh==6: # 18 UTC
    if hh>=0:
        for lev_index,lev in enumerate(levs):
            # if lev_index==11: #800 hPa
            # if lev_index==3: 
            if lev_index>=0: 
            # if lev==850:
                # print(str(dates[i]))
                print(lev,str(time[hh].strftime('%Y %b %d %H')))
                  
                # var='SHF'
                
                skip_interval=30
            
                # if var=='wind':
                #     if event=='20170913-15':
                #         U = nc.variables['u'][i,0,:,:]
                #         V = nc.variables['v'][i,0,:,:]
                #     else:
                #         U = nc.variables['u10'][i,:,:]
                #         V = nc.variables['v10'][i,:,:]
                #         # nc.variables['v'].shape            
                #     plotvar = np.sqrt(U**2 + V**2)
                #     lo=0; hi=36
                #     dx=(hi-lo)/36
                #     units='m s$^{-1}$'
                #     prefix="CARRA 900 hPa wind speed, "
                #     cmap_choice='jet'
                #     quiver_color='w'
                #     skip_interval=8
            
                if var=='w':
                    lo=-1.5; hi=-lo
                    dx=(hi-lo)/40
                    units='m s$^{-1}$'
                    # cmap_choice='bwr'
                    quiver_color='k'
                    skip_interval=5
                    # U=np.mean(Ux[:,lev_index,:,:],axis=0)
                    # V=np.mean(Vx[:,lev_index,:,:],axis=0)
                    # T=np.mean(Tx[:,lev_index,:,:],axis=0)
                    # U=np.mean(Ux[hh,lev_index,:,:],axis=0)
                    # V=np.mean(Vx[hh,lev_index,:,:],axis=0)
                    # T=Tx[hh,lev_index,:,:]
                    plotvar = Wx[hh,lev_index,:,:]
                    Y, X = np.mgrid[0:ni,0:nj]
                    cm = plt.cm.bwr
            
            
                plotvar=np.array(plotvar)
                
                if subset_it:
                    # if event!='2010-10_02-06':
                    #     U=np.rot90(U.T)
                    #     U=U[yc0:yc1,xc0:xc1]
                    #     U=np.rot90(U.T)
                    #     V=np.rot90(V.T)
                    #     V=V[yc0:yc1,xc0:xc1]
                    #     V=np.rot90(V.T)
                    
                    plotvar=np.rot90(plotvar.T)
                    plotvar=plotvar[yc0:yc1,xc0:xc1]
                    plotvar=np.rot90(plotvar.T)
                
                if subset_it==0:skip_interval=30
                clevs = np.arange(lo,hi+dx,dx)
            
                # mcmap = plt.get_cmap(cmap_choice)
                cm=plt.cm.bwr
                cm.set_over('orange')
                cm.set_under('purple')
                
                plt.close()
                plt.clf()
                
                fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
        
                # plotvar[plotvar>=np.max(clevs)]=np.max(clevs)-0.01
                # plotvar[plotvar<=lo]=lo
            
                # np.shape(plotvar)
                # plt.title(datex.strftime('%Y %b %d %HUTC')+' '+str(int(lev))+' hPa',c='k')
                # print(dates[i])
                
                # inv=np.where(lats<50)
                # plotvar[inv[0]]=np.nan
                # print("min",np.nanmin(plotvar),"max",np.nanmax(plotvar))
            
                # m.drawcoastlines()
                # m.drawparallels(np.arange(-90.,120.,30.))
                # m.drawmeridians(np.arange(0.,420.,50.))
            
                
                mine=plt.contourf(plotvar,clevs,cmap=cm,vmin=lo,vmax=hi,extend='both')
                # clevs=np.arange(elev0,elev1,delev)
            
                plt.clim(lo,hi)
                ax.axis('off')
        
            
                du_color_bar=0
                if du_color_bar:
                    # units=str(int(P))+'hPa\n°C'
                    width=0.025
                    cbax = ax.inset_axes([1.02, 0.005, width, 0.41], transform=ax.transAxes)
                    cbax.set_title(units,fontsize=font_size*0.9,c='k',ha='left')
                    cbax.tick_params(labelsize=font_size*0.9)
                    fig.colorbar(mine, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')
                    
                ly='p'
            
                if ly == 'x':
                    plt.show() 
            
                DPI=200
                
                if ly == 'p':                    
                    if subset_it:
                        subset_name='_subset'
                        figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/subset/'

                        # os.system('mkdir -p '+figpath)
                        figname=figpath+var+'_'+str(int(lev))+'hPa'+'_'+time[hh].strftime('%Y_%m_%d_%H')+subset_name
                        plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                        print(figname)
                    else:
                        subset_name=''
                        figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'+var+'/'
                        # os.system('mkdir -p '+figpath)
                        # # figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'+var+'/'+time[hh].strftime('%Y_%m_%d_%H')+'/'
                        # os.system('mkdir -p '+figpath)
                        # # os.system('mkdir -p '+figpath)
                        # figname=figpath+var+'_'+time[hh].strftime('%Y_%m_%d_%H')+'_'+str(int(lev))+'hPa'+'_'+subset_name
                        # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                        # print(figname)
                        # by hour
                        figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'+var+'/'+time[hh].strftime('%Y_%m_%d_%H')+'/'
                        os.system('mkdir -p '+figpath)
                        # os.system('mkdir -p '+figpath)
                        figname=figpath+var+'_'+time[hh].strftime('%Y_%m_%d_%H')+'_'+str(int(lev))+'hPa'+subset_name
                        plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                        print(figname)
                        # by level
                        figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'+var+'/'+str(int(lev))+'/'
                        os.system('mkdir -p '+figpath)
                        # os.system('mkdir -p '+figpath)
                        figname=figpath+var+'_'+time[hh].strftime('%Y_%m_%d_%H')+'_'+str(int(lev))+'hPa'+subset_name
                        plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                        print(figname)

# #%% do daily
# var='w'
# # var='UV'

# for lev_index,lev in enumerate(levs):
#     # if lev_index==11: #800 hPa
#     # if lev_index==3: 
#     if lev_index>=0: # 900
#     # if lev=='800':
#         # print(str(dates[i]))
#         datex=pd.to_datetime(dates[0])
#         print(datex)
        
#         if var=='w':
#             lo=-1.; hi=-lo
#             dx=(hi-lo)/40
#             units='m s$^{-1}$'
#             # cmap_choice='bwr'
#             quiver_color='k'
#             # U=np.mean(Ux[:,lev_index,:,:],axis=0)
#             # V=np.mean(Vx[:,lev_index,:,:],axis=0)
#             # T=np.mean(Tx[:,lev_index,:,:],axis=0)
#             # U=np.mean(Ux[hh,lev_index,:,:],axis=0)
#             plotvar=np.mean(Wx[:,lev_index,:,:],axis=0)
#             # T=Tx[hh,lev_index,:,:]
#             Y, X = np.mgrid[0:ni,0:nj]
#             # plotvar = Wx[hh,lev_index,:,:]
#             cm = plt.cm.bwr
#             # mcmap = plt.get_cmap(cmap_choice)
#             cm=plt.cm.bwr
#             cm.set_over('orange')
#             cm.set_under('purple')

#         if var=='UV':
#             lo=0.; hi=36
#             dx=(hi-lo)/36
#             units='m s$^{-1}$'
#             # cmap_choice='bwr'
#             # U=np.mean(Ux[:,lev_index,:,:],axis=0)
#             # V=np.mean(Vx[:,lev_index,:,:],axis=0)
#             # T=np.mean(Tx[:,lev_index,:,:],axis=0)
#             # U=np.mean(Ux[hh,lev_index,:,:],axis=0)
#             U=np.mean(Ux[:,lev_index,:,:],axis=0)
#             V=np.mean(Vx[:,lev_index,:,:],axis=0)
#             plotvar = np.sqrt(U**2 + V**2)

#             # T=Tx[hh,lev_index,:,:]
#             Y, X = np.mgrid[0:ni,0:nj]
#             # plotvar = Wx[hh,lev_index,:,:]
#             # cm = plt.cm.viridis
#             cm = plt.cm.jet
#             # mcmap = plt.get_cmap(cmap_choice)
#             cm=plt.cm.jet
#             # cm.set_over('orange')
#             # cm.set_under('purple')
    
#         plotvar=np.array(plotvar)
        
#         if subset_it:
#             # if event!='2010-10_02-06':
#             #     U=np.rot90(U.T)
#             #     U=U[yc0:yc1,xc0:xc1]
#             #     U=np.rot90(U.T)
#             #     V=np.rot90(V.T)
#             #     V=V[yc0:yc1,xc0:xc1]
#             #     V=np.rot90(V.T)
            
#             plotvar=np.rot90(plotvar.T)
#             plotvar=plotvar[yc0:yc1,xc0:xc1]
#             plotvar=np.rot90(plotvar.T)
#             subset_name='_subset'
        
#         if subset_it==0:skip_interval=30
#         clevs = np.arange(lo,hi+dx,dx)
    

        
#         plt.close()
#         plt.clf()
        
#         fig, ax = plt.subplots(figsize=(10, 10*ni/nj))

#         # plotvar[plotvar>=np.max(clevs)]=np.max(clevs)-0.01
#         # plotvar[plotvar<=lo]=lo
    
#         # np.shape(plotvar)
#         # plt.title(datex.strftime('%Y %b %d %HUTC')+' '+str(int(lev))+' hPa',c='k')
#         # print(dates[i])
        
#         # inv=np.where(lats<50)
#         # plotvar[inv[0]]=np.nan
#         # print("min",np.nanmin(plotvar),"max",np.nanmax(plotvar))
    
#         # m.drawcoastlines()
#         # m.drawparallels(np.arange(-90.,120.,30.))
#         # m.drawmeridians(np.arange(0.,420.,50.))
    
        
#         mine=plt.contourf(plotvar,clevs,cmap=cm,vmin=lo,vmax=hi,extend='both')
#         clevs=np.arange(elev0,elev1,delev)
    
#         plt.clim(lo,hi)
#         ax.axis('off')

    
#     du_color_bar=0
#     if du_color_bar:
#         # units=str(int(P))+'hPa\n°C'
#         width=0.025
#         cbax = ax.inset_axes([1.02, 0.005, width, 0.41], transform=ax.transAxes)
#         cbax.set_title(units,fontsize=font_size*0.9,c='k',ha='left')
#         cbax.tick_params(labelsize=font_size*0.9)
#         fig.colorbar(mine, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')
        
#     ly='p'

#     if ly == 'x':
#         plt.show() 

#     DPI=200
    
#     if ly == 'p':
#         figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/subset/'
#         # os.system('mkdir -p '+figpath)
#         figname=figpath+var+'_'+str(int(lev))+'hPa'+'_'+datex.strftime('%Y_%m_%d')+subset_name
#         plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
# #%%
# make_gif=0
# # nam='2017-09-14_speed_orthographic'

# if make_gif:
#     print("making gif")
#     os.system('mkdir -p '+'./Figs/event/anim/')
#     animpath='./Figs/event/anim/'
#     inpath=figpath
#     msg='convert  -delay 20  -loop 0   '+figpath+subset_name+'*.png  '+animpath+subset_name+var+'_'+event+'_'+str(DPI)+'DPI.gif'
#     os.system(msg)
