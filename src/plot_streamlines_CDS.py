#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:12:42 2023

@author: jason
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
# from datetime import datetime 

os.chdir('/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/')


# ------------------------------------------- global plot settings
th=1  # line thickness
font_size=16
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'

ni=1269 ; nj=1069

#functions to read in files
def gett2m(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    t2m=ds.variables['t2m'].values-273.15
    time=ds.variables['time'].values
    return t2m,time

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

def get_UV(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    
    # print(ds.variables)
    # P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u'].values
    V=ds.variables['v'].values
    # W=ds.variables['wz'].values
    time=ds.variables['time'].values
    # P=P[0]
    return U,V,time

def get_UVT(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    
    # print(ds.variables)
    # P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u'].values
    V=ds.variables['v'].values
    T=ds.variables['t'].values-273.15
    W=ds.variables['wz'].values
    time=ds.variables['time'].values
    # P=P[0]
    return U,V,T,W,time


days=np.arange(14,15).astype(str) ; event='2017-09-14'

density=1.75
hour0=0 ; hour1=7

test=0
subset_name=''

if test:
    density=1
    ly='x'
    hour0=0 ; hour1=1
    days=np.arange(2,3).astype(str) ; event='test'

# levs=[950,900,800,700,600,500]
# dictx={'pressure_level': [
#     '500',
#     '600', '700', '750',
#     '800', '825', '850',
#     '875', '900', '925',
#     '950', '1000',
#     ]}

# levs=np.array(dictx['pressure_level'])
# print(levs)
# print(np.flipud(levs))
# levs=np.flipud(levs)
# print(levs)
n_hours=8


for i,day in enumerate(days):
    # if int(day)>=0:
    # if int(day)==2:
    if i>=0:
        fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvt_1000-500hPa_12_levs.grib'
        fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvw_1000-300hPa_14_levs.grib'
        # fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvwt_3_levs.grib'
        ds=xr.open_dataset(fn,engine='cfgrib')
        ds.variables
        # Ux,Vx,Wx,P,timex=get_UVW(fn)
        Ux,Vx,Wx,P,times=get_UVW(fn)
        time=pd.to_datetime(times)
        # print(P)
        levs=np.array(P)
        do_daily=0
        for hh in range(n_hours):
            # print(hh)
            # if hh>3:
            if hh==5:
            # if hh==6: # 18 UTC
                # print(hh,times[hh])

                for lev_index,lev in enumerate(levs):
                    # if lev_index==11: #800 hPa
                    # if lev_index>=0: 
                    if lev==925:
                        print(lev_index,lev)
            
                        units='m s$^{-1}$'
            
                        # for i in range(hour0+1,hour1):
                        # for i in range(hour0,hour0+1):
                        plt.close()
                        plt.clf()
                        fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
                        print(str(time[i].strftime('%Y %b %d %H')))
                        if do_daily:
                            U=np.mean(Ux[:,lev_index,:,:],axis=0)
                            V=np.mean(Vx[:,lev_index,:,:],axis=0)
                            # T=np.mean(Tx[:,lev_index,:,:],axis=0)
                        else:
                            U=Ux[hh,lev_index,:,:]
                            V=Vx[hh,lev_index,:,:]
                            W=Wx[hh,lev_index,:,:]
                        Y, X = np.mgrid[0:ni,0:nj]
                        
                        subset_it=1
                        subset_name=''
                        
                        if subset_it:
        
                            ni=1269 ; nj=1069
        
                            # fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
                            # lat=np.fromfile(fn, dtype=np.float32)
                            # lat=lat.reshape(ni, nj)
        
                            # fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
                            # lon=np.fromfile(fn, dtype=np.float32)
                            # lon=lon.reshape(ni, nj)
        
                            max_value=12
                            max_value=36
        
                            # 50,3000,200
                            max_value=24
                            max_value=36
        
                            # lon=np.rot90(lon.T)
                            # lat=np.rot90(lat.T)
                            # lat=lat[yc0:yc1,xc0:xc1]
                            # lon=lon[yc0:yc1,xc0:xc1]                        
                            subset_name='_subset'
                            xc0=210 ; xc1=400 # wider and taller
                            yc0=1000 ; yc1=1210
        
                            U=np.rot90(U.T)
                            U=U[yc0:yc1,xc0:xc1]
                            U=np.rot90(U.T)
        
                            V=np.rot90(V.T)
                            V=V[yc0:yc1,xc0:xc1]
                            V=np.rot90(V.T)
                            
                            W=np.rot90(W.T)
                            W=W[yc0:yc1,xc0:xc1]
                            W=np.rot90(W.T)
                            
                            ni=yc1-yc0 ; nj=xc1-xc0
                            Y, X = np.mgrid[0:ni,0:nj]
                            density=0.75
        
            # plt.imshow(W)
                        speed = np.sqrt(U**2 + V**2)
                
                        lw = 2.5*speed / speed.max()
        
                        # print(np.shape(speed))
            
                        color_by_speed=1

                        # if color_by_speed:
                        #     lo=5; hi=35
                        #     # color_var=T
                        #     color_var=W
                        #     # color_var[color_var<lo]=lo
                        #     # color_var[color_var>hi]=hi
                        #     cm_name='viridis'
                        #     cm_name='bwr'
                        #     cmap = plt.cm.get_cmap(cm_name)
                        
                        if color_by_speed:
                            lo=5; hi=35
                            # color_var=T
                            color_var=speed
                            color_var[color_var<lo]=lo
                            color_var[color_var>hi]=hi
                            cm_name='viridis'
                            cm_name='cividis'
                        #     cmap = plt.cm.get_cmap(cm_name)
                        else:
                            cm_name='bwr'
                            cmap = plt.cm.get_cmap(cm_name)
                            lo=-15; hi=-lo
                            # color_var=T
                            # hi=np.max(color_var)
                            # lo=np.min(color_var)
                            # if abs(lo)>hi:hi=abs(lo)
                            # color_var[0,0]=hi; color_var[0,1]=lo
                            extreme=10.
                            lo=-extreme; hi=-lo
                            # color_var=T
                            # hi=np.min(color_var)
                            color_var[color_var<-extreme]=-extreme
                            color_var[color_var>extreme]=extreme
                            
                        strm = ax.streamplot(X, Y, U, V,density=density,color=color_var,cmap=cm_name,linewidth=lw,arrowstyle='->',
                                     broken_streamlines=False)
                        # ax.set_clim(vmin=-his[lev_index], vmax=his[lev_index])
            
                        # ax.grid(off)
                        ax.axis('off')
                
                        mult=1.2
                        color_code='k'
                        props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
        
                        xx0=0.4 ; yy0=0.965
                        # ax.text(xx0, yy0, 'CARRA '+lev+' hPa winds and air temperatures',
                        #         fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
                
                        ax.text(xx0, yy0, str(int(lev))+' hPa '+time[hh].strftime('%Y %b %d %HUTC'),
                                fontsize=font_size*mult,color=color_code,bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
                
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
                            width=0.025
                            cbax = ax.inset_axes([1.02, 0.51, width, 0.41], transform=ax.transAxes)
                            cbax.set_title(units,fontsize=font_size*0.9,c='k',ha='left')
                            cbax.tick_params(labelsize=font_size*0.9) 
                            fig.colorbar(strm.lines, ax=ax, cax=cbax, shrink=0.7, orientation='vertical',extend='both')
            
                
                        # clb.set_clim(0,50)
                        # cbar.draw_all()
                        ly='p'
                        if ly == 'x':
                            plt.show() 
                    
                        DPI=300
                        

                        if ly == 'p':
                            var='streamline'
                            # fig_basepath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'
                            # figpath=fig_basepath+var+'/'
                            # os.system('mkdir -p '+figpath)
                            # figpath=fig_basepath+var+'/'+event+'/'
                            # os.system('mkdir -p '+figpath)
                            # figpath=fig_basepath+var+'/'+event+'/'+cm_name+'/'
                            figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/subset/'
                            # os.system('mkdir -p '+figpath)
                            if do_daily:
                                figname=figpath+time[hh].strftime('%Y %m %d')+'_'+str(int(lev))+with_colorbarname+subset_name
                            else:
                                figname=figpath+time[hh].strftime('%Y %m %d %H')+'_'+str(int(lev))+with_colorbarname+subset_name
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