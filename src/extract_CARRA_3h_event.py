#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:52:21 2021

@author: Jason Box

"""
import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import calendar
from datetime import timedelta


outpath='/Users/jason/0_dat/CARRA/output/annual/' 
outpath='/Volumes/LaCie/0_dat/CARRA/output/annual/' 

ch=2 # tp
# ch=1 # rf
# ch=0 # t2m
prt_time=0
tst_plot=0

# years=np.arange(2020,2021).astype('str')
# years=np.arange(2021,2022).astype('str')
# years=np.arange(1991,2022).astype('str')
years=np.arange(2017,2018).astype('str')
years=np.arange(1999,2000).astype('str')

# for a later version that maps the result
ni=1269 ; nj=1069

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
raw_path='/Users/jason/0_dat/CARRA_raw/'
# raw_path='/Volumes/LaCie/0_dat/CARRA/CARRA_raw/'
    
os.chdir(path)


fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)


#functions to read in files
def gett2m(val, var,year,suffix):
    fn=raw_path+'CARRA-West_T2m_mean_'+year+'_'+val+'h.'+suffix
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    t2m=ds.variables[var].values#-273.15
    time=ds.variables['time'].values#-273.15
    return t2m,time

def gettp(val, var,year,suffix):
    fn=raw_path+'CARRA-West_prec_'+year+'_3h_acc_'+val+'h.'+suffix
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    tp=ds.variables[var].values 
    return tp

def get_rf(t2m,tp):
    # rain phasing
    rainos=0.
    x0=0.5 ; x1=2.5
    # x0=-2 ; x1=0
    x0-=rainos
    x1-=rainos
    y0=0 ; y1=1
    a1=(y1-y0)/(x1-x0)
    a0=y0-a1*x0

    f=np.zeros((ni,nj))
    # np.squeeze(t2m, axis=0)
    v=np.where(((t2m>x0)&(t2m<x1)))
    f[v]=t2m[v]*a1+a0
    v=np.where(t2m>x1) ; f[v]=1
    v=np.where(t2m<x0) ; f[v]=0
    
    rf=tp*f
    return rf


# ---------- read in data
time_list_t2m=['00_UTC_fl_3-6','00_UTC_fl_6-9','00_UTC_fl_9-12','00_UTC_fl_12-15','12_UTC_fl_3-6','12_UTC_fl_6-9','12_UTC_fl_9-12','12_UTC_fl_12-15']
time_list_tp =['00_UTC_fl_6-3','00_UTC_fl_9-6','00_UTC_fl_12-9','00_UTC_fl_15-12','12_UTC_fl_6-3','12_UTC_fl_9-6','12_UTC_fl_12-9','12_UTC_fl_15-12']

time_list_hours=['03','06','09','12','15','18','21','00']

fn='/Users/jason/Dropbox/CARRA/CARRA_rain/stats/rf_extremes.csv'
df=pd.read_csv(fn)

print(df.columns)

# minval=300 ;scaling_factor=3000
# v=np.where(df.maxlocalrate>minval)

v=np.where(df.Gt_overall>2.2)
# v=v[0]
print(v[0])

# print()
minval=np.min(df.maxlocalrate[v[0]]) ; scaling_factor=1500
listx=[v[0][-1]] # 2021
# listx=[v[0][-4]] # 2017
# listx=v[0]
for i in listx:
    # co_index=(df.maxlocalrate[i]-200)/(np.max(df.maxlocalrate)-200)*255
    timex=pd.to_datetime(df.date[i])
    ddx=int(timex.strftime('%j'))
    ddx0=ddx-1 ; ddx1=ddx+1
    ymd=timex.strftime('%Y-%m-%d')
    year=timex.strftime('%Y')
    timex0=pd.to_datetime(df.date[i])- timedelta(days=1)
    ymd0=timex0.strftime('%Y-%m-%d')
    
    timex1=pd.to_datetime(df.date[i])+ timedelta(days=1)
    ymd1=timex1.strftime('%Y-%m-%d')
    
    print(i,ymd0,ymd,ymd1,ddx,df.lon[i],df.lat[i],df.Gt_overall[i],df.maxlocalrate[i])
    
    if calendar.isleap(int(year)):
        print('is leap year')
        n_days=366
    else:
        print('not a leap year')
        n_days=365

    tp_3h_4dcube=np.zeros((((8,n_days,ni,nj))))
    rf_3h_4dcube=np.zeros((((8,n_days,ni,nj))))
    t2m_3h_4dcube=np.zeros((((8,n_days,ni,nj))))
    
    var=['t2m', 'mn2t6', 'tp', 'time']
    
    for cc in np.arange(0,8):
    # for cc in np.arange(0,1):
    # for cc in np.arange(6,8):
        print('cc',cc)
        t_varnam=var[0]
        if cc==0:t_varnam=var[1]
        if cc==4:t_varnam=var[1]
        # if str(year)==2021:
        suffix='grb2'
        # else:suffix='nc'
        t2m,time=gett2m(time_list_t2m[cc], t_varnam,year,suffix) 
        # if str(year)==2021:
        suffix='grib'
        # else:suffix='nc'            
        tp=gettp(time_list_tp[cc], var[2],year,suffix)
        
        tp_3h_4dcube[cc,:,:,:]=tp
        t2m_3h_4dcube[cc,:,:,:]=t2m
        # time=gett2m(time_list_t2m[cc], var[3])
        if cc==0:
            timex=pd.to_datetime(time).strftime('%Y-%m-%d-%H')
            jday=pd.to_datetime(time).strftime('%j')

#%%
    for j,dd in enumerate(range(ddx0-1,ddx1)):
        rf_3h=np.zeros((ni,nj))
        t2m_3h=np.zeros((ni,nj))
        tp_3h=np.zeros((ni,nj))
        ymdx=ymd0
        if j==1:ymdx=ymd
        if j==2:ymdx=ymd1
        for hh in range(8):            
            print(j,hh,ymdx,dd)
            
        # ------------------------------------------- rain fraction
            t2m=t2m_3h_4dcube[hh,dd,:,:]-273.15
            rfx=get_rf(t2m,tp_3h_4dcube[hh,dd,:,:]) ; rf_rot_T=np.rot90(rfx.T)
            rf_3h+=np.rot90(rfx.T)
            t2m_rot_T=np.rot90(t2m.T)
            t2m_3h+=t2m_rot_T
            tp_3h+=np.rot90(tp_3h_4dcube[hh,dd,:,:].T)
            
            outpath='/Users/jason/0_dat/CARRA/output/event/rf/'
            # rf_rot_T.astype('float16').tofile(outpath+ymd+'-'+str((hh+1)*3)+'_1269x1069.npy')
            do_plot=1
            if do_plot:
                # plt.imshow(rf_3h,vmin=0,vmax=20)
                plt.close()
                # plt.imshow(rf_3h)
                plt.imshow(rf_rot_T)
                plt.grid(False)
                plt.colorbar()
                plt.title(ymdx+' '+str(hh))
                plt.show()
#%%
for yy,year in enumerate(years):


                        #%%

        tp=0
        
        t2m_daily_3dcube=np.zeros(((n_days,ni,nj)))
        tp_daily_3dcube=np.zeros(((n_days,ni,nj)))
        rf_daily_3dcube=np.zeros(((n_days,ni,nj)))
        
        for dd in range(n_days):
        # for dd in range(2):
            # if dd==190:
            if dd!=1900:
                print(dd)
                rf_daily=np.zeros((ni,nj))
                t2m_daily=np.zeros((ni,nj))
                tp_daily=np.zeros((ni,nj))
                for hh in range(8):            
                    # print(hh)
                # ------------------------------------------- rain fraction
                    t2m=t2m_3h_4dcube[hh,dd,:,:]-273.15
                    rfx=get_rf(t2m,tp_3h_4dcube[hh,dd,:,:]) ; rf_rot_T=np.rot90(rfx.T)
                    rf_daily+=np.rot90(rfx.T)
                    t2m_rot_T=np.rot90(t2m.T)
                    t2m_daily+=t2m_rot_T
                    # temp_tp=
                    tp_daily+=np.rot90(tp_3h_4dcube[hh,dd,:,:].T)
                    # plt.imshow(rf_rot_T,vmin=0,vmax=10)
                    # plt.imshow(t2m_rot_T)
                    # plt.grid(False)
                    # plt.colorbar()
                    # plt.title(hh)
                    # plt.show()
                t2m_daily_3dcube[dd,:,:]=t2m_daily/8        
                tp_daily_3dcube[dd,:,:]=tp_daily     
                rf_daily_3dcube[dd,:,:]=rf_daily
                
                # plt.imshow(np.rot90(tp_daily_3dcube[dd,:,:].T))
                # plt.axis(False)
                # plt.colorbar()
                # plt.title(hh)
                # plt.show()
                    # tpx=tp[hh,dd,:,:] ; tp_rot_T=np.rot90(tpx.T)
                    # t2mx=t2m[hh,dd,:,:]-273.15 ; t2m_rot_T=np.rot90(t2mx.T)
                    
                    # timestamp_string=str(timex[dd][:-3])+'-'+time_list_hours[cc]
                    # print(timestamp_string)
                    # outpath='/Users/jason/0_dat/CARRA/output/rf/'
                    # rf_rot_T.astype('float16').tofile(outpath+timestamp_string+'.npy')
                    # outpath='/Users/jason/0_dat/CARRA/output/tp/'
                    # tp_rot_T.astype('float16').tofile(outpath+timestamp_string+'.npy')
                    # outpath='/Users/jason/0_dat/CARRA/output/t2m/'
                    # t2m_rot_T.astype('float16').tofile(outpath+timestamp_string+'.npy')
        
                    # do_plt=0
                    # if do_plt:
                    #     plt_x(rf_rot_T,0,15,'rf','cbarnam','mm/3h')
                    #     plt_x(tp_rot_T,0,25,'tp','cbarnam','mm/3h')
                    #     plt_x(t2m_rot_T,-12,12,'t2m','cbarnam','deg. C')
                        # plt.close()
                    # plt.imshow(rf_rot_T,vmin=0,vmax=15)
                    # plt.title(timestamp_string)
                    # plt.axis('off')
                    # # plt.colorbar()
                    # clb = plt.colorbar(fraction=0.046/2., pad=0.02)
                    # clb.ax.set_title('mm/3h',fontsize=7)
                    # ly='p'
                    # fig_path='/Users/jason/0_dat/CARRA/temp/'
                    # if ly=='p':
                    #     plt.savefig(fig_path+'_rf_'+timestamp_string+'.png', bbox_inches='tight')#, dpi=DPI, facecolor=bg, edgecolor=fg)


        # varnam='rf' ; varxx=rf_daily_3dcube


        
        write_nc('rf',rf_daily_3dcube,n_days,ni,nj,outpath)
        write_nc('tp',tp_daily_3dcube,n_days,ni,nj,outpath)
        write_nc('t2m',t2m_daily_3dcube,n_days,ni,nj,outpath)

#             #%%
#         import datetime
#         import numpy as np
#         import os
#         from netCDF4 import Dataset
#         import pandas as pd
#         import matplotlib.pyplot as plt
#         import xarray as xr
#         import calendar
        
#         outpath='/Volumes/LaCie/0_dat/CARRA/output/annual/' 
#         year='1996'
        
#         if calendar.isleap(int(year)):
#             print('is leap year')
#             n_days=366
#         else:
#             print('not a leap year')
#             n_days=365

#         varnam='tp'
#         varnam='rf'
#         # varnam='t2m'
#         ofile=outpath+varnam+'_'+year+'.nc'

#         ds=xr.open_dataset(ofile)
#         print(ds.variables)
        
#         # print(np.shape(ds.rf))
#         # print(np.shape(ds.lat))
#         print(ds.time)

#         ly='p'
#         for dd in range(n_days):
#         # for dd in range(3):
#             date=datetime.datetime.strptime(str(int(year))+' '+str(int(dd+1)), '%Y %j')
#             timex=pd.to_datetime(date).strftime('%Y-%m-%d')
#             print(timex)
#             plt.close()
#             if varnam=='rf':
#                 plt.imshow(ds.rf[dd,:,:],vmin=0,vmax=30,cmap='jet')
#             # plt.imshow(ds.t2m[1,:,:])
#             if varnam=='tp':
#                 plt.imshow(ds.tp[dd,:,:],vmin=0,vmax=120,cmap='jet')
#             plt.axis(False)
#             plt.colorbar()
#             plt.title(varnam+' '+timex)
#             if ly == 'x':plt.show()
            
#             if ly == 'p':
#                 fig_path='/Users/jason/0_dat/CARRA_temp/'+varnam+'/'+str(year)+'/'
#                 os.system('mkdir -p '+fig_path)
#                 figname=fig_path+timex+'.png'
#                 plt.savefig(figname, bbox_inches='tight', dpi=250)
# #%%
#         ds=xr.open_dataset('/Volumes/LaCie/0_dat/CARRA/output/rf_2002.nc')
#         print(ds.variables)