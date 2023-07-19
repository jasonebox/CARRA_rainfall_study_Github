#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:04:02 2021

@author: jeb 

reads in daily .nc files and outputs multi-annual .csv for rf and tp of extremes over a threshold of rate and total ice sheet mass flux
preceeded by /Users/jason/Dropbox/CARRA/CARRA_rainfall_study/src/extract_CARRA_daily_tp_rf_t2m_to_annual_nc_from_3h_GRIB.py

"""
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob
from netCDF4 import Dataset
# from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
os.chdir(path)

datapath='/Users/jason/0_dat/CARRA/output/annual/'
datapath='/Volumes/LaCie/0_dat/CARRA/output/annual/'
# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
# plt.imshow(mask)

# rf is raingall, tp is total precipitation
varnams=['rf','tp']

years=np.arange(1991,2022).astype('str')
# years=np.arange(2017,2018).astype('str')
# years=np.arange(1996,1997).astype('str')
# years=np.arange(1997,1998).astype('str')
# years=np.arange(2021,2022).astype('str')

ni=1269 ; nj=1069

days=np.arange(256,257)
days=np.arange(244,304) # sept oct
days=np.arange(1,364) # sept oct

wo=1

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)


fn='./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
elev=np.fromfile(fn, dtype=np.float32)
elev=elev.reshape(ni, nj)

mask_svalbard=1;mask_iceland=1
if mask_iceland:
    mask[((lon-360>-30)&(lat<66.6))]=0
if mask_svalbard:
    mask[((lon-360>-20)&(lat>70))]=0
    
latx=np.rot90(lat.T)
lonx=np.rot90(lon.T)
elevx=np.rot90(elev.T)
# res='l'
# lonx=lonx-360



mask = np.rot90(mask.T)

# plt.imshow(mask)
# # plt.colorbar()

max_val=[60,200]
areax=2.5e3**2
for i in range(2):
    # if i==1: # tp
    # if i==0: # rf
    if i>=0: # both
        if varnams[i]=='tp':
               hi_mass_flux=15
               hi_ppt_rate=200
        if varnams[i]=='rf':
               hi_mass_flux=1
               hi_ppt_rate=200               
        if wo:
            ofile='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/stats/'+varnams[i]+'_extremes.csv'
            out=open(ofile,'w+')
            out.write('date,Gt_overall,maxlocalrate,lat,lon,elev\n')

        for yy,year in enumerate(years):
        # for year in years[1:2]:
            if yy>=0:
            # if year=='2021':
            # if yy<2:
            # if ((yy>0=)&(yy<=3)):
                fn=datapath+varnams[i]+'_'+year+'.nc'
                print("reading "+fn)
                nc = Dataset(fn, mode='r')
                # print(nc.variables)
                z = nc.variables[varnams[i]][:,:,:]
                # plt.imshow(z[2,:,:])
                # #%%
                # print("summing "+fn)
                # plotvar = np.sum(z, axis=0)
                for day_index in days:
                    date=datetime.strptime(year+' '+str(day_index+1), '%Y %j')

                    # print(day_index,date)
                    
                    plotvar=z[day_index,:,:]
                    # if ( ((int(year)>1997)&(int(year)<2021) )):
                    #     plotvar=np.rot90(plotvar.T)
                    
                    plotvar[mask==0]=0

                    
                    mass=np.sum(plotvar[mask>0])/1e9*areax/1000

                    if i<2:plotvar[mask==0]=-1
                                            
                    maxval=np.max(plotvar)
                    alllat=latx[plotvar==maxval][0]
                    alllon=lonx[plotvar==maxval][0]
                    allelev=elevx[plotvar==maxval][0]
                    
                    # if mass>0.1:
                    if maxval>hi_ppt_rate or mass>hi_mass_flux:

                        msg=date.strftime('%d/%m/%Y')+",{:.1f}".format(mass)+", {:.0f}".format(np.max(plotvar)) +\
                        ",{:.4f}".format(alllat)+", "+"{:.4f}".format(alllon-360)+", "+"{:.0f}".format(allelev)+"\n"
                        print(msg)
                        do_plot=1
                        
                        if do_plot:
                            ly='p'
                            plt.close()
                            plt.imshow(plotvar,vmin=0,vmax=max_val[i])
                            plt.axis(False)
                            plt.colorbar()
                            plt.title(msg)
                            if ly == 'x':
                                plt.show()
    
                            DPIs=[150]
                            # DPIs=[150]
    
                            if ly =='p':
                                for DPI in DPIs:
                                    figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/extremes/'+varnams[i]+'/'
                                    os.system('mkdir -p '+figpath)
                                    plt.savefig(figpath+varnams[i]+'_'+date.strftime('%Y-%m-%d')+'.png', bbox_inches='tight', dpi=DPI)

                        out.write(msg)
                       

        out.close()
        os.system('head -n1 '+ofile) 