#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:04:02 2021

@author: jeb and AW
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
from glob import glob
from netCDF4 import Dataset
# from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 
import numpy as np

# CARRA grid info
# Lambert_Conformal()
#     grid_mapping_name: lambert_conformal_conic
#     standard_parallel: 72.0
#     longitude_of_central_meridian: -36.0
#     latitude_of_projection_origin: 72.0
#     earth_radius: 6367470.0
#     false_easting: 1334211.3405653758
#     false_northing: 1584010.8994621644
#     longitudeOfFirstGridPointInDegrees: 302.903
#     latitudeOfFirstGridPointInDegrees: 55.81

AW=0
path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
if AW:path='/Users/jason/Dropbox/CARRA/prog/map_CARRA_west/'
os.chdir(path)

ly='x'

res='l'
if ly=='p':res='h'

# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size

# # read ice mask
# fn='./ancil/CARRA_W_domain_ice_mask.nc'
# nc2 = Dataset(fn, mode='r')
# # print(nc2.variables)
# mask = nc2.variables['z'][:,:]
# # mask = np.rot90(mask.T)
# # plt.imshow(mask)

# rf is raingall, tp is total precipitation
varnams=['rf','tp','t2m']
varnams=['tp']

iyear=2008 ; fyear=2010 ; n_years=fyear-iyear+1
years=np.arange(1991,2022).astype('str')
years=np.arange(iyear,fyear+1).astype('str')
# years=np.arange(1998,1999).astype('str')
# years=np.arange(2012,2013).astype('str')

wo=1

ni=1269 ; nj=1069

n_days=365

for i,varnam in enumerate(varnams):
    # if i==0:
    if i>=0:
        totvar=np.zeros(((n_days,ni,nj)))
        for yy,year in enumerate(years):
        # for year in years[1:2]:
            if yy>=0:
            # if year=='2017':
                
            # if yy==2:
            # if ((yy>0=)&(yy<=3)):
        
                fn='/Users/jason/0_dat/CARRA/output/annual/'+varnams[i]+'_'+year+'.nc'
                print("reading",fn)
                nc = Dataset(fn, mode='r')
                # print(nc.variables)
                z = nc.variables[varnams[i]][:,:,:]
                # print(z.shape)
                totvar.shape
                totvar+=np.array(z[0:365,:,:])
                # sasa
                #%%
                plotvar = np.mean(totvar, axis=0)
                #%%
                # plt.imshow(z[2,:,:])
                # #%%
                print("working")

                if i!=2:
                    plotvar = np.sum(z[ss0:ss1,:,:], axis=0)
                else:
                    
                
                if ( ((int(year)>1997)&(int(year)<2021))):
                    plotvar=np.rot90(plotvar.T)           
                
                plt.close()
                plt.imshow(plotvar)
                plt.colorbar()
                plt.axis('off')
                plt.title(varnam+' '+year+' '+season)
                plt.show()
                # plt.colorbar()

                if wo:
                    result=plotvar.filled(fill_value=0)
                    # opath='./output_seasonal/'
                    opath='/Users/jason/0_dat/CARRA/CARRA_rain/output_seasonal/'
                    opath='/Users/jason/Dropbox/CARRA/CARRA_rain/output_seasonal/'
                    os.system('mkdir -p '+opath)
                    ofile=opath+varnams[i]+'_'+year+season_name[ss]+str(ni)+'x'+str(nj)+'_float16.npy'
                    result.astype('float16').tofile(ofile)        # plt.imshow(plotvar)
                    print('writing '+ofile)
                # if yy<=3:plotvar = np.rot90(plotvar.T)
                # plt.imshow(plotvar)
  
    