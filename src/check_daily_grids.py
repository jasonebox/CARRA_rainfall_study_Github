#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 07:59:35 2022

@author: jason
"""

import datetime
import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import calendar

outpath='/Volumes/LaCie/0_dat/CARRA/output/annual/'
# outpath='/Users/jason/0_dat/CARRA/output/annual/'
 
year='2008'
# year='2020'

if calendar.isleap(int(year)):
    print('is leap year')
    n_days=366
else:
    print('not a leap year')
    n_days=365

varnam='tp'
# varnam='rf'
# varnam='t2m'
ofile=outpath+varnam+'_'+year+'.nc'

os.system('ls -lF '+ofile)

ds=xr.open_dataset(ofile)
print(ds.variables)

# print(np.shape(ds.rf))
# print(np.shape(ds.lat))
print(ds.time)

ly='x'
# for dd in range(n_days):
for dd in range(2):
    date=datetime.datetime.strptime(str(int(year))+' '+str(int(dd+1)), '%Y %j')
    timex=pd.to_datetime(date).strftime('%Y-%m-%d')
    print(timex)
    plt.close()
    if varnam=='rf':
        plt.imshow(ds.rf[dd,:,:],vmin=0,vmax=30,cmap='jet')
    if varnam=='t2m':
        plt.imshow(ds.t2m[dd,:,:])
    if varnam=='tp':
        plt.imshow(ds.tp[dd,:,:],vmin=0,vmax=120,cmap='jet')
    plt.axis(False)
    plt.colorbar()
    plt.title(varnam+' '+timex)
    if ly == 'x':plt.show()
    
    if ly == 'p':
        fig_path='/Users/jason/0_dat/CARRA_temp/'+varnam+'/'+str(year)+'/'
        os.system('mkdir -p '+fig_path)
        figname=fig_path+timex+'.png'
        plt.savefig(figname, bbox_inches='tight', dpi=250)