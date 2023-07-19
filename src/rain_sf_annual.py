#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:38:04 2023

@author: jason
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:12:42 2023

@author: jason
"""


import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
# import calendar
# from datetime import timedelta
# from mpl_toolkits.basemap import Basemap

path='/Users/jason/Dropbox/CARRA/CARRA_202209/'
os.chdir(path)

# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# viridis = cm.get_cmap('viridis', 12)
# print(viridis)
# #%%
# print('viridis.colors', viridis.colors)

# fn='/Users/jason/Dropbox/CARRA/ancil/Const.Clim.sfx.nc'
# nc = Dataset(fn, mode='r')

# # fn='/Users/jason/Dropbox/CARRA/CARRA_at_points/site_coords/TIN_Greenland_ccordinates_for_CARRA.csv'
# # df_TIN=pd.read_csv(fn)
# # n=len(df_TIN)
# # size=4

# fn='/Users/jason/Dropbox/CARRA/ancil/Const.Clim.sfx_var_list.csv'
# df=pd.read_csv(fn)
# print(df)
# d=nc.variables.keys()
# # print(d)
# for i in range(len(df)):
#     # if i<42:
#     # if i==1:
#     if i>0:
#         print(i,df.name[i])#,str(df.id[i]))
#         # if df.name[i]=='Fraction of sea':
#         if df.name[i]=='LAND SEA MASK':
#         # if df.name[i]!='INDUSTRIES':
#             mask=nc.variables['var'+str(df.id[i])][0,0,:,:]
            # mask=np.rot90(nc.variables['var'+str(df.id[i])][0,0,:,:].T)

# ------------------------------------------- global plot settings
th=1  # line thickness
font_size=18
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams["mathtext.default"]='regular'

ly='p' # x for console, p for out to png file
res='h' # low res for console, high res for png
if ly=='p':res='h'

ni=1269 ; nj=1069


# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)
# lat=np.rot90(lat.T)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)
# lon=np.rot90(lon.T)

mask_iceland=1
mask_svalbard=1

if mask_iceland:
    mask[((lon-360>-30)&(lat<66.6))]=0
if mask_svalbard:
    mask[((lon-360>-20)&(lat>70))]=0

v=np.where(mask>0)
ice_area=len(v[0])*2.5**2
print('ice area ',ice_area)

# mask=np.rot90(mask.T)

plt.imshow(mask)
#%%

ly='x'
do_plot=0
areax=2.5e3**2

hours=['00','12']

days=np.arange(1,31).astype(str) 

iyear=1990 ; fyear=2022
iyear=2021 ; fyear=2022
n_years=fyear-iyear+1

years=np.arange(iyear,fyear+1).astype(str)
months=np.arange(1,13).astype(str)
# months=np.arange(1,2).astype(str)

rain_tot=np.zeros(((n_years,ni,nj)))

var='rainfall'
# var='snowfall'

for yy,year in enumerate(years):
    for mm,monthx in enumerate(months):
        month=str(monthx).zfill(2)    
        print(year,month)
        if var=='snowfall':
            fn='/Users/jason/0_dat/CARRA_raw/snow/'+year+month+'.grib' # !! actually tp
            sf=xr.open_dataset(fn,engine='cfgrib')
        else:
            fn='/Users/jason/0_dat/CARRA_raw/rain/'+year+month+'.grib'
            ds=xr.open_dataset(fn,engine='cfgrib')
            tirf=ds.variables['tirf'].values
            time=ds.variables['time'].values

        # list(ds.keys())
        # print('extract variable')
            # tp=ds2.variables['tp'].values
            # tirf=tp-tirf
        # print(np.shape(tirf))
        # print(len(time))
        # print(time[0],time[-1])

        for i in range(len(time)):
            # for hh,hour in enumerate(hours):
            #     # print(yy+iyear,day)
            #     # if i==1:
            #     # if i<=3:
        
            #     if i>=0:
                    # #%%
                # plotvar=tirf[i,:,:]
                rain_tot[yy,:,:]+=tirf[i,:,:]*mask
                # plotvar[plotvar<lo]=lo
                # plotvar[((mask<0.1)&(np.isnan(plotvar)))]=0
                # plotvar[mask==0]=np.nan
                # print(i+1,)
                # if do_plot:
                #     plotvar*=mask
                #     v=np.where(plotvar>0)
                #     result=len(v[0])*areax/1e12
                #     units='mm'
        
                #     plt.close()
                #     plt.clf()
                #     fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
                        
                #     # inferno
                #     # cool
                #     cm_name='viridis'
                #     cmap = plt.cm.get_cmap(cm_name)
                #     cm = plt.cm.viridis
                #     cm.set_under('w')
                #     # cm.set_over('b')
                #     lo=0. ; hi=10 ; dx=0.1
                #     clevs=np.arange(lo,hi+dx,dx)
        
                #     mine=plt.contourf(plotvar,clevs,cmap=cm,linewidths=th/1., vmin=lo,vmax=hi,extend='both')
                #     # plt.imshow(plotvar)
                #     # plt.colorbar()
                #     # pp=plt.contourf(x,y,z,clevs,cmap=cm, extend='both')
                #     # ax.grid(off)
                #     ax.axis('off')
            
            
                #     props = dict(boxstyle='round', facecolor='w', alpha=0,edgecolor='w')
            
                #     xx0=0.5 ; yy0=0.95
                #     mult=1.1
                    
                #     ax.text(xx0, yy0, 'Greenland CARRA '+var+' flux',
                #             fontsize=font_size*mult,color='k',ha='center',
                #             bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
            
                #     xx0=0.6 ; yy0=0.06
                #     mult=1.1
                #     ax.text(xx0, yy0, '2022 '+month+' '+str(i+1).zfill(2)+' '+hours[hh]+' UTC',
                #             fontsize=font_size*mult,color='k',bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
                    
                #     du_color_bar=1
                #     with_colorbarname=''
                #     if du_color_bar:
                #         with_colorbarname='_w_colorbar'
                #         width=0.03
                #         cbax = ax.inset_axes([0.92, 0.2, width, 0.5], transform=ax.transAxes)
                #         cbax.set_title(units,fontsize=font_size,c='k',ha='left')
                #         fig.colorbar(mine, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')
            
                #     # clb.set_clim(0,50)
                #     # cbar.draw_all()
                #     if ly == 'x':
                #         plt.show() 
                
                #     DPI=300
                    
                #     if ly == 'p':
                #         var='tirf'
                #         fig_basepath='./Figs/'
                #         fig_basepath='/Users/jason/0_dat/CARRA/202209/Figs/'
                #         figpath=fig_basepath+var+'/'
                #         os.system('mkdir -p '+figpath)
                #         # figpath=fig_basepath+var+'/'+event+'/'+cm_name+'/'
                #         # os.system('mkdir -p '+figpath)
                #         figname=figpath+'2022'+month+str(i+1).zfill(2)+hours[hh]
                #         # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, transparent=True)#, facecolor='w', edgecolor='k')
                #         plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                #         # figname=figpath+'case'
                #             # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

#%%
plt.imshow(rain_tot[0,:,:]*mask)
plt.colorbar()                 
#%%
fig, ax = plt.subplots(figsize=(10, 6))

x=np.arange(iyear,fyear+1).astype(int)

tots=np.zeros(n_years)

for yy in range(n_years):
    tots[yy]=np.sum(rain_tot[yy,:,:]*mask)*areax/1e12

plt.plot(x,tots)
plt.ylabel(var+' over Greenland ice,\nGigatonnes, source: CARRA')

#%%
out=pd.DataFrame({'year':x,'rf':tots})
out.to_csv('/Users/jason/Dropbox/CARRA/CARRA_202209/stats/Greenland_Sept_'+var+'.csv')
#%% plot mean
mean_rain=np.mean(rain_tot[1:31,:,:],axis=0)
np.shape(mean_rain)

anom=rain_tot[32,:,:]-mean_rain

# plotvar=mean_rain
plotvar=anom
plotvar*=mask
# plotvar[mask<0.1]=-0.1

# plotvar[((plotvar>-0.1)&(plotvar<0.0))]=-0.1
units='mm'

print(np.max(plotvar))
plt.close()
plt.clf()
fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
    
# inferno
# cool
cm_name='bwr'
cmap = plt.cm.get_cmap(cm_name)
cm = plt.cm.bwr
# cm.set_under('w')
# cm.set_over('m')
lo=-150 ; hi=-lo ; dx=5
clevs=np.arange(lo,hi+dx,dx)

plotvar[plotvar<0.2]=-0.1

mine=plt.contourf(plotvar,clevs,cmap=cm,linewidths=th/1., vmin=lo,vmax=hi,extend='both')
# plt.imshow(plotvar)
# plt.colorbar()
# pp=plt.contourf(x,y,z,clevs,cmap=cm, extend='both')
# ax.grid(off)
ax.axis('off')


props = dict(boxstyle='round', facecolor='w', alpha=0,edgecolor='w')

xx0=0.5 ; yy0=0.88
mult=1.1

ax.text(xx0, yy0, 'Greenland glaciated area CARRA September '+var+'\nanomaly vs 1991-2020 baseline',
        fontsize=font_size*mult,color='k',ha='center',
        bbox=props,rotation=0,transform=ax.transAxes,zorder=20)

# xx0=0.6 ; yy0=0.06
# mult=1.1
# ax.text(xx0, yy0, '2022 September',
#         fontsize=font_size*mult,color='k',bbox=props,rotation=0,transform=ax.transAxes,zorder=20)


du_color_bar=1
with_colorbarname=''
if du_color_bar:
    with_colorbarname='_w_colorbar'
    width=0.03
    cbax = ax.inset_axes([0.82, 0.2, width, 0.5], transform=ax.transAxes)
    cbax.set_title(units,fontsize=font_size,c='k',ha='left')
    fig.colorbar(mine, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')


# clb.set_clim(0,50)
# cbar.draw_all()

ly='p'
if ly == 'x':
    plt.show() 

DPI=300

if ly == 'p':
    fig_basepath='./Figs/'
    fig_basepath='/Users/jason/0_dat/CARRA/202209/Figs/'
    figpath=fig_basepath+var+'/'
    os.system('mkdir -p '+figpath)
    # figpath=fig_basepath+var+'/'+event+'/'+cm_name+'/'
    # os.system('mkdir -p '+figpath)
    figname=figpath+'202209_'+var+'_anom_map'
    # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, transparent=True)#, facecolor='w', edgecolor='k')
    plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
    # figname=figpath+'case'
    # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

    
    