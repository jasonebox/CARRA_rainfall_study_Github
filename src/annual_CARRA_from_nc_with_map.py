#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:04:02 2021

outputs all years annual .npy and simple-ish maps for rf, tp and t2m
preceeded by /Users/jason/Dropbox/CARRA/CARRA_rainfall_study/src/extract_CARRA_daily_tp_rf_t2m_to_annual_nc_from_3h_GRIB.py

"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
from glob import glob
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
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

inpath='/Users/jason/0_dat/CARRA/output/annual/'
inpath='/Volumes/LaCie/0_dat/CARRA/output/annual/'

os.chdir(path)

ly='p'

res='l'
if ly=='p':res='h'

# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
# mask = np.rot90(mask.T)
# plt.imshow(mask)

# rf is raingall, tp is total precipitation
varnams=['rf','tp','t2m']

years=np.arange(1991,2022).astype('str')
# years=np.arange(2008,2010).astype('str')
# years=np.arange(1991,1992).astype('str')
# years=np.arange(2012,2013).astype('str')

wo=1

ni=1269 ; nj=1069

for i in range(3):
# for i in range(1):
    if i>0:
        if i==2:
            plt.rcParams['axes.facecolor']='w'
            plt.rcParams['savefig.facecolor']='w'
        for yy,year in enumerate(years):
        # for year in years[1:2]:
            if yy>=0:
            # if year=='2017':
            # if yy==2:
            # if ((yy>0=)&(yy<=3)):
        
                fn=inpath+varnams[i]+'_'+year+'.nc'
                print("reading "+fn)
                nc = Dataset(fn, mode='r')
                # print(nc.variables)
                z = nc.variables[varnams[i]][:,:,:]
                # plt.imshow(z[2,:,:])
                # #%%
                if i<2:
                    print("summing "+fn)
                    plotvar = np.sum(z, axis=0)
                else:
                    plotvar = np.mean(z, axis=0)
                    # plotvar=np.array(z)
                # if ( ((int(year)<1998)or (int(year)>2020) )):
                plotvar=np.rot90(plotvar.T)
                if wo:
                    # if i>2:
                    result=plotvar.filled(fill_value=0)
                    # else:
                        # result=plotvar
                        
                    # ofile='./output_annual/'+varnams[i]+'_'+year+'_'+str(ni)+'x'+str(nj)+'_float32.npy'
                    # result.astype('float32').tofile(ofile)        # plt.imshow(plotvar)
                    ofile='./output_annual/'+varnams[i]+'_'+year+'_'+str(ni)+'x'+str(nj)+'_float16.npy'
                    result.astype('float16').tofile(ofile)        # plt.imshow(plotvar)
                    # if yy<=3:plotvar = np.rot90(plotvar.T)
                    #%%
                # plt.imshow(plotvar)
                plotvar_non_fuzzy =plotvar
                varnam2=['rainfall','precip.',r'$(t2m_{max} + t2m_{min})/2$']
                min_value=[0,0,-30]
                max_value=[720,6000,4]
                units=['mm w.e.','mm w.e.','deg C']
                 
                mask_svalbard=1;mask_iceland=1;mask_jan_mayen=1

                plt.close()
                fig, ax = plt.subplots(figsize=(12, 12))

                map_version=1
                if map_version:
                    fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
                    lat=np.fromfile(fn, dtype=np.float32)
                    lat=lat.reshape(ni, nj)
                
                    fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
                    lon=np.fromfile(fn, dtype=np.float32)
                    lon=lon.reshape(ni, nj)

                    fn='./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
                    elev=np.fromfile(fn, dtype=np.float32)
                    elev=elev.reshape(ni, nj)
                    # latx=np.rot90(lat.T)
                    # lonx=np.rot90(lon.T)
                    offset=0
                    lon=lon[offset:ni-offset,offset:nj-offset]
                    lat=lat[offset:ni-offset,offset:nj-offset]
                    ni-=offset*2
                    nj-=offset*2
                    # print(ni,nj)
                    LLlat=lat[0,0]
                    LLlon=lon[0,0]-360
                    # print("LL",LLlat,LLlon)
                    # print("UL",lat[ni-1,0],lon[ni-1,0]-360)
                    lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
                    lat0=lat[int(round(ni/2)),int(round(nj/2))]
                    # print("mid lat lon",lat0,lon0)
                    
                    URlat=lat[ni-1,nj-1]
                    URlon=lon[ni-1,nj-1]
                    # print("LR",lat[0,nj-1],lon[0,nj-1]-360)
                    # print("UR",URlat,URlon)
                    
                    # m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=72, lon_0=-36, resolution='l', projection='lcc')
                    m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=lat0, lon_0=lon0, resolution=res, projection='lcc')
                    x, y = m(lat, lon)
                
                def loadct(i):
                    loval=188
                    if i==0:loval=255
                    print("loval",loval)
                    r=[188,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156, 255, 255]
                    g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28, 255, 255 ]
                    b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196, 0, 255 ]
                    r=[loval,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156]
                    g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28 ]
                    b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196]
                    colors = np.array([r, g, b]).T / 255
                    n_bin = 24
                    cmap_name = 'my_list'
                    # Create the colormap
                    cm = LinearSegmentedColormap.from_list(
                        cmap_name, colors, N=n_bin)
                    cm.set_under('w') #  brown, land
                    # my_cmap.set_over('#E4EEF8')
                    return cm
                
                if i==2:
                    cm='jet'
                
                if i<3:
                    if mask_iceland:
                        mask[((lon-360>-30)&(lat<66.6))]=0
                    if mask_svalbard:
                        mask[((lon-360>-20)&(lat>70))]=0
					# if mask_jan_mayen:
					#     mask[((lon-360>-15)&(lat>66.6)&(lat<75))]=0
					# if mask_iceland:
					#     mask[((lon-360>-30)&(lat<66.6))]=0
					# if mask_svalbard:
					#     mask[0:300,800:]=0
					#                     plotvar[mask==0]=0
                
                plotvar_non_fuzzy=plotvar
                
                plotvar*=mask
                plotvar[mask<0]=0
                plotvar_non_fuzzy[mask<0]=0
                if i==2:
                    plotvar_non_fuzzy[mask==0]=np.nan
                    plotvar[mask==0]=np.nan
                
                areax=2.5e3**2
                mass=np.sum(plotvar[mask>0]*mask[mask>0])/1e9*areax/1000

                if i<2:plotvar_non_fuzzy[mask==0]=-1
                
                # plt.imshow(lon)
                ax = plt.subplot(111)
                tit=year+' CARRA '+varnam2[i]+' over Greenland ice'
                # ax.set_title(tit)
                
                if i<2:cm=loadct(i)
                
                if map_version==0:
                    pp=plt.imshow(plotvar_non_fuzzy, interpolation='nearest', origin='lower', cmap=cm,vmin=min_value[i],vmax=max_value[i]) ; plt.axis('off') 
                
                lon-=360
                if i<2:
                    v=np.where((lon<-45)&(lat<62))
                    maxval=np.max(plotvar_non_fuzzy)
                    minval=np.min(plotvar_non_fuzzy[plotvar_non_fuzzy>1])
                    maxval2=np.max(plotvar_non_fuzzy[v])
                    # print("position of extremes")
                    # print(np.max(plotvar_non_fuzzy))
                    # print(maxval)
                    # print(lat[plotvar==maxval2])
                    # print(lon[plotvar==maxval2])
                    SSWlat=lat[plotvar_non_fuzzy==maxval2][0]
                    SSWlon=lon[plotvar_non_fuzzy==maxval2][0]
                    elev_atSSW=elev[plotvar_non_fuzzy==maxval2][0]
                    # print(lat[plotvar==maxval2])
                    # print(lon[plotvar==maxval2])
                    alllat=lat[plotvar_non_fuzzy==maxval][0]
                    alllon=lon[plotvar_non_fuzzy==maxval][0]
                    minlat=lat[plotvar_non_fuzzy==minval][0]
                    minlon=lon[plotvar_non_fuzzy==minval][0]
                
                if map_version:
                    pp=m.imshow(plotvar, cmap = cm,vmin=min_value[i],vmax=max_value[i]) 
                    # m.axis('off')
                    m.drawcoastlines(color='k',linewidth=0.5)
                    # m.drawparallels([66.6],color='gray')
                    # m.drawparallels([60,70,80],dashes=[2,4],color='k')
                    # m.drawmeridians(np.arange(0.,420.,10.))
                    # m.drawmapboundary(fill_color='aqua')
                    ax = plt.gca()     
                    # plt.title("Lambert Conformal Projection")
                    # plt.show()
                    lons, lats = m(lon, lat)
                    # m.scatter(lons[plotvar_non_fuzzy==maxval2],lats[plotvar_non_fuzzy==maxval2], s=380, facecolors='none', edgecolors='m')
                    m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=400, facecolors='none', edgecolors='k',linewidths=th*2)
                    m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=380, facecolors='none', edgecolors='m',linewidths=th*1)
                    # m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='m')
                
                if i<2: # rf or tp
                    cbar_min=min_value[i]
                    cbar_max=max_value[i]
                    cbar_step=max_value[i]/24
                    cbar_num_format = "%d"
                
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    
                    # plt.colorbar(im)            
                    cbar = plt.colorbar(pp,
                                        orientation='vertical',
                                        ticks=np.arange(cbar_min,
                                        cbar_max+cbar_step, cbar_step),format=cbar_num_format, cax=cax)
                    cbar.ax.set_ylabel(units[i], fontsize = font_size)
                    tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step).astype(int)
                    # print(tickranges)
                    cbar.ax.set_yticklabels(tickranges, fontsize=font_size)
                
                if i==2: # t2m
                    cbar_min=min_value[i]
                    cbar_max=max_value[i]
                    cbar_step=max_value[i]/24
                    cbar_num_format = "%d"

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    
                    # plt.colorbar(im)            
                    cbar = plt.colorbar(pp,orientation='vertical',format=cbar_num_format, cax=cax)
                    cbar.ax.set_ylabel(units[i], fontsize = font_size)
                    # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
                    # # print(tickranges)
                    # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)
                
                # cc=0
                # xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
                # mult=0.7
                # color_code='grey'
                # plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
                #   transform=ax.transAxes,color=color_code) ; cc+=1. 
                
                
                if i<2:
                    cc=0
                    xx0=0.44 ; yy0=0.17 ; dy2=-0.028
                    mult=0.8
                    color_code='k'
                    print()
                    print(year)
                    plt.text(xx0, yy0+cc*dy2,year, fontsize=font_size*mult,
                      transform=ax.transAxes,color=color_code) ; cc+=1.
                
                    msg="{:.0f}".format(mass)+" Gt / y "+varnam2[i]+" total mass flux"
                    print(msg)
                    plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
                      transform=ax.transAxes,color=color_code) ; cc+=1. 
                
                    msg="{:.0f}".format(np.max(plotvar))+" mm / y "+"max "+varnam2[i]+""
                    print(msg)
                    plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
                      transform=ax.transAxes,color=color_code) ; cc+=1. 
                
                    msg="{:.4f}".format(alllat)+" N, "+"{:.4f}".format(abs(alllon))+" W"
                    print(msg)
                    plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
                      transform=ax.transAxes,color=color_code) ; cc+=1. 
                    
                    # msg="{:.0f}".format(np.max(plotvar[v]))+" mm / y "+"max "+varnam2[i]+" SSW"
                    # print(msg)
                    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
                    #   transform=ax.transAxes,color=color_code) ; cc+=1. 
                
                    # msg="{:.4f}".format(SSWlat)+" N, "+"{:.4f}".format(abs(SSWlon))+" W"
                    # print(msg)
                    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
                    #   transform=ax.transAxes,color=color_code) ; cc+=1. 
                
                if ly == 'x':
                    plt.show()
                
                DPIs=[150,300]
                # DPIs=[150]
                
                if ly =='p':
                    for DPI in DPIs:
                        figpath='./Figs/annual/'+varnams[i]+'/'
                        os.system('mkdir -p '+figpath)
                        figpath='./Figs/annual/'+varnams[i]+'/'+str(DPI)+'/'
                        os.system('mkdir -p '+figpath)
                        figname=figpath+year+'_'+varnams[i]+'_over_ice_'+str(DPI)+'DPI'
                        if i<2:
                            plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
                        else:
                            plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)#, facecolor=fig.get_facecolor(), edgecolor='none')
#             #%%
# make_gif=1

# if make_gif:
#     animpath='/Users/jason/Dropbox/CARRA/anim/'
#     # inpath='/Users/jason/Dropbox/CARRA/Figs/'
#     inpath=figpath
#     msg='convert  -delay 70  -loop 0   '+inpath+'*'+varnams[i]+'*'+'.png  '+animpath+varnams[i]+'_1991-2021.gif'
#     os.system(msg)