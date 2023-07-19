#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outputs to map_daily_extreme_CARRA.py

updated Nov 2022
@author: Jason Box, GEUS, jeb@geus.dk

"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 

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
annual_nc_path='/Users/jason/0_dat/CARRA/output/'
annual_nc_path='/Users/jason/0_dat/CARRA/output/annual/'
# annual_nc_path='/Volumes/LaCie/0_dat/CARRA/output/annual/'


# --------------------------------------------

ly='x'
mask_out_ice=0

res='l'
if ly=='p':res='h'

# global plot settings
th=1
font_size=16
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
varnams=['rf','tp','sf']

years=np.arange(1998,2021).astype('str')
# years=np.arange(2007,2008).astype('str')
years=np.arange(2017,2018).astype('str')

ni=1269 ; nj=1069

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

days=np.arange(256,257)
# days=np.arange(244,304) # sept oct
# days=np.arange(1,365) 
# days=np.arange(268,273) # sept oct

# for i in range(3):
#     if i==0: # rf
i=0
# if i==1: # tp
# if i==2: # sf
for yy,year in enumerate(years):
# for year in years[1:2]:
    # if yy>=0:
    if year=='2017':
    # if int(year)<2008:
    # if yy==2:
    # if ((yy>0=)&(yy<=3)):
        print(yy)
        if i<2:
            fn=annual_nc_path+varnams[i]+'_'+year+'.nc'
            print("reading "+fn)
            nc = Dataset(fn, mode='r')
            # print(nc.variables)
            z = nc.variables[varnams[i]][:,:,:]
        if i==2:
            fn=annual_nc_path+varnams[0]+'_'+year+'.nc'
            print("reading "+fn)
            nc = Dataset(fn, mode='r')
            # print(nc.variables)
            rfx = nc.variables[varnams[0]][:,:,:]

            fn=annual_nc_path+varnams[1]+'_'+year+'.nc'
            print("reading "+fn)
            nc = Dataset(fn, mode='r')
            # print(nc.variables)
            tpx = nc.variables[varnams[1]][:,:,:]
            
            z=tpx-rfx
        # z=np.rot90(z.T)
        # plt.imshow(z[2,:,:])
        # print("summing "+fn)
        # plotvar = np.sum(z, axis=0)
        # if yy<=3:plotvar = np.rot90(plotvar.T)
        # plt.imshow(plotvar)
        # #%%
        # year='2017'
        # i=0
        # fn=annual_nc_path+varnams[i]+'_'+year+'.nc'
        # print("reading "+fn)
        # nc = Dataset(fn, mode='r')
        # # print(nc.variables)
        # z = nc.variables[varnams[i]][:,:,:]
        map_version=1

        if map_version:
            fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
            lat=np.fromfile(fn, dtype=np.float32)
            lat=lat.reshape(ni, nj)
        
            fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
            lon=np.fromfile(fn, dtype=np.float32)
            lon=lon.reshape(ni, nj)
        
            # print(z.shape)
            # plt.rcParams['axes.grid'] = False
            # x=np.rot90(z[248,:,:].T)
            # # x[:,300]=0
            # x[1010:1200,295]=80
            # # print(len(x[1010:1200,295]))
            

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

        varnam2=['rainfall','precip.',r'$(t2m_{max} + t2m_{min})/2$']
        varnam2=['rainfall','precip.','snowfall']
        units=['mm w.e.','mm w.e.','mm w.e.']
        
        mask_iceland=1
        mask_svalbard=1
        
        if i<2:
            if mask_iceland:
                mask[((lon-360>-30)&(lat<66.6))]=0
            if mask_svalbard:
                mask[((lon-360>-20)&(lat>70))]=0
        # plt.imshow(lon);plt.colorbar()

#%%
        
        max_value=[48,6000/200,6000/200]
        max_value=[96,6000/200,6000/200]

        for day_index in days:
        # for day_index in days[20:21]:
            date=datetime.strptime(year+' '+str(day_index+1), '%Y %j')                   

            plotvar=z[day_index,:,:]
            plotvar=np.rot90(plotvar.T)

            # plotvar=np.rot90(plotvar.T)
            plotvar_non_fuzzy =plotvar
            wo=1

            if wo:
                # result=result.filled(fill_value=0)
                ofile='/Users/jason/0_dat/CARRA/output/event/rf/'+varnams[i]+'_'+date.strftime('%d-%m-%Y')+'_'+str(ni)+'x'+str(nj)+'.npy'
                temp=np.array(plotvar)
                temp.astype('float16').tofile(ofile)        # plt.imshow(plotvar)

            if mask_out_ice:
                plotvar[mask==0]=0
   
            plt.close()
            fig, ax = plt.subplots(figsize=(12, 12))

            
            plotvar_non_fuzzy=plotvar
            
            if mask_out_ice:
                plotvar*=mask
                plotvar[mask<0]=0
                plotvar_non_fuzzy[mask<0]=0
            
            areax=2.5e3**2
            mass=np.sum(plotvar[mask>0]*mask[mask>0])/1e9*areax/1000
            print(day_index,date,"{:.1f}".format(mass))

            # if mass>0.5: #rf
            if mass>0: #tp
            
                if mask_out_ice:
                    if i<2:plotvar_non_fuzzy[mask==0]=-1
                
                # plt.imshow(lon)
                ax = plt.subplot(111)
                tit=date.strftime('%d/%m/%Y')+' CARRA '+varnam2[i]+' over Greenland ice'
                ax.set_title(tit)
                
                if i<2:cm=loadct(i)
                # plt.imshow(plotvar_non_fuzzy)
                if map_version==0:
                    pp=plt.imshow(plotvar_non_fuzzy, interpolation='nearest', origin='lower', cmap=cm,vmin=0,vmax=max_value[i]) ; plt.axis('off') 
                
                lon-=360
                v=np.where((lon<-5)&(lat<82))
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
                # print(lat[plotvar==maxval2])
                # print(lon[plotvar==maxval2])
                alllat=lat[plotvar_non_fuzzy==maxval][0]
                alllon=lon[plotvar_non_fuzzy==maxval][0]
                minlat=lat[plotvar_non_fuzzy==minval][0]
                minlon=lon[plotvar_non_fuzzy==minval][0]
                
                # plt.imshow(plotvar)
                if map_version:
                    pp=m.imshow(plotvar, cmap = cm,vmin=0,vmax=max_value[i]) 
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
                    m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=780, facecolors='none', edgecolors='k')
                    # m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='m')
                
                if i<2:
                    cbar_min=0
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
                
                if i==2:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    
                    # plt.colorbar(im)            
                    cbar = plt.colorbar(pp,orientation='vertical',format=cbar_num_format, cax=cax)
                    cbar.ax.set_ylabel(units[i], fontsize = font_size)
                    # tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
                    # # print(tickranges)
                    # cbar.ax.set_yticklabels(tickranges, fontsize=font_size)
                
                cc=0
                xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
                mult=0.7
                color_code='grey'
                plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
                  transform=ax.transAxes,color=color_code) ; cc+=1. 
                
                
                cc=0
                xx0=0.44 ; yy0=0.17 ; dy2=-0.028
                mult=0.8
                color_code='k'
                print()
                print(year)
                plt.text(xx0, yy0+cc*dy2,date.strftime('%Y-%m-%d'), fontsize=font_size*mult,
                  transform=ax.transAxes,color=color_code) ; cc+=1.
            
                msg="{:.1f}".format(mass)+" Gt / day "+varnam2[i]+" total mass flux"
                print(msg)
                plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
                  transform=ax.transAxes,color=color_code) ; cc+=1. 
            
                msg="{:.0f}".format(np.max(plotvar))+" mm / day "+"max "+varnam2[i]+""
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
                
                DPIs=[150,600]
                DPIs=[72]
                
                if ly =='p':
                    # os.system('mkdir -p '+'./Figs/daily/max in sep to oct range/')
                    # figpath='./Figs/daily/max in sep to oct range/'
                    figpath='/Users/jason/0_dat/CARRA/Fig/event/'+varnams[i]+'/'
                    os.system('mkdir -p '+figpath)

                    for DPI in DPIs:
                        figname=figpath+"{:.1f}".format(mass)+'_'+date.strftime('%Y-%m-%d')+'_'+varnams[i]+'_over_ice_'+str(DPI)+'DPI'
                        plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
        make_gif=0
            
        if make_gif:
            os.system('mkdir -p '+'./Figs/daily/anim/')
            animpath='./Figs/daily/anim/'
            inpath='./Figs/daily/'
            msg='convert  -delay 50  -loop 0   '+inpath+'*'+'.png  '+animpath+varnams[i]+'.gif'
            os.system(msg)