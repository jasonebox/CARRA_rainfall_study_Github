#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
makes multi-annual rf map

updated Nov 2022
@author: Jason Box, GEUS, jeb@geus.dk
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
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Polygon 


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

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
# mask = np.rot90(mask.T)
# plt.imshow(mask)

# rf is raingall, tp is total precipitation
varnams=['rf','tp','t2m']



ni=1269 ; nj=1069

# for i in range(3):
for i,varnam in enumerate(varnams):
    if i==0:
        plt.rcParams['axes.facecolor']='w'
        plt.rcParams['savefig.facecolor']='w'
  
    
        fn='/Users/jason/Dropbox/CARRA/CARRA_rain/output_annual/'+varnam+'_1991-2021_1269x1069_float16.npy'
        os.system('ls -lF '+fn)

        plotvar=np.fromfile(fn, dtype=np.float16)
        plotvar=plotvar.reshape(ni, nj)
        plotvar=np.rot90(plotvar.T)

        # plt.imshow(plotvar)
        plotvar_non_fuzzy =plotvar
        varnam2=['rainfall','precip.',r'$(t2m_{max} + t2m_{min})/2$']
        min_value=[0,0,-30]
        max_value=[720,6000,4]
        units=['mm','mm','deg C']
         
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
        
        # def loadct(i):
        #     loval=188
        #     if i==0:loval=255
        #     print("loval",loval)
        #     r=[188,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156, 255, 255]
        #     g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28, 255, 255 ]
        #     b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196, 0, 255 ]
        #     r=[loval,108,76,0,      172,92,0,0,      255,255,255,220, 204,172,140,108, 255,255,255,236, 212,188,164,156]
        #     g=[255,255,188,124, 255,255,220,156, 255,188,156,124,  156,124,92,60,   188,140,72,0,    148,124,68,28 ]
        #     b=[255,255,255,255,  172,92,0,0,      172,60,0,0,       156,124,92,60,   220,196,164,0,  255,255,255,196]
        #     colors = np.array([r, g, b]).T / 255
        #     n_bin = 24
        #     cmap_name = 'my_list'
        #     # Create the colormap
        #     cm = LinearSegmentedColormap.from_list(
        #         cmap_name, colors, N=n_bin)
        #     cm.set_under('w') #  brown, land
        #     # my_cmap.set_over('#E4EEF8')
        #     return cm
        
        # if i==2:
        #     cm='jet'
        
        
        
        col_bins=4
        bin_bins=0
        off=0
        # colors0 = plt.cm.binary(np.linspace(0.9, 0, bin_bins)) #binary
        colors7 = plt.cm.Blues(np.linspace(0.1, 0.9, col_bins))
        colors6 = plt.cm.Greens(np.linspace(0.1, 0.9, col_bins))
        colors5 = plt.cm.BrBG(np.linspace(0.4, 0.1, col_bins)) #browns
        colors4= plt.cm.Reds(np.linspace(0.1, 0.9, col_bins))
        colors3 = plt.cm.Purples(np.linspace(0.1, 0.9, col_bins)) 
        colors2 = plt.cm.RdPu(np.linspace(0.7, 0.8, 1)) #magenta
        colors1 = plt.cm.autumn(np.linspace(0.9, 1, 1)) #yellow
        colors = np.vstack((colors7, colors6, colors5, colors4, colors3, colors2, colors1))
        colors=colors[0:int(len(colors))-off+2]
        n_bin = bin_bins + col_bins*5 +2 - off
        colors=colors[0:int(len(colors))-bin_bins] 
        n_bin=n_bin-bin_bins
        
        bounds = [0,1,2,3,4,6,8,10,14,24,30,40,55,70,90,120,160,200,265,350]
        
        max_value=np.max(bounds)
        min_value=0

        #create colormap
        cmap_name = 'my_list'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
        cm.set_bad(color='white') #set color of zeros white
        norm = BoundaryNorm(bounds, cm.N)
        cbar_num_format = "%d"

# # N=len(bounds)
# # bounds=np.logspace(0.1, 1, N, endpoint=True)*35-35-8
# # print(bounds)

# # plotvar_non_fuzzy[plotvar_non_fuzzy==0]=np.nan #set 0 to nan so they don't show in map

# max_value=np.max(bounds)
# min_value=0

# #create colormap
# cmap_name = 'my_list'
# cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
# cm.set_bad(color='white') #set color of zeros white
# norm = BoundaryNorm(bounds, cm.N)
# cbar_num_format = "%d"
#         if i<3:
#             if mask_iceland:
#                 mask[((lon-360>-30)&(lat<66.6))]=0
#             if mask_svalbard:
#                 mask[((lon-360>-20)&(lat>70))]=0
# 					# if mask_jan_mayen:
# 					#     mask[((lon-360>-15)&(lat>66.6)&(lat<75))]=0
# 					# if mask_iceland:
# 					#     mask[((lon-360>-30)&(lat<66.6))]=0
# 					# if mask_svalbard:
# 					#     mask[0:300,800:]=0
# 					#                     plotvar[mask==0]=0
        
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
        tit=' CARRA '+varnam2[i]+' over Greenland ice'
        # ax.set_title(tit)
        
        # if i<2:cm=loadct(i)
        
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
            SSWelev_with_max=elev[plotvar_non_fuzzy==maxval2][0]
            # print(lat[plotvar==maxval2])
            # print(lon[plotvar==maxval2])
            alllat=lat[plotvar_non_fuzzy==maxval][0]
            alllon=lon[plotvar_non_fuzzy==maxval][0]
            minlat=lat[plotvar_non_fuzzy==minval][0]
            minlon=lon[plotvar_non_fuzzy==minval][0]
            elev_with_max=elev[plotvar_non_fuzzy==maxval][0]
        
        plot_only_background=1
        
        if map_version:
            # pp=m.imshow(plotvar, cmap = cm,vmin=min_value[i],vmax=max_value[i]) 
            if plot_only_background==0:pp=m.imshow(plotvar, cmap = cm, norm=LogNorm(vmin=1,vmax=bounds[-1:][0])) 

            # m.axis('off')
            if plot_only_background==0:m.drawcoastlines(color='k',linewidth=0.5)
            if plot_only_background:m.drawparallels([66.6],color='gray')
            if plot_only_background:m.drawparallels([60,70,80,83],dashes=[2,4],color='k')
            # m.drawmeridians(np.arange(0.,420.,10.))
            if plot_only_background:m.drawmeridians([-68,-62,-56,-50,-44,-38,-32,-26,-20])
            # m.drawmapboundary(fill_color='aqua')
            ax = plt.gca()     
            # plt.title("Lambert Conformal Projection")
            # plt.show()
            lons, lats = m(lon, lat)
            # m.scatter(lons[plotvar_non_fuzzy==maxval2],lats[plotvar_non_fuzzy==maxval2], s=380, facecolors='none', edgecolors='m')
            # m.scatter(lons[plotvar_non_fuzzy==maxval2][0],lats[plotvar_non_fuzzy==maxval2][0], s=400, facecolors='none', edgecolors='k',linewidths=th*2)
            # m.scatter(lons[plotvar_non_fuzzy==maxval2][0],lats[plotvar_non_fuzzy==maxval2][0], s=380, facecolors='none', edgecolors='m',linewidths=th*1)
            
            # m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=400, facecolors='none', edgecolors='k',linewidths=th*2)
            # m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=380, facecolors='none', edgecolors='m',linewidths=th*1)
            # m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='m')
        
        # if i<2: # rf or tp
        #     cbar_min=min_value[i]
        #     cbar_max=max_value[i]
        #     cbar_step=max_value[i]/24
        #     cbar_num_format = "%d"
        
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
            
        #     # plt.colorbar(im)            
        #     cbar = plt.colorbar(pp,
        #                         orientation='vertical',
        #                         ticks=np.arange(cbar_min,
        #                         cbar_max+cbar_step, cbar_step),format=cbar_num_format, cax=cax)
        #     cbar.ax.set_ylabel(units[i], fontsize = font_size)
        #     tickranges=np.arange(cbar_min, cbar_max+cbar_step, cbar_step).astype(int)
        #     # print(tickranges)
        #     cbar.ax.set_yticklabels(tickranges, fontsize=font_size)

        # --------------------- mask Iceland Sval with a polygon
        xx0=0.7
        X = np.array([[xx0,0], [xx0,1], [1, 1], [1, 0],[xx0,0]])
        ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
                transform=ax.transAxes,zorder=9)) 
        # --------------------- mask Canada with a polygon
        # X = np.array([[0,0.7], [0.39,0.9], [0.39, 1], [0.0, 1]])
        # ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
        #         transform=ax.transAxes,zorder=9)) 
        # ---------------------

        if plot_only_background:
            yy0x=0.35
            # --------------------- colorbar location
            cbaxes = fig.add_axes([xx0-0.04, 0.16, 0.01, yy0x]) 
            cbar = plt.colorbar(pp,orientation='vertical',format="%d",cax=cbaxes, ticks=bounds)
            cbar.ax.minorticks_off()
    
            # cbar.ax.tick_params(length=0)
    
            # from matplotlib.ticker import LogFormatter 
            # formatter = LogFormatter(10, labelOnlyBase=False) 
    
            # cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes, ticks=bounds, format=formatter)
            # ticks=[1,5,10,20,50]
            # cbar.outline.set_linewidth(.4)  
            # cbar.ax.tick_params(width=.4)
            mult=0.6
            cbar.ax.set_yticklabels(bounds, fontsize=font_size*mult)
            
            #units
            mult=0.7
            yy0=yy0x+0.18 ; dy2=-0.03 ; cc=0
            plt.text(xx0+0.02, yy0+cc*dy2,'mm', fontsize=font_size*mult,
                     transform=ax.transAxes, color='k') ; cc+=1. 

        
        # cc=0
        # xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
        # mult=0.7
        # color_code='grey'
        # plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
        #   transform=ax.transAxes,color=color_code) ; cc+=1. 
        
        annotatex=1
        
        if annotatex:
            cc=0
            xx0=0.41 ; yy0=0.18 ; dy2=-0.024
            mult=0.72
            color_code='k'
            print()
            plt.text(xx0, yy0+cc*dy2,'1991-2021', fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1.
        
            msg="{:.0f}".format(mass)+" Gty $^{-1}$ "+varnam2[i]+" total mass flux"
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1. 
        
            msg="overall max {:.0f}".format(np.max(plotvar))+" mm y $^{-1}$ "+"max "+varnam2[i]+""
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1. 
        
            msg="{:.4f}".format(alllat)+"°N, "+"{:.4f}".format(abs(alllon))+"°W, {:.0f}".format(elev_with_max)+' m'
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=2. 

            msg="Qagssimiut lobe max {:.0f}".format(maxval2)+" mm y $^{-1}$ "+"max "+varnam2[i]+""
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1. 
        
            msg="{:.4f}".format(SSWlat)+"°N, "+"{:.4f}".format(abs(SSWlon))+"°W, {:.0f}".format(SSWelev_with_max)+' m'
            print(msg)
            plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
              transform=ax.transAxes,color=color_code) ; cc+=1. 
            
            msg="{:.0f}".format(maxval2)+" {:.4f}".format(SSWlat)+"°N, "+"{:.4f}".format(abs(SSWlon))+"°W, {:.0f}".format(SSWelev_with_max)+' m'
            print(msg)
            # msg="{:.0f}".format(np.max(plotvar[v]))+" mm y $^{-1}$ "+"max "+varnam2[i]+" SSW"
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
        DPIs=[400]
        
        if ly =='p':
            for DPI in DPIs:
                figpath='./Figs/annual/'+varnams[i]+'/'
                figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'
                # os.system('mkdir -p '+figpath)
                # figpath='./Figs/annual/'+varnams[i]+'/'+str(DPI)+'/'
                # os.system('mkdir -p '+figpath)
                figname=figpath+varnam+'_map_1991-2021_'+str(DPI)
                # if i<2:
                if plot_only_background:
                    plt.savefig(figname+'.svg', bbox_inches='tight', dpi=DPI)
                else:
                    plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
                # else:
                        # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)#, facecolor=fig.get_facecolor(), edgecolor='none')
#             #%%
# make_gif=1

# if make_gif:
#     animpath='/Users/jason/Dropbox/CARRA/anim/'
#     # inpath='/Users/jason/Dropbox/CARRA/Figs/'
#     inpath=figpath
#     msg='convert  -delay 70  -loop 0   '+inpath+'*'+varnams[i]+'*'+'.png  '+animpath+varnams[i]+'_1991-2021.gif'
#     os.system(msg)