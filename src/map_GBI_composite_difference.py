#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes and outputs .npy statistics files and a map of rainfall trending

updated Nov 2022
@author: Jason Box, GEUS, jeb@geus.dk

"""


from netCDF4 import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy import stats
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Polygon 
from matplotlib.pyplot import figure


# DPI=100
# figure(num=None, figsize=(12, 12), dpi=DPI, facecolor='w', edgecolor='k')

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
os.chdir(path)

# read ice mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = Dataset(fn, mode='r')
# print(nc2.variables)
mask = nc2.variables['z'][:,:]
mask=np.rot90(mask.T)

# plt.imshow(mask)

ni=1269 ; nj=1069

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)
lat=np.rot90(lat.T)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)
lon=np.rot90(lon.T)

mask_iceland=0
mask_svalbard=0

if mask_iceland:
    mask[((lon-360>-30)&(lat<66.6))]=0
if mask_svalbard:
    mask[((lon-360>-20)&(lat>70))]=0
plt.imshow(mask)
#%%
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size


# iyear=1991 ; fyear=2021
# n_years=fyear-iyear+1

# years=np.arange(iyear,fyear+1).astype('str')



# for i,varnam in enumerate(variables):
#     if i==2:
varnam='rf'
varnam='tp'
if varnam=='rf':
    varnam2='rainfall'
    maxval=1000
if varnam=='tp':
    varnam2='total precip.'
    maxval=6000
# confidence=np.zeros(((12,nj,ni)))


#%% map setup

print('map setup')
ly='p'

res='l'
if ly=='p':res='h'

# global plot settings
th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
mask_svalbard=1;mask_iceland=1;mask_jan_mayen=1


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
# x, y = m(lat, lon)
x, y = m(lon, lat)

#%% map trend
varnam='tp'
th=1
fn='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/output/tp_average_GBI-_1269x1069.npy'
a=np.fromfile(fn, dtype=np.float16)
a=a.reshape(ni, nj)
fn='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/output/tp_average_GBI+_1269x1069.npy'
b=np.fromfile(fn, dtype=np.float16)
b=b.reshape(ni, nj)

# slope=np.rot90(slope.T)
plotvar=b-a
plt.imshow(plotvar)
plt.axis('off')
plt.colorbar()

plotvar_non_fuzzy=np.rot90(plotvar.T)
# plotvar_non_fuzzy[mask<0]=0


areax=2.5e3**2
mass=np.nansum(plotvar[mask>0]*mask[mask>0])/1e9*areax/1000

# mass2=np.nansum(meanx[mask>0]*mask[mask>0])/1e9*areax/1000

# plotvar_non_fuzzy[mask==0]=np.nan

plt.close()
fig, ax = plt.subplots(figsize=(12, 12))

do_regional_min_max=1

if do_regional_min_max:
    lonx=lon.copy()
    lonx-=360
    # v=np.where((lonx<-45)&(lat<62))
    maxval=np.nanmax(plotvar_non_fuzzy)
    minval=np.nanmin(plotvar_non_fuzzy)
    # maxval2=np.nanmax(plotvar_non_fuzzy[v])
    # print("position of extremes")
    # print(np.max(plotvar_non_fuzzy))
    # print(maxval)
    # print(lat[plotvar==maxval2])
    # print(lon[plotvar==maxval2])
    # SSWlat=lat[plotvar_non_fuzzy==maxval2][0]
    # SSWlon=lon[plotvar_non_fuzzy==maxval2][0]
    # SSWelev_with_max=elev[plotvar_non_fuzzy==maxval2][0]
    # print(lat[plotvar==maxval2])
    # print(lon[plotvar==maxval2])
    alllat=lat[plotvar_non_fuzzy==maxval][0]
    alllon=lonx[plotvar_non_fuzzy==maxval][0]
    minlat=lat[plotvar_non_fuzzy==minval][0]
    minlon=lonx[plotvar_non_fuzzy==minval][0]
    elev_with_max=elev[plotvar_non_fuzzy==maxval][0]

# plot_only_background=1


# if plot_only_background==0:
# pp=m.imshow(np.rot90(plotvar.T), cmap = 'BrBG', vmin=lo,vmax=-lo) 

dx=20
lo=-300
hi=-lo
clevs=np.arange(lo,hi,dx)
z=np.rot90(plotvar.T)
cm = plt.cm.BrBG
cm.set_over('b')
cm.set_under('r')
pp=m.contourf(x,y,z,clevs,cmap=cm, extend='both')

xx=m.contour(x,y,z,clevs=[0],colors='grey')



# if plot_only_background==0:
m.drawcoastlines(color='k',linewidth=0.5)
# if plot_only_background:m.drawparallels([66.6],color='gray')
# if plot_only_background:m.drawparallels([60,70,80,83],dashes=[2,4],color='k')
# m.drawmeridians(np.arange(0.,420.,10.))
# if plot_only_background:m.drawmeridians([-68,-62,-56,-50,-44,-38,-32,-26,-20])
# m.drawmapboundary(fill_color='aqua')
ax = plt.gca()     
if do_regional_min_max:
    lons, lats = m(lonx, lat)
    m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=400, facecolors='none', edgecolors='k',linewidths=th*2,zorder=29)
    m.scatter(lons[plotvar_non_fuzzy==maxval],lats[plotvar_non_fuzzy==maxval], s=380, facecolors='none', edgecolors='m',linewidths=th*1,zorder=30)
    # m.scatter(lons[plotvar_non_fuzzy==minval],lats[plotvar_non_fuzzy==minval], s=380, facecolors='none', edgecolors='m',zorder=30)

tight_layout=0

if tight_layout:
    xx0=0.695
    yy0x=0.18
    dyx=0.4
    
    # --------------------- mask Iceland Sval with a polygon
    X = np.array([[xx0,0], [xx0,1], [1, 1], [1, 0],[xx0,0]])
    ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
            transform=ax.transAxes,zorder=9)) 
    # --------------------- mask Canada with a polygon
    # X = np.array([[0,0.7], [0.39,0.9], [0.39, 1], [0.0, 1]])
    # ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
    #         transform=ax.transAxes,zorder=9)) 
    # ---------------------    
    # --------------------- colorbar location
    cbaxes = fig.add_axes([xx0-0.04, yy0x, 0.015, dyx]) 
    cbar = plt.colorbar(pp,orientation='vertical',format="%d",cax=cbaxes)
    # cbar = plt.colorbar(pp,orientation='vertical')
    #units
    mult=1
    yy0=yy0x+0.45 ; dy2=-0.03 ; cc=0
    units2='mm y$^{-1}$'
    units2='mm'
    plt.text(xx0+0.01, yy0+cc*dy2,units2, fontsize=font_size*mult,
             transform=ax.transAxes, color='k') ; cc+=1. 

if tight_layout==0:
    xx0=1
    yy0x=0.13
    dyx=0.5

    # --------------------- colorbar location
    cbaxes = fig.add_axes([xx0-0.15, yy0x, 0.015, dyx]) 
    cbar = plt.colorbar(pp,orientation='vertical',format="%d",cax=cbaxes)
    # cbar = plt.colorbar(pp,orientation='vertical')
    #units
    mult=1
    # yy0=yy0x+0.45 ; dy2=-0.03 ; cc=0
    plt.text(xx0+0.01, 0.69,'mm y$^{-1}$', fontsize=font_size*mult,
             transform=ax.transAxes, color='k') 
# cbar.ax.minorticks_off()

# cbar.ax.tick_params(length=0)

# from matplotlib.ticker import LogFormatter 
# formatter = LogFormatter(10, labelOnlyBase=False) 

# cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes, ticks=bounds, format=formatter)
# ticks=[1,5,10,20,50]
# cbar.outline.set_linewidth(.4)  
# cbar.ax.tick_params(width=.4)
# mult=0.6
# cbar.ax.set_yticklabels(bounds, fontsize=font_size*mult)



# mult=0.8
# plt.text(0.65,0.8,'rainfall change\n1991 to 2021', fontsize=font_size*mult,
#          transform=ax.transAxes, color='k') ; cc+=1. 

# plt.colorbar()
# cc=0
# xx0=0.0 ; yy0=-0.02 ; dy2=-0.04
# mult=0.7
# color_code='grey'
# plt.text(xx0, yy0+cc*dy2,'Box, Nielsen and the CARRA team', fontsize=font_size*mult,
#   transform=ax.transAxes,color=color_code) ; cc+=1. 

annotatex=1

if annotatex:
    cc=0
    xx0=1.02 ; yy0=0.98 ; dy2=-0.035

    if tight_layout:
        xx0=0.43 ; yy0=0.21 ; dy2=-0.035
    mult=0.95
    color_code='k'
    print()
    plt.text(xx0, yy0+cc*dy2,'CARRA', fontsize=font_size*mult,
             transform=ax.transAxes,color=color_code) ; cc+=1.
    plt.text(xx0, yy0+cc*dy2,varnam2, fontsize=font_size*mult,
             transform=ax.transAxes,color=color_code) ; cc+=1.   
    plt.text(xx0, yy0+cc*dy2,'GBI+ minus GBI-', fontsize=font_size*mult,
             transform=ax.transAxes,color=color_code) ; cc+=1.
    plt.text(xx0, yy0+cc*dy2,'Greenland mass flux', fontsize=font_size*mult,
             transform=ax.transAxes,color=color_code) ; cc+=1.
    plt.text(xx0, yy0+cc*dy2,'difference:', fontsize=font_size*mult,
             transform=ax.transAxes,color=color_code) ; cc+=1.
    
    units=" Gt y$^{-1}$"
    units=" Gt"
    
    msg="{:.0f}".format(mass)+units
    print(msg)
    plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
      transform=ax.transAxes,color=color_code) ; cc+=0.9

    # msg="change: +{:.0f}".format(((mass/mass2)*100))+" %"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
      # transform=ax.transAxes,color=color_code) ; cc+=1. 
    
    # msg="overall max {:.0f}".format(np.max(plotvar))+" mm y $^{-1}$ "+"max rainfall"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 
    
    # msg="{:.4f}".format(alllat)+"°N, "+"{:.4f}".format(abs(alllon))+"°W, {:.0f}".format(elev_with_max)+' m'
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=2. 

    # msg="Qagssimiut lobe max {:.0f}".format(maxval2)+" mm y $^{-1}$ "+"max rainfall"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

    # msg="{:.4f}".format(SSWlat)+"°N, "+"{:.4f}".format(abs(SSWlon))+"°W, {:.0f}".format(SSWelev_with_max)+' m'
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 
    
    # msg="{:.0f}".format(maxval2)+" {:.4f}".format(SSWlat)+"°N, "+"{:.4f}".format(abs(SSWlon))+"°W, {:.0f}".format(SSWelev_with_max)+' m'
    # print(msg)
    # msg="{:.0f}".format(np.max(plotvar[v]))+" mm y $^{-1}$ "+"max rainfall SSW"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

    # msg="{:.4f}".format(SSWlat)+" N, "+"{:.4f}".format(abs(SSWlon))+" W"
    # print(msg)
    # plt.text(xx0, yy0+cc*dy2,msg, fontsize=font_size*mult,
    #   transform=ax.transAxes,color=color_code) ; cc+=1. 

ly='p'

DPI=300

if tight_layout:
    # ----------  Crop Image
    from PIL import Image
    
    if ly =='p':
        # figpath=
        # os.system('mkdir -p '+figpath)
        # figname=figpath+str(sum_tp)+dates
        DPI = 300
        # figname=figpath+year+'_map_annual_swgf_'+str(DPI)+'DPI'
        # plt.savefig('/tmp/t.png', bbox_inches='tight', dpi=DPI)
        plt.savefig('/tmp/t.png', bbox_inches='tight', dpi=DPI)
      # os.system('open /tmp/t.png')
        
        im1 = Image.open('/tmp/t.png', 'r')
        width, height = im1.size
        border=90
        # Setting the points for cropped image
        left = 440
        top = 240
        right = width-400
        bottom = height-210
         
        # Cropped image of above dimension
        im1 = im1.crop((left, top, right, bottom))
        out_im = im1.copy()
        
        figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'
        fn=figpath+varnam+'_GBI_composite_GBI+_minus_GBI-.png'
        out_im.save(fn)  
        os.system('open '+fn)
else:
        figpath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/'
        figname=figpath+varnam+'_GBI_composite_GBI+_minus_GBI-.png'
        plt.savefig(figname, bbox_inches='tight', dpi=DPI)
        # os.system('open '+figname)
        
if ly == 'x':
    plt.show()

# #%%
# DPIs=[150,300]
# DPIs=[400]

# if ly =='p':
#     for DPI in DPIs:
#         figpath='./Figs/annual/'+varnams[i]+'/'
#         figpath='/Users/jason/Dropbox/CARRA/CARRA rainfall study/Figs/'
#         # os.system('mkdir -p '+figpath)
#         # figpath='./Figs/annual/'+varnams[i]+'/'+str(DPI)+'/'
#         # os.system('mkdir -p '+figpath)
#         figname=figpath+varnam+'_map_'+str(iyear)+'-'+str(fyear)+'_'+str(DPI)
#         # if i<2:
#         if plot_only_background:
#             plt.savefig(figname+'.svg', bbox_inches='tight', dpi=DPI)
#         else:
#             plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
#         # else:
#                 # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)#, facecolor=fig.get_facecolor(), edgecolor='none')
