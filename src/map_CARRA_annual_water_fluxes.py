# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:04:13 2021

@author: Armin Dachauer
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import os
# from scipy.interpolate import griddata
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
# from shapely.geometry import Point, LineString
import os
# from scipy import stats
# import netCDF4
# from netCDF4 import Dataset,num2date
# import matplotlib.colors as mcolors
# import geopandas as gpd
# from pyproj import Proj, transform
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
# import datetime
import matplotlib.patches as patches
from matplotlib.patches import Polygon 
#from datetime import datetime

AD=0
if AD:
    os.environ['PROJ_LIB'] = r'C:/Users/Armin/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share' #Armin needed to not get an error with import basemap: see https://stackoverflow.com/questions/52295117/basemap-import-error-in-pycharm-keyerror-proj-lib

from mpl_toolkits.basemap import Basemap
import xarray as xr
import cfgrib
# load numpy array from npy file

#-------------------------------------- change path

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
raw_path='./output_annual/'
if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    raw_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'
    
os.chdir(path)

#---------------------------------------- user defined parameters
# ly='x' # output to png, set to 'x' for console
ly='x'

plt_name=1 ; extreme=1000
plt_fit=1
mult=1


ni=1269 ; nj=1069
iyear=1997 # CARRA data dont start until this year
years=np.arange(2019,2020).astype('str') #iyear
months=np.arange(8,12)+1
months = ["%02d" % n for n in months]

Ls = 2834000
Lv = 2501000

# ---------------------------------- graphics settings
fs=8 # font size
fs=16 # font size
ms=80 # marker size
mult=1
th=1
d=0
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "grey"
plt.rcParams["font.size"] = fs
plt.rcParams['ytick.labelsize'] = fs  
plt.rcParams['xtick.labelsize'] = fs
plt.rc('legend',fontsize=fs) 

#%%
#------------------------------------------- Map   
year='2012'
year='1991-2021'
#-------------------------------- get masks
fn=path+'./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)
# lat=np.rot90(lat.T)

fn=path+'./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)
lon=lon.reshape(ni, nj)

# lon[lon<-300]=np.nan
# plt.imshow(lon)
# plt.colorbar()

fn=path+'./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
elev=np.fromfile(fn, dtype=np.float32)
elev=elev.reshape(ni, nj)


# ice mask
fn2=path+'./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = xr.open_dataset(fn2)
# print(nc2.variables)
mask = nc2.z
#mask = nc2.variables['z'][:,:]
mask=np.asarray(mask)
mask_svalbard=1;mask_iceland=1
if mask_iceland:
    mask[((lon-360>-30)&(lat<66.6))]=0
if mask_svalbard:
    mask[((lon-360>-20)&(lat>70))]=0


# ofile='/Users/jason/Dropbox/CARRA/CARRA_rain/ancil/mask_ice_1269x1069.npy'
# mask.astype('float32').tofile(ofile) 
#%%
# ----------------------------------- Map extend settings
LLlat=lat[0,0]
LLlon=lon[0,0]-360
lon0=lon[int(round(ni/2)),int(round(nj/2))]-360
lat0=lat[int(round(ni/2)),int(round(nj/2))]         
URlat=lat[ni-1,nj-1]
URlon=lon[ni-1,nj-1]

res='h' # coastline resolution
if ly=='x':res='h'


#read in data
var_names=['tp','bss', 'swgf', 'rf','rain_frac']
var_names=var_names[-1:]
var_names=['rf']

for ii, var_name in enumerate(var_names): 
    if var_name=='rf':
        fn='/Users/jason/Dropbox/CARRA/CARRA_rain/output_annual/rf_1991-2021_1269x1069_float16.npy'
        y=np.fromfile(fn, dtype=np.float16)
        var_to_total=y.reshape(ni, nj)
        var_to_total=np.rot90(var_to_total.T)
    if var_name=='rain_frac':
        fn='./output_annual/rf_'+year+'_'+str(ni)+'x'+str(nj)+'_float32.npy'
        print(fn)
        # fn='/Users/jason/Dropbox/CARRA/CARRA_rain/output_annual/bss_2019_1269x1069_float32.npy'
        y=np.fromfile(fn, dtype=np.float32)
        y=y.reshape(ni, nj)
        
        fn='./output_annual/tp_'+year+'_'+str(ni)+'x'+str(nj)+'_float32.npy'
        print(fn)
        # fn='/Users/jason/Dropbox/CARRA/CARRA_rain/output_annual/bss_2019_1269x1069_float32.npy'
        x=np.fromfile(fn, dtype=np.float32)
        x=x.reshape(ni, nj)
        var_to_total=y/x

    if var_name=='bss':var_to_total=-var_to_total; bss=var_to_total
    if var_name=='tp': tp=var_to_total
    if var_name=='swgf': swgf=var_to_total
    if var_name=='rf': rf=var_to_total
    # tit=' CARRA annual net surface water gas flux '+year
    tit=''

plt.imshow(var_to_total)
plt.colorbar()
#%% takes 10 sec
m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=lat0, lon_0=lon0, resolution=res, projection='lcc')

lon-=360
#%%
#title names
mass_flux_names=['rain fraction','rainfall','total\nprecipitation', 'blowing snow\nsublimation','surface water\nvapour flux','total precipitation\n+surface water\nvapour flux', 'total precipitation + \n surface water vapour \n flux + blowing snow \n sublimation', 'total net water\nvapour flux', 'snowfall'] #'blowing snow\nsublimation','surface\nwater gas flux'

#what to plot------------
plot_names=['rain_frac','rf','tp','bss', 'swgf','tp_swgf', 'tp_swgf_bss', 'swgf_bss', 'sf']
plot_name=plot_names[1] #define variable to plot
#-------------------------

#write out / calculate data to plot
if plot_name=='tp': var_to_total=tp.copy(); mass_flux_name=mass_flux_names[0]
if plot_name=='bss': var_to_total=bss.copy(); mass_flux_name=mass_flux_names[1]
if plot_name=='swgf': var_to_total=swgf.copy(); mass_flux_name=mass_flux_names[2]
if plot_name=='tp_swgf': var_to_total=tp.copy()+swgf.copy(); mass_flux_name=mass_flux_names[3]
if plot_name=='tp_swgf_bss': var_to_total=tp.copy()+swgf.copy()+bss.copy(); mass_flux_name=mass_flux_names[4]
if plot_name=='swgf_bss': var_to_total=swgf.copy()+bss.copy(); mass_flux_name=mass_flux_names[5]
if plot_name=='sf': var_to_total=tp.copy()-rf.copy(); mass_flux_name=mass_flux_names[6]

if plot_name=='rf': var_to_total=rf.copy(); mass_flux_name=mass_flux_names[1]
if plot_name=='rain_frac': mass_flux_name=mass_flux_names[0]


var_name=plot_name
plotvar_non_fuzzy=var_to_total.copy()

# masking values non GrIS
var_to_total[mask<=0]=0
plotvar_non_fuzzy[mask<=0]=0
plotvar_non_fuzzy = np.ma.masked_where(mask == 0, plotvar_non_fuzzy) #mask 0's so they don't show in map

maxv=lon>-30
temp=plotvar_non_fuzzy*mask
temp[maxv]=np.nan
maxv=temp==np.nanmax(temp)
print('All Greenland','{:.2f}'.format(lat[maxv][0])+'°N','{:.2f}'.format(abs(lon[maxv][0]))+'°W','{:.0f}'.format(elev[maxv][0]),'{:.2f}'.format(temp[maxv][0])+' mm')

maxv2=lon>-44
temp=plotvar_non_fuzzy*mask
temp[maxv]=np.nan
maxv2=temp==np.nanmax(temp)
print('West Greenland','{:.2f}'.format(lat[maxv2][0])+'°N','{:.2f}'.format(abs(lon[maxv2][0]))+'°W','{:.0f}'.format(elev[maxv2][0]),'{:.2f}'.format(temp[maxv2][0])+' mm')

#%%
v=mask>0
areax=2.5**2
ice_area=np.sum(v)*areax
print('area of ice',ice_area)

var_to_total*=mask
v=var_to_total>0.1
print('fractional area above 0.1',(np.sum(v)*areax)/ice_area)

v=var_to_total>0.5
print('fractional area above 0.5',(np.sum(v)*areax)/ice_area)

# plt.imshow(var_to_total)
#%%
# plotvar[mask<=0]=0 #making nonmasks invisible
# plotvar_non_fuzzy[mask==0]=np.nan #making nonmasks invisible


# plot colorbar

# Create the colormap
#colormap for tp, tp+swgf, tp+swgf-bss


if var_name=='rain_frac':
    plotvar_non_fuzzy[plotvar_non_fuzzy>0.99]=np.nan
    plotvar_non_fuzzy[plotvar_non_fuzzy<0.01]=np.nan
    col_bins=4
    bin_bins=0
    off=0
    colors7 = plt.cm.Blues(np.linspace(0.1, 0.9, col_bins))
    colors6 = plt.cm.Greens(np.linspace(0.1, 0.9, col_bins))
    colors5 = plt.cm.BrBG(np.linspace(0.4, 0.1, col_bins)) #browns
    colors4 = plt.cm.Reds(np.linspace(0.1, 0.9, col_bins))
    colors3 = plt.cm.Purples(np.linspace(0.1, 0.9, col_bins)) 
    colors2 = plt.cm.RdPu(np.linspace(0.7, 0.8, 1)) #magenta
    colors1 = plt.cm.autumn(np.linspace(0.9, 1, 1)) #yellow
    # colors1 = plt.cm.autumn(np.linspace(0.7, 0.8, 1)) # peak color
    colors = np.vstack((colors7, colors6, colors5, colors4, colors3, colors2, colors1))
    colors=colors[0:int(len(colors))-off+2]
    n_bin = bin_bins + col_bins*5 +2 - off
    colors=colors[0:int(len(colors))-bin_bins] 
    n_bin=n_bin-bin_bins
    
    bounds=np.arange(0,1,1/n_bin)*0.6
    print(len(bounds),n_bin) # n bounds and n bins must be the same

    # plotvar_non_fuzzy[plotvar_non_fuzzy==0]=np.nan #set 0 to nan so they don't show in map    
    max_value=np.max(bounds)
    min_value=0
    
if var_name=='rf':
    plotvar_non_fuzzy[plotvar_non_fuzzy<30]=np.nan
    col_bins=4
    bin_bins=0
    off=0
    colors7 = plt.cm.Blues(np.linspace(0.1, 0.9, col_bins))
    colors6 = plt.cm.Greens(np.linspace(0.1, 0.9, col_bins))
    colors5 = plt.cm.BrBG(np.linspace(0.4, 0.1, col_bins)) #browns
    colors4= plt.cm.Reds(np.linspace(0.1, 0.9, col_bins))
    colors3 = plt.cm.Purples(np.linspace(0.1, 0.9, col_bins)) 
    colors2 = plt.cm.RdPu(np.linspace(0.7, 0.8, 1)) #magenta
    colors1 = plt.cm.autumn(np.linspace(0.9, 1, 3)) #yellow
    # colors0 = plt.cm.autumn(np.linspace(0.7, 0.8, 2)) # peak color
    colors = np.vstack((colors7, colors6, colors5, colors4, colors3, colors2, colors1))
    colors=colors[0:int(len(colors))-off+2]
    len(colors)
    n_bin = bin_bins + col_bins*5 +2 - off+2
    colors=colors[0:int(len(colors))-bin_bins] 
    n_bin=n_bin-bin_bins
    
    bounds=np.arange(0,730,30)*1.
    len(bounds)
    # plotvar_non_fuzzy[plotvar_non_fuzzy==0]=np.nan #set 0 to nan so they don't show in map    
    max_value=np.max(bounds)
    min_value=0
    
# if var_name=='tp' or var_name=='tp_swgf' or var_name=='tp_swgf_bss' or 'sf':
#     col_bins=4
#     bin_bins=0
#     off=0
#     # colors0 = plt.cm.binary(np.linspace(0.9, 0, bin_bins)) #binary
#     colors7 = plt.cm.Blues(np.linspace(0.1, 0.9, col_bins))
#     colors6 = plt.cm.Greens(np.linspace(0.1, 0.9, col_bins))
#     colors5 = plt.cm.BrBG(np.linspace(0.4, 0.1, col_bins)) #browns
#     colors4= plt.cm.Reds(np.linspace(0.1, 0.9, col_bins))
#     colors3 = plt.cm.Purples(np.linspace(0.1, 0.9, col_bins)) 
#     colors2 = plt.cm.RdPu(np.linspace(0.7, 0.8, 1)) #magenta
#     colors1 = plt.cm.autumn(np.linspace(0.9, 1, 1)) #yellow
#     colors = np.vstack((colors7, colors6, colors5, colors4, colors3, colors2, colors1))
#     colors=colors[0:int(len(colors))-off+2]
#     n_bin = bin_bins + col_bins*5 +2 - off
#     colors=colors[0:int(len(colors))-bin_bins] 
#     n_bin=n_bin-bin_bins
    
#     bounds = [0,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2500,3000,3500,4000, 6000, 8000]

#     # plotvar_non_fuzzy[plotvar_non_fuzzy==0]=np.nan #set 0 to nan so they don't show in map
    
#     max_value=np.max(bounds)
#     min_value=0

# #colormap for swgf, bss and swgf-bss    
# if var_name=='swgf' or var_name=='bss' or var_name=='swgf_bss':
#     max_value=15
#     min_value=-350
#     bounds = [-350, -325,-300,-275,-250,-225,-200,-175,-150,-125,-100,-75,-50,-25,0,1,2,3,4]
#     col_bins=3
#     bin_bins=4
#     off=0
#     colors0 = plt.cm.binary(np.linspace(0, 0.9, bin_bins))
#     colors7 = plt.cm.Blues(np.linspace(0.9, 0.1, col_bins))
#     colors6 = plt.cm.Greens(np.linspace(0.9, 0.1, col_bins))
#     colors5 = plt.cm.BrBG(np.linspace(0.1, 0.4, col_bins)) #browns
#     colors4= plt.cm.Reds(np.linspace(0.9, 0.1, col_bins))
#     # colors3 = plt.cm.Purples(np.linspace(0.9, 0.1, col_bins))  
#     colors2 = plt.cm.RdPu(np.linspace(0.8, 0.7, 1)) #magenta
#     colors1 = plt.cm.autumn(np.linspace(0.9, 1, 1)) #yellow
#     colors = np.vstack((colors1, colors2, colors4, colors5, colors6, colors7, colors0))
#     colors=colors[off:int(len(colors))]
#     n_bin = bin_bins + col_bins*4 - off +2
    
#     if var_name=='bss' or var_name=='swgf_bss':
#         max_value=0
#         colors=colors[0:int(len(colors))-bin_bins] 
#         bounds = [-350, -325,-300,-275,-250,-225,-200,-175,-150,-125,-100,-75,-50,-25,-0]
#         n_bin=n_bin-bin_bins
#         if var_name=='swgf_bss':
#             bounds = [-1000, -800, -600, -550, -500, -450, -400, -350, -300,-250,-200,-150,-100,-50,0,]
#             min_value=-1000

#create colormap
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
cm.set_bad(color='white') #set color of zeros white
norm = BoundaryNorm(bounds, cm.N)


#plot
# fig = plt.figure()
fig = plt.figure(figsize=(9,11))

ax = plt.subplot(111)
ax.set_title(tit, fontsize=fs)
ax = plt.gca()    
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

fig.tight_layout()

if ((var_name=='bss') or (var_name=='swgf')):
    plotvar_non_fuzzy[plotvar_non_fuzzy<-325]=-325

#plot map
coastline_width=0.7
pp = m.imshow(plotvar_non_fuzzy, cmap =cm , norm=norm) 
# lonx=lon.copy()
# lonx-=360
# lons, lats = m(lonx, lat)
# m.contour(lons, lats,plotvar_non_fuzzy,[0.1],linewidths=2,colors='grey',zorder=30)

m.drawcoastlines(color='k',linewidth=coastline_width)
# parallels = np.arange(59,83,1.)
# m.drawparallels(parallels,labels=[True,True,True,True])
# meridians = np.arange(10.,351.,2.)
# m.drawmeridians(meridians,labels=[True,True,True,True])


lons, lats = m(lon, lat)
m.scatter(lons[maxv],lats[maxv], s=400, facecolors='none', edgecolors='grey',linewidths=th*2)

m.scatter(lons[maxv2],lats[maxv2], s=400, facecolors='none', edgecolors='grey',linewidths=th*2)


# --------------------- mask iceland and svalbard with a rotated rectangle
rect=patches.Rectangle(
        xy=(0.665, 0.1),  # point of origin.
        width=0.29,
        height=1,
        # linewidth=1,
        color='w',
        transform=ax.transAxes,zorder=9)
t_start = ax.transAxes ; t = mpl.transforms.Affine2D().rotate_deg(-4) ; t_end = t_start + t
rect.set_transform(t_end)
ax.add_patch(rect)
# --------------------- mask Canada with a polygon
X = np.array([[0,0.7], [0.39,0.9], [0.39, 1], [0.0, 1]])
ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
        transform=ax.transAxes,zorder=9)) 
# ---------------------

# --------------------- colorbar location
if var_name=='bss' or var_name=='swgf_bss': 
    cbaxes = fig.add_axes([0.62, 0.15, 0.018, 0.474]) 
else: 
    cbaxes = fig.add_axes([0.7, 0.15, 0.018, 0.61]) 
cbar = plt.colorbar(pp,orientation='vertical',cax=cbaxes, ticks=bounds)
unitsx='mm w e'
if var_name=='rain_frac':
    cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in bounds])
    unitsx=''
    
    # cbar_num_format='% .2f'

cbar.outline.set_linewidth(.4)  
cbar.ax.tick_params(width=.4)
# cbar.ax.set_yticklabels(bounds, fontsize=fs)
# cbar.set_ticks(bounds)

#--------------plot text
#units
mult=1
xx0=0.7 ; yy0=0.79 ; dy2=-0.028 ; cc=0
xx0=0.72 ; yy0=0.79 ; dy2=-0.028 ; cc=0
if var_name=='bss' or var_name=='swgf_bss': yy0=0.65
plt.text(xx0, yy0+0.03,unitsx, fontsize=fs,
          transform=ax.transAxes, color='k') ; cc+=1. 

# title
cc=0
xx0=0.77 ; yy0=0.84 ; dy2=-0.08 
if var_name=='tp_swgf_bss': xx0=0.75; yy0=0.83  #since text is larger
if var_name=='tp' or var_name=='tp_swgf' or var_name=='tp_swgf_bss': 
    mult=1
    plt.text(xx0, yy0+cc*dy2,mass_flux_name, fontsize=fs*mult,ha='center',
                  transform=ax.transAxes, color='k') ; cc+=1.  #smaller fontsize
else: 
    mult=1.2
    plt.text(xx0, yy0+0.015,year+'\n'+mass_flux_name, fontsize=fs*mult,ha='center',
                  transform=ax.transAxes, color='k') ; cc+=1. #larger fontsize
    
#flux
mult=0.9
cc=0
xx0=0.43 ; yy0=0.12 ; dy2=-0.08
areax=2.5e3**2
mass=np.sum(var_to_total[mask>0]*mask[mask>0])/1e9*areax/1000 #*mask[mask>0] makes it fuzzy... 
mult*=1
msg="{:.0f}".format(mass)+" Gt y$^{-1}$"
# print(msg)
# plt.text(xx0, yy0+cc*dy2,year+' total:\n'+msg+'\nBox, Nielsen, Dachauer', fontsize=fs*mult,
#                   transform=ax.transAxes, color='k') ; cc+=1. 


# ----------  Crop Image

ly='p'
ly='x'

from PIL import Image

if ly =='p':
    # figpath=
    # os.system('mkdir -p '+figpath)
    # figname=figpath+str(sum_tp)+dates
    DPI = 300
    # figname=figpath+year+'_map_annual_swgf_'+str(DPI)+'DPI'
    # plt.savefig('/tmp/t.png', bbox_inches='tight', dpi=DPI)
    plt.savefig('./Figs/swvf/t.png', bbox_inches='tight', dpi=DPI)
  # os.system('open /tmp/t.png')
    
    im1 = Image.open('./Figs/swvf/t.png', 'r')
    width, height = im1.size
    border=90
    # Setting the points for cropped image
    left = 440
    top = 240
    right = width-320
    bottom = height-210
     
    # Cropped image of above dimension
    im1 = im1.crop((left, top, right, bottom))
    out_im = im1.copy()
    
    figpath='./Figs/'
    fn=figpath+year+'_'+var_name+'.png'
    out_im.save(fn)  
    os.system('open '+fn)

    # ------------------------------ Q lobe
    out_im2 = out_im.copy()
    border=90
    # Setting the points for cropped image
    width, height = out_im2.size
    left = 120
    top = 1900
    right = width-1150
    bottom = height-40
     
    # # Cropped image of above dimension
    out_im2 = out_im2.crop((left, top, right, bottom))
    fn=figpath+year+'_'+var_name+'_Q.png'
    out_im2.save(fn)  
    # os.system('open '+fn)

    # ------------------------------ NW
    out_im3 = out_im.copy()
    # Setting the points for cropped image
    width, height = out_im3.size
    left = 60
    top = 250
    right = width-1260
    bottom = height-1730
     
    # # Cropped image of above dimension
    out_im3 = out_im3.crop((left, top, right, bottom))
    fn=figpath+year+'_'+var_name+'_NW.png'
    out_im3.save(fn)  
    # os.system('open '+fn)
    
# if ly =='p':
#     figpath=path+'Figs/swvf/'
#     os.system('mkdir -p '+figpath)
#     # figname=figpath+str(sum_tp)+dates
#     DPI = 250
#     figname=
#     plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)
#%% 
make_gif=0
# nam='2017-09-14_speed_orthographic'
nam=var_names[2]+str(DPI)+'DPI'
if make_gif:
    print("making gif")
    # animpath='/Users/jason/Dropbox/ERA5/anim/'+event+'_'+var+'/'
    if AD: 
        animpath = path+'Figs/'
        frames = []
        imgs = []
        for name in plot_names:
            img_path = path+'Figs/'+year+'_'+name+".png"
            new_frame = Image.open(img_path)
            frames.append(new_frame)
        frames[0].save(animpath+year+'_'+plot_names[2]+'.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,   
               duration=800, loop=0)
        
    os.system('mkdir -p '+animpath)
    inpath=figpath
    # msg='convert  -delay 8  -loop 0   '+inpath+'*.png  '+animpath'_'+nam+'.gif'
    # os.system(msg)
