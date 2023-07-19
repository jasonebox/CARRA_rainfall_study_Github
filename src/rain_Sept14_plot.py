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


# # read ice mask
# fn='./ancil/CARRA_W_domain_ice_mask.nc'
# nc2 = Dataset(fn, mode='r')
# # print(nc2.variables)
# mask = nc2.variables['z'][:,:]
# # mask=np.rot90(mask.T)

# fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
# lat=np.fromfile(fn, dtype=np.float32)
# lat=lat.reshape(ni, nj)
# # lat=np.rot90(lat.T)

# fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
# lon=np.fromfile(fn, dtype=np.float32)
# lon=lon.reshape(ni, nj)
# # lon=np.rot90(lon.T)

# mask_iceland=1
# mask_svalbard=1

# if mask_iceland:
#     mask[((lon-360>-30)&(lat<66.6))]=0
# if mask_svalbard:
#     mask[((lon-360>-10)&(lat>70))]=0

# v=np.where(mask>0)
# ice_area=len(v[0])*2.5**2
# print('ice area ',ice_area)

do_plot=1
areax=2.5e3**2

do_cum=0

fn='/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvw_1000-300hPa_14_levs.grib'

var='w'

def get_UVW(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    # print(ds.variables)
    P=np.array(ds.variables['isobaricInhPa'].values)
    U=ds.variables['u'].values
    V=ds.variables['v'].values
    W=ds.variables['wz'].values
    # T=ds.variables['t'].values-273.15
    time=ds.variables['time'].values
    # P=P[0]
    return U,V,W,P,time

Ux,Vx,Wx,P,times=get_UVW(fn)

def get_CARRA(fn):
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    tirf=ds.variables['tirf'].values
    time=ds.variables['time'].values
    return tirf,time

tirf,times=get_CARRA('/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_rfsf.grib')

#%%

#%%
subset_it=1
subset_name=''

if subset_it:
    elev0=50;elev1=3000;delev=200
    # 50,3000,200
    max_value=24
    max_value=36
    subset_name='_subset'
    # lon=np.rot90(lon.T)
    # lat=np.rot90(lat.T)
    xc0=200 ; xc1=450 # 250
    yc0=950 ; yc1=1200 # 250
    xc0=210 ; xc1=400
    yc0=1000 ; yc1=1210

from matplotlib.colors import LogNorm
# from pylab import figure, cm
from matplotlib import ticker, cm

rain_tot=np.zeros((ni,nj))


for hh,time in enumerate(times):
    # print(yy+iyear,day)
    # if i==1:
    # if i<=3:
    timex=pd.to_datetime(time)
    # if hh>=0:
    if hh==5: # 15h
    # if hh==6: # 18h
        print(hh,timex.strftime('%Y %b %d %H'))
        # #%%

        # np.shape(tirf)
        rain_tot+=tirf[hh,:,:]
        plotvar=tirf[hh,:,:]
        plotvar=np.array(plotvar)
        
        # plotvar_w= np.mean(Wx[hh,9,:,:],axis=2)
        # plotvar_w= Wx[hh,8,:,:]
        lo_p=9 ; hi_p=11
        lo_p=8 ; hi_p=11
        print(P[lo_p:hi_p])
        plotvar_w= np.mean(Wx[hh,lo_p:hi_p,:,:],axis=0)
        np.shape(Wx)

        np.shape(plotvar_w)

        x = np.linspace(0,1,nj)
        y = np.linspace(0,1,ni)
        X, Y = np.meshgrid(x, y)
        np.shape(X)

        if subset_it:
            # if event!='2010-10_02-06':
            #     U=np.rot90(U.T)
            #     U=U[yc0:yc1,xc0:xc1]
            #     U=np.rot90(U.T)
            #     V=np.rot90(V.T)
            #     V=V[yc0:yc1,xc0:xc1]
            #     V=np.rot90(V.T)
            
            plotvar=np.rot90(plotvar.T)
            plotvar=plotvar[yc0:yc1,xc0:xc1]
            plotvar=np.rot90(plotvar.T)

            plotvar_w=np.rot90(plotvar_w.T)
            plotvar_w=plotvar_w[yc0:yc1,xc0:xc1]
            plotvar_w=np.rot90(plotvar_w.T)
            
            X=np.rot90(X.T)
            X=X[yc0:yc1,xc0:xc1]
            X=np.rot90(X.T)
            np.shape(X)
            Y=np.rot90(Y.T)
            Y=Y[yc0:yc1,xc0:xc1]
            Y=np.rot90(Y.T)
            
        lo=0. ; hi=20 ; dx=0.2
        cum_name=''
        if do_cum:
            lo=0. ; hi=100 ; dx=0.1
            plotvar=rain_tot
            cum_name='_cumulative'
        clevs=np.arange(lo,hi+dx,dx)

        # plotvar[plotvar<lo]=lo
        # plotvar[((mask<0.1)&(np.isnan(plotvar)))]=0
        # plotvar[mask==0]=np.nan
        # print(i+1,)
        # plotvar*=mask
        v=np.where(plotvar>0)
        result=np.nansum(plotvar[v])*areax/1e12
        print(timex.strftime('%Y %b %d %H'),hh,result)
            
        if do_plot:
            units='mm'

            plt.close()
            plt.clf()
            fig, ax = plt.subplots(figsize=(6, 6*ni/nj))
                
            # inferno
            # cool
            # cm_name='viridis'
            # cmap = plt.cm.get_cmap(cm_name)
            # cm = plt.cm.viridis
            # cm.set_under('w')
            # cm.set_over('y')
            plotvar[plotvar==0]=np.nan
            levs=np.arange(0,90,5).astype(float)
            levs=[1e0,2e0,4e0,8e0,16e0,32e0,64e0,96]
            # mine=plt.contourf(plotvar,clevs,cmap=cm,linewidths=th/1., vmin=lo,vmax=hi,extend='both')
            # mine=plt.imshow(plotvar,cmap=cm.viridis,norm=LogNorm(vmin=lo, vmax=hi))
            mine=plt.contourf(X,Y,plotvar, levels=levs,locator=ticker.LogLocator(), cmap=cm.Blues,extend='max')
            
            # CS=plt.contour(X,Y,plotvar_w,levels=[-0.5,0.5],linestyles=['dashed','dotted','solid'],colors=['k','g','r'])
            # ,linestyles=['solid','solid']
            # CS=plt.contour(X,Y,plotvar_w,levels=[-0.4,-0.2,0.2,0.4],colors=['b','b','r','r'])
            # CS=plt.contourf(X,Y,plotvar_w,levels=[-0.4,-0.2,0.2,0.4],colors=['b','b','r','r'],hatches=['-', '-', '\\', '\\'], alpha=0.2)
            # CS=plt.contourf(X,Y,plotvar_w,levels=[-0.2,0.2],colors=['k','r'],hatches=['-', '\\'], alpha=0.5)
            CS=plt.contour(X,Y,plotvar_w,levels=[-0.6,-0.3,0.3,0.6],colors=['darkgrey','grey','r','orange'])
            # CS=plt.contour(X,Y,plotvar_w,levels=[-0.3,0.3],colors=['k','r'])
            
            # for line in CS.collections:
            #     if line.get_linestyle() == [(None, None)]:
            #         print("Solid Line")
            #     else:
            #         line.set_linestyle([(0, (12.0, 3.0))])
            #         line.set_color('red')
            
            # plt.show()
            # lev_exp = np.arange(np.floor(np.log10(plotvar.min())-1),
            #                     np.ceil(np.log10(plotvar.max())+1))
            # levs = np.power(10, lev_exp)
            # mine = ax.contourf(X, Y, plotvar, levs, norm=colors.LogNorm())
            # plt.imshow(plotvar)
            # plt.colorbar()
            # pp=plt.contourf(x,y,z,clevs,cmap=cm, extend='both')
            # ax.grid(off)
            ax.axis('off')
    
            # cbar = fig.colorbar(mine)

    
            props = dict(boxstyle='round', facecolor='w', alpha=1,edgecolor='w')
    
            # xx0=0.5 ; yy0=0.95
            # mult=1.1
            
            # ax.text(xx0, yy0, 'Greenland CARRA rain flux',
            #         fontsize=font_size*mult,color='k',ha='center',
            #         bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
    
            annotatex=0

            if annotatex:
                xx0=0.1 ; yy0=0.96
                mult=1.1
                ax.text(xx0, yy0, timex.strftime('%Y %b %d %H')+' UTC',
                        fontsize=font_size*mult,color='k',bbox=props,rotation=0,transform=ax.transAxes,zorder=20)
            
            du_color_bar=0
            with_colorbarname=''
            if du_color_bar:
                with_colorbarname='_w_colorbar'
                width=0.03
                cbax = ax.inset_axes([1.01, 0.2, width, 0.5], transform=ax.transAxes)
                cbax.set_title(units,fontsize=font_size,c='k',ha='left')
                fig.colorbar(mine, ax=ax, cax=cbax, shrink=0.7, orientation='vertical')
                
                # t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                # f.colorbar(im, cax=axcolor, ticks=t, format="$%.2f$")
    
            # clb.set_clim(0,50)
            # cbar.draw_all()
            
            ly='p'

            if ly == 'x':
                plt.show() 
        
            DPI=200
            
            if ly == 'p':
                var='tirf'
                fig_basepath='./Figs/'
                fig_basepath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/rf/'
                figpath=fig_basepath#+var+'/'
                # os.system('mkdir -p '+figpath)
                # figpath=fig_basepath+var+'/'+event+'/'+cm_name+'/'
                # os.system('mkdir -p '+figpath)
                figname=figpath+timex.strftime('%Y %b %d %H')+cum_name+subset_name
                # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, transparent=True)#, facecolor='w', edgecolor='k')
                plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
                # figname=figpath+'case'
                # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

# ofile='//Users/jason/Dropbox/CARRA/CARRA_rainfall_study/output/20220914_rf'+'_'+str(ni)+'x'+str(nj)+'.npz'
# np.savez_compressed(ofile,rf=plotvar)

#%% multi color
# rf=np.load(ofile)
plotvar=rf['rf'].astype(float)
plotvar=np.rot90(plotvar.T)*1.1

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.colors import LogNorm

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

# N=len(bounds)
# bounds=np.logspace(0.1, 1, N, endpoint=True)*35-35-8
# print(bounds)

# plotvar_non_fuzzy[plotvar_non_fuzzy==0]=np.nan #set 0 to nan so they don't show in map

max_value=np.max(bounds)
min_value=0

#create colormap
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
cm.set_bad(color='white') #set color of zeros white
norm = BoundaryNorm(bounds, cm.N)
cbar_num_format = "%d"


    # # plotvar[plotvar<10]=10
    
    # # plt.imshow(plotvar)
    # if mask_out_ice:
    #     plotvar[mask==0]=0
   
plt.close()

plotvar_non_fuzzy=plotvar

# if mask_out_ice:
#     plotvar*=mask
#     plotvar[mask<0]=0
#     plotvar_non_fuzzy[mask<0]=0

areax=2.5e3**2
mass=np.nansum(plotvar[v])*areax/1e12
print("mass {:.1f}".format(mass))

# if mask_out_ice:
#     if i<2:plotvar_non_fuzzy[mask==0]=-1
    
# plt.imshow(lon)
#plot
fig = plt.figure(figsize=(8,10))
ax = plt.subplot(111)
# ax.set_title(tit, fontsize=fs)
ax = plt.gca()    

# tit=date.strftime('%d/%m/%Y')+' CARRA '+varnam2[i]
# ax.set_title(tit)


# lon-=360
# v=np.where((lon<-5)&(lat<82))
# maxval=np.max(plotvar_non_fuzzy)
# minval=np.min(plotvar_non_fuzzy[plotvar_non_fuzzy>1])
# maxval2=np.max(plotvar_non_fuzzy[v])
# # print("position of extremes")
# # print(np.max(plotvar_non_fuzzy))
# # print(maxval)
# # print(lat[plotvar==maxval2])
# # print(lon[plotvar==maxval2])
# SSWlat=lat[plotvar_non_fuzzy==maxval2][0]
# SSWlon=lon[plotvar_non_fuzzy==maxval2][0]
# # print(lat[plotvar==maxval2])
# # print(lon[plotvar==maxval2])
# alllat=lat[plotvar_non_fuzzy==maxval][0]
# alllon=lon[plotvar_non_fuzzy==maxval][0]
# minlat=lat[plotvar_non_fuzzy==minval][0]
# minlon=lon[plotvar_non_fuzzy==minval][0]

# plt.imshow(plotvar)
# pp=m.imshow(plotvar, cmap = cm,vmin=0,vmax=bounds[-1:][0]) 


pp=plt.imshow(plotvar, cmap = cm, norm=LogNorm(vmin=1,vmax=bounds[-1:][0]))
ax.axis('off')

# m.axis('off')
# m.drawcoastlines(color='k',linewidth=0.5)
# m.drawparallels([66.6],color='gray')
# m.drawparallels([60,70,80],dashes=[2,4],color='k')
# m.drawmeridians(np.arange(0.,420.,10.))
# m.drawmapboundary(fill_color='aqua')


# # --------------------- mask Iceland Sval with a polygon
# xx0=0.55
# X = np.array([[xx0,0], [xx0,1], [1, 1], [1, 0],[xx0,0]])
# ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
#         transform=ax.transAxes,zorder=9)) 
# # --------------------- mask Canada with a polygon
# X = np.array([[0,0.7], [0.39,0.9], [0.39, 1], [0.0, 1]])
# ax.add_patch(Polygon(X, closed=True,fill=True,color='w',
#         transform=ax.transAxes,zorder=9)) 
# # ---------------------


plt_colorbar=0

if plt_colorbar:
    ax = plt.gca()     
    # plt.title("Lambert Conformal Projection")
    # plt.show()
    # lons, lats = m(lon, lat)
    
    
    yy0x=0.35
    # --------------------- colorbar location
    cbaxes = fig.add_axes([xx0+0.01, 0.16, 0.01, yy0x]) 
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
    xx0=0.6 ; yy0=yy0x+0.18 ; dy2=-0.03 ; cc=0
    plt.text(xx0-0.05, yy0+cc*dy2,'mm w e', fontsize=font_size*mult,
             transform=ax.transAxes, color='k') ; cc+=1. 

ly='x'

if ly == 'x':
    plt.show()


DPI=300

if ly == 'p':
    fig_basepath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/streamline/2017-09-14/'
    plt.savefig(fig_basepath+'rf_20170914.png', bbox_inches='tight', dpi=DPI)#, transparent=True)#, facecolor='w', edgecolor='k')
    # figname=figpath+'case'
    # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
#%%
fig, ax = plt.subplots(figsize=(10, 6))

x=np.arange(iyear,fyear+1).astype(int)

tots=np.zeros(n_years)

for yy in range(n_years):
    tots[yy]=np.sum(rain_tot[yy,:,:]*mask)*areax/1e12

plt.plot(x,tots)
plt.ylabel('rainfall over Greenland ice,\nGigatonnes, source: CARRA')

#%%
out=pd.DataFrame({'year':x,'rf':tots})
out.to_csv('/Users/jason/Dropbox/CARRA/CARRA_202209/stats/Greenland_Sept_rainfall.csv')
#%% plot mean
mean_rain=np.mean(rain_tot[0:30,:,:],axis=0)
np.shape(mean_rain)

anom=rain_tot[32,:,:]-mean_rain
ofile='/Users/jason/Dropbox/CARRA/CARRA_202209/stats/CARRA_sept_2022_rainfall_anomaly_vs_1991-2020.npy'
anom.astype('float16').tofile(ofile)

ofile='/Users/jason/Dropbox/CARRA/CARRA_202209/stats/CARRA_sept_2022_rainfall.npy'
rain_tot[32,:,:].astype('float16').tofile(ofile)

# plotvar=mean_rain
plotvar=anom
# plotvar*=mask
# plotvar[mask<0.1]=-0.1

# plotvar[((plotvar>-0.1)&(plotvar<0.0))]=-0.1
units='mm'

print(np.max(plotvar))
plt.close()
plt.clf()
fig, ax = plt.subplots(figsize=(10, 10*ni/nj))
    
# inferno
# cool
cm_name='bwr_r'
cmap = plt.cm.get_cmap(cm_name)
cm = plt.cm.bwr_r
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

# xx0=0.5 ; yy0=0.95
# mult=1.1

# ax.text(xx0, yy0, 'Greenland glaciated area CARRA rainfall anomaly',
#         fontsize=font_size*mult,color='k',ha='center',
#         bbox=props,rotation=0,transform=ax.transAxes,zorder=20)

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

ly='x'
if ly == 'x':
    plt.show() 

DPI=300

if ly == 'p':
    var='tirf'
    fig_basepath='./Figs/'
    fig_basepath='/Users/jason/0_dat/CARRA/202209/Figs/'
    figpath=fig_basepath+var+'/'
    os.system('mkdir -p '+figpath)
    # figpath=fig_basepath+var+'/'+event+'/'+cm_name+'/'
    # os.system('mkdir -p '+figpath)
    figname=figpath+'202209_rain_anom_map'
    # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, transparent=True)#, facecolor='w', edgecolor='k')
    plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')
    # figname=figpath+'case'
    # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI, facecolor='w', edgecolor='k')

    
    