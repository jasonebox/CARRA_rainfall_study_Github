#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi annual barplot, feeds graphics that are assembled in AI

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
from scipy import stats
from numpy.polynomial.polynomial import polyfit

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
os.chdir(path)


wo=0

if wo:
    opath='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/stats/'
    ofile=opath+'tabulate_annual_CARRA_part_2_barplot_climatestats.csv'
    # ofile='./stats/tabulate_annual_CARRA_part_2_barplot_climatestats'+suffix+'.csv'
    out=open(ofile,'w+')
    out.write(',1991 to 2021 average,standard deviation,relative change 1991 to 2021,1-p\n')


for do_PG in range(2):
    if do_PG:
        hi_rf=5 ;hi_sf=37; sf_yos=1 ; rf_yos=0.2 ; rf_units="{:.1f}"  ; sf_units="{:.1f}"
        suffix='_PG'
    else:
        hi_rf=90 ; hi_sf=1070 ; sf_yos=10 ; rf_yos=1 ; rf_units="{:.0f}" ; sf_units="{:.0f}"
        suffix=''
    

    rf_var='rf'+suffix
    sf_var='sf'+suffix
    tp_var='tp'+suffix
       
    
    # global plot settings
    th=1
    font_size=18
    plt.rcParams['axes.facecolor'] = 'k'
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams["font.size"] = font_size
    
    
    def statsx(varnam,x,y,co,formatx):
        # v=((df.year>=0)&(~np.isnan(y)))
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        coefs=stats.pearsonr(x,y)
        sign='+'
        if m<0:sign=""
        lab=varnam+\
        "\n\u0394"+varnam+": "+sign+formatx.format((m*n_years))+" Gt y$^{-1}$,\n"+\
        sign+f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+"%, "+\
        "1-p:"+f'{(1-coefs[1]):.2f}'
        # print(varnam,"{:.1f}".format(np.nanmean(y))+'±'+"{:.1f}".format(np.nanstd(y))+', change = '+sign+f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+"%, "+\
        # "1-p = "+f'{(1-coefs[1]):.2f}')
        msg=varnam+','+"{:.1f}".format(np.nanmean(y))+','+"{:.1f}".format(np.nanstd(y))+','f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+","+\
        f'{(1-coefs[1]):.3f}'+'\n'
        print(msg)
        out.write(msg)
    
        return m,b,xx,lab,msg
    
    
    # read ice mask
    fn='./ancil/CARRA_W_domain_ice_mask.nc'
    nc2 = Dataset(fn, mode='r')
    # print(nc2.variables)
    mask = nc2.variables['z'][:,:]
    # mask = np.rot90(mask.T)
    # plt.imshow(mask)
    
    years=np.arange(1991,2022).astype('str')
    
    n_years=len(years)
    
    
    fs=22
    th=1
    # plt.rcParams['font.sans-serif'] = ['Georgia']
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.color'] = "grey"
    plt.rcParams["font.size"] = fs
    
    
    print(len(years))
    
    fn='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/stats/tabulate_annual_CARRA.csv'
    df=pd.read_csv(fn)
    x=df.year
    
    df[sf_var]=df[tp_var]-df[rf_var]
    
    print(len(df))
    print(df.columns)
    
    # print('2010 fr anom',df.fr[df.year==2010]/np.mean(df.fr))
    # print('2012 fr anom',df.fr[df.year==2012]/np.mean(df.fr))
    # print('2019 fr anom',df.fr[df.year==2019]/np.mean(df.fr))
    
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 12))
    w = 0.5
    divisor_for_bar_width=4
    
    sf_color='c'
    # snowfall
    m,b,xx,lab,msg=statsx('snowfall',x,df[sf_var],sf_color,sf_units)
    ax.bar(x-w/divisor_for_bar_width, df[sf_var], width=w, color=sf_color, align='center',label=lab)
    ax.plot(xx,[m*xx[0]+b,m*xx[1]+b],c=sf_color,ls='--',linewidth=th+2)
    
    mult=1 ; yy0=1.02 ; xx0=1.1
    leg = ax.legend(loc='upper left', bbox_to_anchor=(xx0, yy0),fontsize=font_size*mult)
    leg.get_frame().set_linewidth(0.0)
    
    ax.set_ylim(0,hi_sf)
    ax.set_xlim(1990.2,2021.6)
    ax.spines['left'].set_color(sf_color)
    ax.yaxis.label.set_color(sf_color)
    ax.tick_params(axis='y', colors=sf_color)
    ax.set_ylabel('total snowfall, Gt y$^{-1}$')
    
    mult=0.75
    
    y=df[sf_var]
    for i in range(len(df)):
        ax.text(x[i]-w/divisor_for_bar_width,y[i]+sf_yos,"{:.0f}".format((df[tp_var][i]-df[rf_var][i])),color=sf_color,ha="center",size=fs*mult)
    
    # rainfall
    rf_color='b'
    ax2 = ax.twinx()
    ax2.set_ylim(0,hi_rf)
    ax2.set_ylabel('total rainfall, Gt y$^{-1}$',color=rf_color)
    ax2.tick_params(axis='y', colors=rf_color)
    
    
    m,b,xx,lab,msg=statsx('rainfall',x,df[rf_var],rf_color,rf_units)

    m,b,xx,lab,msg=statsx('total precipitation',x,df[rf_var]+df[sf_var],rf_color,rf_units)
    
    ax2.bar(x+w/2,df[rf_var], width=w, color=rf_color, align='center',label=lab)
    
    ax2.plot(xx,[m*xx[0]+b,m*xx[1]+b],c=rf_color,ls='--',linewidth=th+2)
    
    mult=1 ; yy0=0.9
    leg2 = ax2.legend(loc='upper left', bbox_to_anchor=(xx0, yy0),fontsize=font_size*mult)
    leg2.get_frame().set_linewidth(0.0)
    
    mult=0.7
    y=df[rf_var]
    for i in range(len(years)):
        ax2.text(x[i]+w/2,y[i]+rf_yos,rf_units.format(df[rf_var][i]),color=rf_color,ha="center",size=fs*mult)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center' ) #rotation_mode="anchor",
    ax.set_xticks(np.arange(1991,2022))
    
    # plt.yscale('log')
    # plt.xlabel('maximum daily rainfall rate, mm\n for Greenland glaciated area')
    
    ly='x'

    if ly == 'p':
        figname='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/tabulate_annual_CARRA_part_2_barplot'+suffix+'.svg'
        plt.savefig(figname, bbox_inches='tight')#, dpi=150)
    
    # print(df.sort_values('maxlocalrate')[-10:])
    
    def statsx_rf(varnam,x,y,co):
        # v=((df.year>=0)&(~np.isnan(y)))
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        coefs=stats.pearsonr(x,y)
        sign='+'
        if m<0:sign=""
        lab=varnam+\
        "\n\u0394"+varnam+": "+sign+f'{(m*n_years):.1f}'+" %,\n"+\
        'relative change of '+sign+f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+"%,\n"+\
        "1-p:"+f'{(1-coefs[1]):.2f}'
        
        # print(varnam,"{:.1f}".format(np.nanmean(y))+'±'+"{:.1f}".format(np.nanstd(y))+', change = '+sign+f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+"%, "+\
        # "1-p = "+f'{(1-coefs[1]):.2f}')
        msg=varnam+','+"{:.1f}".format(np.nanmean(y))+','+"{:.1f}".format(np.nanstd(y))+','f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+","+\
        f'{(1-coefs[1]):.3f}'+'\n'
        print(msg)
        out.write(msg)

        return m,b,xx,lab,msg
    #%%
    do_fr=1
    if do_fr:
        plt.close()
        fig, ax = plt.subplots(figsize=(12, 12))
        
        y=df[rf_var]/df[tp_var]*100
        m,b,xx,lab,msg=statsx_rf('rain fraction',x,y,'k')
        
        ax.plot(x+0.3, y, color='k',label=lab)
        ax.plot(xx,[m*xx[0]+b,m*xx[1]+b],c='k',ls='--',linewidth=th+2)
        ax.set_ylim(0,5.2)
        
        ax2 = ax.twinx()
        ax2.plot(x+0.3, y, color='k',label=lab)
        ax2.set_ylim(0,5.2)
        
        mult=1 ; yy0=0.9
        leg2 = ax.legend(loc='upper left', bbox_to_anchor=(xx0, yy0),fontsize=font_size*mult)
        leg2.get_frame().set_linewidth(0.0)
        
        yos=1 ; mult=0.7
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center' ) #rotation_mode="anchor",
        ax.set_xticks(np.arange(1991,2022))
        
        ly='x'
        
        if ly == 'p':
            figname='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/Figs/tabulate_annual_CARRA_part_2_rain_fraction'+suffix+'.svg'
            plt.savefig(figname, bbox_inches='tight')#, dpi=150)

out.close()
