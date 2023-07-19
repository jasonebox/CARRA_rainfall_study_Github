#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:17:04 2022

@author: jason
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats

os.chdir('/Users/jason/Dropbox/AWS/T_records_Greenland_ice/')

GBI_or_NAO='NAO'
GBI_or_NAO='GBI'

def statsx2(varnam,x,y,color):
    # v=((df.year>=0)&(~np.isnan(y)))
    b, m = polyfit(x,y, 1)
    xx=[np.min(x),np.max(x)]
    coefs=stats.pearsonr(x,y)
    yy=[b + m * xx[0],b + m * xx[1]]
    # print(yy)
    # msg=region+" {:.1f}".format(m)+'x Global, '+str(year0_baseline)+' '+str(year1_baseline)
    ax.plot(xx, yy, '--',c=color,linewidth=2)

    # lab=formatx.format((m))+" ,\n"+\
    # sign+f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+"%, "+\
    # "1-p:"+f'{(1-coefs[1]):.2f}'
    # print(varnam,"{:.1f}".format(np.nanmean(y))+'±'+"{:.1f}".format(np.nanstd(y))+', change = '+sign+f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+"%, "+\
    # "1-p = "+f'{(1-coefs[1]):.2f}')
    # msg=varnam+','+"{:.1f}".format(np.nanmean(y))+','+"{:.1f}".format(np.nanstd(y))+','f'{((100*m*n_years/(xx[0]*m+b))):.0f}'+","+\
    # f'{(1-coefs[1]):.3f}'+'\n'
    # print(msg)
    # out.write(msg)
    conf=1-coefs[1]
    # m=m[0]
    # b=b[0]
    r=coefs[0]
    
    return m,b,r,conf

site='Summit'
site='Swiss Camp'
site='CARRA_tp'

if site=='Summit':
    fn='/Users/jason/Dropbox/AWS/T_records_Greenland_ice/monthly_air_T/Summit_T_monthly_seasonal_annual_1987-2022.csv'
    iyear=1987

if site=='Swiss Camp':
    fn='/Users/jason/Dropbox/AWS/T_records_Greenland_ice/monthly_air_T/SWC_T_monthly_seasonal_annual_1990-2022.csv'
    iyear=1990

fmonth=9
imonth=6
season='JJAS'

if site=='CARRA_tp':
    fn='/Users/jason/Dropbox/CARRA/CARRA_rainfall_study/stats/tabulate_annual_CARRA.csv'
    iyear=1991
    fmonth=12 ; imonth=1 ; season='annual'
    # fmonth=12 ; imonth=4 ; season='April through December'
    fmonth=9 ; imonth=6 ; season=' JJAS'
    # fmonth=12 ; imonth=12 ; season=' JFMAM'
    varnam='rf' ; varnam2='rainfall'
    # varnam='tp' ; varnam2='total precipitation, mm'

in_situ=pd.read_csv(fn)
fyear=2021
n_years=fyear-iyear+1
years=np.arange(iyear,fyear+1)


if GBI_or_NAO=='NAO':
    #---------------- 

    # years_NAO=np.arange(1991,2022).astype('str')
    # n_years_NAO=len(years_NAO)
    #---------------- 

    fn='/Users/jason/Dropbox/NAO/NCDC monthly NAO 1950-2021/nao_index.txt'
    df=pd.read_csv(fn,skiprows=9,delim_whitespace=True,names=['year','month','NAO'])

    # NAO["date"]=pd.to_datetime(NAO['YYYYMM'], format='%Y%m')
    # NAO['year'] = pd.DatetimeIndex(NAO['date']).year
    # print(len(NAO))
    # print(NAO.columns)

    NAO_annual=[]
    for yy in range(n_years):
        # v=NAO.year.astype(str)==((str(year))&())
        v=np.where((df.year==yy+iyear)&(df.month>=imonth)&(df.month<=fmonth))
        print('NAO',year,v[0])
        NAO_annual.append(np.mean(df.NAO[v[0]]))

    # teleconnection=pd.DataFrame({'year':years.astype(int),'NAO':np.array(NAO_annual)})
    # NAO.to_csv('/Users/jason/Dropbox/NAO/NCDC monthly NAO 1950-2021/output/NAO_1950_2022.csv')
    teleconnection=np.array(NAO_annual)
    teleconnection_name2='NAO'


if GBI_or_NAO=='GBI':
    ##%% GBI
    fn='/Users/jason/Dropbox/GBI/raw/gbi.ncep.day.txt'
    df=pd.read_csv(fn, delim_whitespace=True,names=['year','month','day','GBI'])
    df["time"]=pd.to_datetime(df[['year', 'month', 'day']])

    # df.index = pd.to_datetime(df.time)
    # df = df.loc[df['time']>='2021-01-01',:] 
    # iyear=1948 ; fyear=2021
    # n_years=fyear-iyear+1
    
    GBI=[]
    for yy in range(n_years):
        v=np.where((df.year==yy+iyear)&(df.month>=imonth)&(df.month<=fmonth))
        v=v[0]
        result=np.nanmean(df.GBI[v])
        GBI.append(result)
        print(yy+iyear,result)
    
    teleconnection=np.array(GBI)
    teleconnection_name2='GBI'


# len(teleconnection)
# plt.plot(years,teleconnection)
# #%%
# # --------------------------------------------------- plot stuff
th=1
font_size=20
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size

plt.close()
fig, ax = plt.subplots(figsize=(12,10))
    
# print(len(x),len(JJA_subset))
fig, ax = plt.subplots(figsize=(10,10))

# plt.scatter(teleconnection,in_situ.JJA)

v=((years>=iyear)&(years<2022))
x=teleconnection[v]
v=((in_situ.year>=iyear)&(in_situ.year<2022))
if site!='CARRA_tp':
    y=in_situ.JJA[v]
else:
    y=in_situ[varnam][v]
print(len(x),len(y))

m,b,r,conf=statsx2('hello',x,y,'k')

msg="$R^2$ = {:.3f}".format(r**2)+", 1-p = {:.3f}".format(conf)
# +\
#     f'{conf:.3f}'+'\n'

plt.scatter(x,y,label=msg)
plt.ylabel(site+' '+season+' '+varnam2)
plt.xlabel(teleconnection_name2)
plt.legend()

print(m,b)

#%% detrend

plt.rcParams['axes.grid'] = True

x=np.array(x)
fig, ax = plt.subplots(figsize=(10,10))

alpha=0.3

# years=years.astype(float)
plt.plot(years,JJA_subset,color='r',linewidth=th*2,label='summer temperature')
result=JJA_subset-(x*m+b)
plt.plot(years,result,color='gray',linewidth=th*3,label='detrended for GBI dependence',zorder=20)
ax[cc]fill_between(years,JJA_subset,result,color='r',linewidth=th*2,zorder=12,alpha=alpha,label='difference')#,label='std. dev.\n2007-2018')
plt.ylabel('Greenland June through August anomaly,\n°C from '+str(year0)+' to '+str(year1))
# plt.xlim(2000,2021)
plt.xlim(1948,2021)
plt.legend()

ly='x'
figname='JJA_Greenland_2tm_GBI_detrended'
if ly=='p':
    DPI=200
    fig_path='/Users/jason/Dropbox/GBI/figs/'
    plt.savefig(fig_path+figname+'.png', bbox_inches='tight', dpi=DPI)#, facecolor=bg, edgecolor=fg)
    # os.system('open '+fig_path+figname+'.png')


if ly == 'x':plt.show()

if ly == 'p':
    figname='./Figs/'+site+'_Air_Temperature_'+str(iyear)+'-'+str(fyear)
    plt.savefig(figname+'.png', bbox_inches='tight', dpi=150)
    plt.savefig(figname+'.pdf', bbox_inches='tight')
    