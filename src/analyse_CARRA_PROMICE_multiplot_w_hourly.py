# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé and Jason Box, GEUS (Geological Survey of Denmark and Greenland)

# analyzes output from ./CARRA_PROMICE_combination.py

"""

import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys

base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'

os.chdir(base_path)

sys.path.append(base_path)
import WetBulb as wb


# graphics settings
fs=22 # fontsize
th=1 # default line thickness
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "grey"
plt.rcParams["font.size"] = fs
plt.rcParams['figure.figsize'] = 25, 10
plt.rcParams["mathtext.default"]='regular'


#  load PROMICE data

L3 = True

site='NUK_L'
site='QAS_M'
# site='QAS_L'
site='QAS_U'
# fn='/Users/jason/Dropbox/AWS/PROMICE/2019-08-02/'+site+'_hour_v03_upd.txt'
# os.system('head -n1 '+fn) 

fn='./PROMICE_hourly_v03/'+site+'_hour_v03_L3.txt' 

if L3:
    
    df=pd.read_csv(fn, sep='\t')
    df.index = pd.to_datetime(df['time'])
    
else:
    
    df=pd.read_csv(fn, delim_whitespace=True)


df = df.loc[df['time']>'2017-09-01',:] 
df = df.loc[df['time']<'2018-01-01',:] 
df = df.reset_index(drop=True)
    
PROMICE_str_time = df.Year.astype(str) + ' '\
    + df.MonthOfYear.astype(str) + ' '\
        + df.DayOfYear.astype(str)+ ' '\
            +df["HourOfDay(UTC)"].astype(str)     


df.index = pd.to_datetime(PROMICE_str_time, format='%Y %m %j %H')
df[df == -999.0] = np.nan


print(df.columns)

## %% Wet Bulb temperature 

Temperature = np.array(df['AirTemperature(C)'])
Pressure = np.array(df['AirPressure(hPa)'] / 0.01)  # hPa to Pa
Humidity = np.array(df['RelativeHumidity(%)'])
Wet_Bulb_Temperature, Equivalent_Temperature, Equivalent_Potential_Temperature \
    = wb.WetBulb(Temperature, Pressure, Humidity, HumidityMode=1)
    
df['Tw'] = Wet_Bulb_Temperature

##%%

#plt.plot(df.Tw-df['AirTemperature(C)'])

# %% -------------------------------------------------------------------- read DVA model output
if site=='QAS_M':
    fn='/Users/jason/Dropbox/AWS/PROMICE/output_SEB/outputDVA/'+site+'_2017.txt' # no QAS_U

    # Z0_ice = 1 cm
    # Precip = 4 mm for every hour with LR_in exceeding sigma*T^4
    # Obviously precip is wrong/parameterized. You could subtract the rain energy and add your own.
    # The plots are not polished in any way, just for my superficial interpretation. I left them in as they may help you too. Even the data file needs polishing, but I reckon you know what’s what.
    # QAS_M starts in Aug 2017 as data before that station visit are crap.
    # QAS_L starts in Jul 2017 because why not.
    # Could add 2018-2020 if input data are ok. This is all run using v03 PROMICE hourly downloads.
    # rain heat flux, melt energy surface, melt energy internal (solar penetration switched off in these runs), surface height change (calculated), surface height change by melt, by sublimation, water runoff, snow thickness, snow accumulation, rainfall, ...
     
    DVA=pd.read_csv(fn, delim_whitespace=True)
    
    DVA_time_str = (DVA.Year+2000).astype(int).astype(str).str.zfill(4) + ' '\
            + DVA.Day.astype(int).astype(str).str.zfill(3) + ' '\
            + DVA.Hour.astype(int).astype(str).str.zfill(2)
                    # 
    DVA["time"]=pd.to_datetime(DVA_time_str, format='%Y %j %H')
    
    DVA = DVA.loc[DVA['time']>'2017-09-01',:] 
    DVA = DVA.loc[DVA['time']<'2018-01-01',:] 
    DVA = DVA.reset_index(drop=True)
    
    DVA.rename(columns={'LHF_Wm-2': 'LatentHeatFlux(W/m2)'}, inplace=True)
    DVA.rename(columns={'SHF_Wm-2': 'SensibleHeatFlux(W/m2)'}, inplace=True)
    DVA.rename(columns={'SRin_Wm-2': 'ShortwaveRadiationDown(W/m2)'}, inplace=True)
    DVA.rename(columns={'SRout_Wm-2': 'ShortwaveRadiationUp(W/m2)'}, inplace=True)
    DVA.rename(columns={'LRin_Wm-2': 'LongwaveRadiationDown(W/m2)'}, inplace=True)
    DVA.rename(columns={'LRout_Wm-2': 'LongwaveRadiationUp(W/m2)'}, inplace=True)
    
    DVA["ME"]=DVA['MEsurf_Wm-2']-DVA['rainHF_Wm-2']
                  
    print(DVA.columns)
    
    print(DVA["time"])

#%% collocate, takes time

print('--------------------------------------------------------------- collocate rain with PROMICE')

df['rain_corrected']=0.
df['rain_uncorrected']=0.

namx=site
if site=='QAS_U':namx='QAS_L'
fn='/Users/jason/Dropbox/rain_PROMICE/products/'+namx+'_rain_data_hourly.csv'
dfr=pd.read_csv(fn)
print(dfr.columns)

dfr_time_str = dfr.year.astype(str).str.zfill(4) + ' '\
        + dfr.month.apply(lambda x: '{0:0>2}'.format(x)) + ' '\
            + dfr.day.apply(lambda x: '{0:0>2}'.format(x))+ ' '\
                +dfr.hour.apply(lambda x: '{0:0>2}'.format(x))
                # 
# dfr.index = pd.to_datetime(dfr_time_str, format='%Y %m %d %H')
dfr["date"]=pd.to_datetime(dfr_time_str, format='%Y %m %d %H')
# if site=='QAS_M':
#     dfr["date"]=dfr["date"]-pd.Timedelta(hours=4)
print(dfr["date"])


Nrain=len(dfr)
for i in range(Nrain):
    v = np.where((df.DayOfYear == dfr['day of year'][i]) & (df.Year == dfr.year[i]) & (df["HourOfDay(UTC)"] == dfr.hour[i]))
    if len(v[0]) > 0:
        # print(i,v[0])
        df['rain_corrected'][v[0]]=dfr['rain_corrected'][i]
        df['rain_uncorrected'][v[0]]=dfr['rain_uncorrected'][i]
# #%%
# plt.plot(df['rain_uncorrected'])
# df.columns

# plt.scatter(df['WindSpeed(m/s)'],df['rain_uncorrected'])

#%% collocate, takes time
if site=='QAS_M':

    print('--------------------------------------------------------------- collocate DVA turb fluxes with PROMICE')
    
    df['LHF']=0.
    df['SHF']=0.
    df['ME']=0.
    
    
    # #%%
    
    N_DVA=len(DVA)
    for i in range(N_DVA):
        v = np.where((df.DayOfYear == DVA.Day[i]) & (df.Year == DVA.Year[i]+2000) & (df["HourOfDay(UTC)"] == DVA.Hour[i]))
        if len(v[0]) > 0:
            # print(i,v[0])
            df['LHF'][v[0]]=DVA['LatentHeatFlux(W/m2)'][i]
            df['SHF'][v[0]]=DVA['SensibleHeatFlux(W/m2)'][i]
            df['ME'][v[0]]=DVA['ME'][i]

    # #%% plt SHF
    
    # t0=datetime(2017, 9, 13) ; t1=datetime(2017, 9, 16,12)
    
    # # plt.plot(df['SHF'][t0:t1]-df['SensibleHeatFlux(W/m2)'][t0:t1],c='r')
    # # plt.plot(df['LHF'][t0:t1]-df['LatentHeatFlux(W/m2)'][t0:t1],c='b')
    
    # plt.plot(df['SHF'][t0:t1],c='r',label='DVA SHF')
    # plt.plot(df['SensibleHeatFlux(W/m2)'][t0:t1],'--',c='r',label='default PROMICE SHF')
    
    # plt.plot(df['LHF'][t0:t1],c='b',label='DVA LHF')
    # plt.plot(df['LatentHeatFlux(W/m2)'][t0:t1],'--',c='b',label='default PROMICE LHF')
    # plt.legend()

#%% compute PROMICE SEB'
#
print('--------------------------------------------------------------- compute PROMICE SEB')
if site=='QAS_U':
    df['SHF']=df['SensibleHeatFlux(W/m2)']
    df['LHF']=df['LatentHeatFlux(W/m2)']

# df["tnet"]=df['SensibleHeatFlux(W/m2)']+df['LatentHeatFlux(W/m2)']
df["tnet"]=df['SHF']+df['LHF']
df["Lnet"]=df['LongwaveRadiationDown(W/m2)']-df['LongwaveRadiationUp(W/m2)']
df["Snet"]=df['ShortwaveRadiationDown(W/m2)']-df['ShortwaveRadiationUp(W/m2)']
# df["z"]=df['DepthPressureTransducer_Cor_adj(m)']-df['DepthPressureTransducer_Cor_adj(m)'][((df.Year==2017)&(df.MonthOfYear==7)&(df.DayOfMonth==13))][0]
# df["z"]=df['SurfaceHeight_adj(m)']-df['SurfaceHeight_adj(m)'][((df.Year==2017)&(df.MonthOfYear==7)&(df.DayOfMonth==13))][0]
# df['SensibleHeatFlux(W/m2)'][np.isnan(df['SensibleHeatFlux(W/m2)'])]=df['WindSpeed(m/s)'][np.isnan(df['SensibleHeatFlux(W/m2)'])]*12.5
# df['LatentHeatFlux(W/m2)'][np.isnan(df['LatentHeatFlux(W/m2)'])]=df['WindSpeed(m/s)'][np.isnan(df['LatentHeatFlux(W/m2)'])]*12.5

# df["tnet"]=df['SensibleHeatFlux(W/m2)']+df['LatentHeatFlux(W/m2)']
# df["tnet"]=df['SHF']+df['LHF']

df["Lnet"]=df['LongwaveRadiationDown(W/m2)']-df['LongwaveRadiationUp(W/m2)']
df["Snet"]=df['ShortwaveRadiationDown(W/m2)']-df['ShortwaveRadiationUp(W/m2)']
df["Rnet"]=df['Snet']+df['Lnet']
L_fusion=334000
sec_per_hour=3600

df["abl_from_EB"]=-(df["tnet"]+df["Rnet"])/L_fusion*sec_per_hour

ro_water=1000
m2mm=1000
cp_water=4200

df['QRx'] =0.
df['QR'] =0.
v=df['AirTemperature(C)']>0

# x=df['AirTemperature(C)']
# c1=0.000646; c2=4153.82
# es = 6.1070*np.exp(17.38*x/(239.0+x))
# t=x
# p  = df['AirPressure(hPa)']
# e  = 1000*ee/pp
# t = (tt+273.15)(1000/p)*0.286-273.15
# p  = 1000.

# f  = 10.
# do while (f.ge.1.0)
# f  = (es(t)-e)/(c2*es(t)/(t+273.15)**2+c1*p)
# e  = e+c1*p*f
# t  = t-f
# tnat = t

df['QRx'][v]=-ro_water*cp_water*df['AirTemperature(C)'][v]*df['rain_corrected'][v]/m2mm/L_fusion
df['QR'][v]=-ro_water*cp_water*df['Tw'][v]*df['rain_corrected'][v]/m2mm/L_fusion

# plt.plot(df['QR'])

#
print('--------------------------------------------------------------- computing cumulative melt')
N=len(df)
# datex=df.time[t0:t1]
meltx=np.zeros(N)
meltx_w_rain=np.zeros(N)

temp=0 ; temp2=0
vars=["abl_from_EB","SensibleHeatFlux(W/m2)","LatentHeatFlux(W/m2)"]
for i in range(1,N-1):
    print(N-i)
    # for var in vars:
    #     if((np.isfinite(df[var][i-1]))&(np.isfinite(df[var][i+1]))&(~np.isfinite(df[var][i]))):
    #         df[var][i]=(df[var][i-1]+df[var][i+1])/2

    if ((df["abl_from_EB"][i]<0)&(np.isfinite(df["abl_from_EB"][i]))):
        temp+=df["abl_from_EB"][i]
        temp2+=(df["abl_from_EB"][i]+df["QR"][i])
    # df['melt'][i]=temp
    meltx[i]=temp
    meltx_w_rain[i]=temp2
    # df['melt_w_rain'][i]=temp2

df['melt']=meltx
# df['melt_DVA']=0
df['melt_w_rain']=meltx_w_rain
# df["z"]=df['SurfaceHeight_summary(m)']-df['SurfaceHeight_summary(m)'][((df.Year==2017)&(df.MonthOfYear==7)&(df.DayOfMonth==13))][0]
# plt.plot(df['melt'])
# plt.plot(df["abl_from_EB"]*1000)
#
#
print(df.columns)
th=1.5

#%%
import matplotlib.dates as mdates

print('--------------------------------------------------------------- plot result')

t0=datetime(2017, 9, 13) ; t1=datetime(2017, 9, 16,12) ; case_name='2017_09_14'
# t0=datetime(2018, 7, 19) ; t1=datetime(2018, 7, 20) ; case_name='2018_07_19'
# t0=datetime(2020, 8, 27) ; t1=datetime(2020, 8, 28) ; case_name='2020_08_28'
# 2018-07-19

plt.close()
n_rows=4
fig, ax = plt.subplots(n_rows,1,figsize=(10,18))

cc=0#----------------------------------------------------------------------------------- upper wind and t
ax[0].set_title(site)#+' '+PROMICE_temperature_name)

ax[cc].plot(df["WindSpeed(m/s)"][t0:t1],linewidth=th*2, color='k',label='wind')        
ax[cc].set_ylabel('wind speed, m s$^{-1}$', color='k')
ax[cc].get_xaxis().set_visible(False)
ax[cc].legend(prop={'size': fs*0.6},loc=2)#'center left', bbox_to_anchor=(1.1, 0.5))


ax[cc].set_ylim(0,10)
ax[cc].set_yticks(np.arange(0, 12, 2))

ax3 = ax[cc].twinx()
vnam='AirTemperature(C)' ; co='orange'
# ax3.plot(CARRA_PROMICE[vnam],'o-', color=co)
ax3.plot(df[vnam][t0:t1],linewidth=th*2, color=co,label='T')
vnam='Tw' ; co='r'
# ax3.plot(CARRA_PROMICE[vnam],'o-', color=co)
ax3.plot(df[vnam][t0:t1],linewidth=th*2, color=co,label='T$_{W}$')


ax3.legend(prop={'size': fs*0.6})#,loc='center left', bbox_to_anchor=(1.1, 0.5))

lo=0 ; hi=7.2
if case_name=='2018_07_19':
    lo=0 ; hi=8
ax3.set_ylim(lo,hi)
ax3.set_yticks(np.arange(lo-1, hi+1, 1))

co='k'
ax3.set_ylabel('air temperature, ° C', color=co)
ax3.spines['right'].set_color(co)
ax3.xaxis.label.set_color(co)
ax3.tick_params(axis='y', colors=co)

ax3.axhline(y=0,linestyle='--',linewidth=th*1.5,color='darkorange')


cc+=1#---------------------------------------------------------------- fluxes

vnam='SHF' ; co='r'
ax[cc].plot(df[vnam][t0:t1],linewidth=th*2, color=co,label="SHF")
# ax[cc].plot(df[vnam], color=co,label="SHF")
vnam='LHF' ; co='b'
ax[cc].plot(df[vnam][t0:t1],linewidth=th*2, color=co,label="LHF")
# ax[cc].plot(df[vnam], color=co,label="LHF")

co='m'
ax[cc].plot(df["tnet"][t0:t1],linewidth=th*2, color=co,label="NetHF")
# ax[cc].plot(df["tnet"], color=co,label="NetHF")



ax[cc].axhline(y=0,linewidth=th*1.5, color='grey')
ax[cc].legend(prop={'size': fs*0.8},loc='center left', bbox_to_anchor=(1, 0.5))

ax[cc].get_xaxis().set_visible(False)
        

vnam='ShortwaveRadiationDown(W/m2)' ; co='darkorange'
# ax[cc].plot(CARRA_PROMICE[vnam], color=co,label='SWD')
# vnam='LongwaveRadiationDown(W/m2)' ; co='purple'
# ax[cc].plot(CARRA_PROMICE[vnam], color=co,label='LWD')
ax[cc].plot(df["Snet"][t0:t1],linewidth=th*2, color='darkorange',label='SNet')        
# ax[cc].plot(df["Snet"],linewidth=th*2, color='darkorange',label='SNet')        
# ax[cc].plot(df["Rnet"][t0:t1],linewidth=th*2, color='k',label='RNet')        

ax[cc].plot(df["Lnet"][t0:t1],linewidth=th*2, color='g',label='LNet')        
# ax[cc].plot(df["Lnet"], color='g',label='LNet')        
ax[cc].axhline(y=0, color='grey')
ax[cc].set_ylabel('energy flux, W m$^{-2}$', color='k')
lo=-100 ; hi=350
if case_name=='2018_07_19':
    lo=0 ; hi=400
ax[cc].set_ylim(lo,hi)
ax[cc].set_yticks(np.arange(-100, 350, 50))
ax[cc].legend(prop={'size': fs*0.8},loc='center left', bbox_to_anchor=(1, 0.5))
ax[cc].get_xaxis().set_visible(False)

cc+=1#---------------------------------------------------------------- rainfall
# ax[cc].set_xlim(t0,t1)
labx='rainfall'
if site=='QAS_U':labx='QAS_L\nrainfall'

# t0x=datetime(2017, 9, 12,0) ; t1x=datetime(2017, 9, 15,20)
# v=((dfr["date"]>=t0x)&(dfr["date"]<=t1x))
ax[cc].plot(df["rain_corrected"][t0:t1],linewidth=th*2, color='b',label=labx) #, drawstyle='steps'
ax[cc].plot(df["rain_uncorrected"][t0:t1],linewidth=th*2, color='purple',label='un-\ncorrected') 
ax[cc].set_ylabel('rainfall rate, mm h$^{-1}$', color='k')
ax[cc].set_ylim(0,23)
ax[cc].set_yticks(np.arange(0, 24, 2))

tot_rain=np.nansum(df["rain_corrected"][t0:t1])
print('mm total rain',tot_rain)
# ax[cc].set_ylim(-100,260)
ax[cc].legend(prop={'size': fs*0.8},loc='center left', bbox_to_anchor=(1, 0.77))
ax[cc].get_xaxis().set_visible(False)
# ax[cc].set_xlim(t0-pd.Timedelta(hours=4),t1-pd.Timedelta(hours=4))
# if site=='QAS_M':ax[cc].set_xlim(t0+pd.Timedelta(hours=1),t1-pd.Timedelta(hours=4))

cc+=1#---------------------------------------------------------------- bottom , surf height and air t
ax[cc].set_xlim(t0,t1)

# vnam='HeightSensorBoom(m)' ; co='r'
# ax[cc].plot(CARRA_PROMICE[vnam],'o-', color=co,label=vnam)

if site=='QAS_U':
    vnam='SurfaceHeight2_adj(m)'
    vnam='DepthPressureTransducer_Cor(m)'

if site=='QAS_M':
    vnam='SurfaceHeight2_adj(m)'
    # vnam='HeightStakes(m)'
if site=='QAS_L':
    print("surface height instruments not functioning")
    vnam='SurfaceHeight_adj(m)'
    vnam='DepthPressureTransducer_Cor(m)'
    vnam='SurfaceHeight_summary(m)'
    vnam='DepthPressureTransducer_Cor_adj(m)'
    vnam='SnowHeight_adj(m)'

# vnam='SurfaceHeight2_adj(m)'
# vnam='HeightStakes(m)'

# # t0=datetime(2017,8, 20) ; t1=datetime(2017, 9, 21)
# plt.plot(df[vnam][t0:t1])
# t0=datetime(2017,9, 16) ; t1=datetime(2017, 9, 21)
# print(np.nanmean(2.6-df['HeightStakes(m)'][t0:t1]))

# if site=='QAS_M':

vnamx='HeightStakes(m)'
y=df[vnamx][t0:t1]
print("initial height stakes",y[0])

# plt.plot(y[0]+df[vnamx][t0:t1])

y=df[vnam][t0:t1]
print("initial height "+vnam,y[0])
y-=y[0]

vnam='DepthPressureTransducer_Cor(m)'
y2=df[vnam][t0:t1]*0.9
print("initial height",y[0])
y2-=y2[0]

# y[0:20]/=10#=y2[0:20]+0.05
co='b'

if site=='QAS_M':
    ax[cc].plot(y+0.05, color=co,label='surface\nheight',zorder=30,linewidth=th*2)

    ax[cc].text(t0,0.03,'        A',color='b')

# t0=datetime(2017, 9, 1) ; t1=datetime(2017, 9, 20)

# plt.plot(y,c='b',drawstyle='steps',label='hydraylic')
# vnamx='HeightStakes(m)'
# y=df[vnamx][t0:t1]
# y=y[0]-y
# print("initial height stakes",y[0])
# plt.plot(y,c='c',drawstyle='steps',label='sonic')
# plt.legend()
# #%%
co='b'
if site!='QAS_M':
    ax[cc].plot(y, color=co,label='ablation PT',zorder=30,linewidth=th*2)

if site=='QAS_M':
    ax[cc].plot(y2*1.15,'--', color=co,label='ablation PT',zorder=30,linewidth=th*2)

h20_density=1000
y2=df['melt'][t0:t1]/h20_density
y2-=y2[0]
y3=df['melt_w_rain'][t0:t1]/h20_density
y3-=y3[0]

melt_wo_rain=-float(y2[-1:])*1000
melt_w_rain=-float(y3[-1:])*1000

rain_melt=(melt_w_rain-melt_wo_rain)
melt_per_mm_rain=rain_melt/tot_rain
melt_per_mm_wo_rain=melt_wo_rain/tot_rain
melt_per_event_scaled_by_rain=melt_wo_rain*melt_wo_rain/tot_rain


melt_rain_melt_per_300_mm=melt_per_mm_rain*330
print('rain_melt',rain_melt)
print('inches rain_melt',rain_melt/25.4)
print('melt_per_mm_rain',melt_per_mm_rain)
print('mm melt_rain_melt_per_330_mm',melt_rain_melt_per_300_mm)
print('inches melt_rain_melt_per_330_mm',melt_rain_melt_per_300_mm/25.4)
print('melt_per_event_scaled_by_rain',melt_per_event_scaled_by_rain)
print("melt at end of period",melt_wo_rain)
print("melt at end of period w rain",melt_w_rain)
print("inches melt at end of period w rain",melt_w_rain/25.4)
print("melt at end of period",float(y2[-1:]))
print("melt ratio at end of period w rain",float(y3[-1:])/float(y2[-1:]))

if site!='QAS_L':ax[cc].plot(y2, color='r',label='melt\nfrom\nSEB',zorder=20,linewidth=th*2)
if site!='QAS_L':ax[cc].plot(y3, color='m',label='melt\nfrom\nSEB\nw rain',zorder=30,linewidth=th*2)
ax[cc].set_ylabel('surface mass flux, m w. equiv.', color='k')
# ax[cc].tick_params(axis='y', colors=co)
# ax[cc].spines['left'].set_color(co)
# ax[cc].yaxis.label.set_color(co)
# ax[cc].tick_params(axis='y', colors=co)
# ax[cc].legend(prop={'size': fs*0.6})#,loc='center left', bbox_to_anchor=(1.1, 0.5))

ax[cc].legend(prop={'size': fs*0.8},loc='upper left', bbox_to_anchor=(1, 1.04))
# ax[cc].legend(prop={'size': fs*0.8},loc='upper right')
lo=-0.13 ; hi=0.06
if case_name=='2018_07_19':
    lo=-0.06 ; hi=0.05
ax[cc].set_ylim(lo,hi)
ax[cc].set_yticks(np.arange(lo+0.01, hi+0.02, 0.02))
ax[cc].xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Hh'))

 # ax3.set_ylim(-2,7)

 # ax3.plot(CARRA_PROMICE[vnam],'o-', color=co)
 # ax3.set_ylabel('deg. C', color=co)
 # ax3.set_ylim(-2,7)

 # ax[cc].set_ylim(-0.5,2)
 # ax[cc].set_ylabel('deg. C', color='k')

 # ax3 = ax[cc].twinx()
 # vnam='AirTemperature(C)' ; co='r'
 # # ax3.plot(vnam, drawstyle='steps', color=co)
 # # ax3.set_ylim(-2,7)

 # ax3.plot(CARRA_PROMICE[vnam],'o-', color=co)
 # ax3.set_ylabel('deg. C', color=co)
 # ax3.set_ylim(-2,7)
 
ax[cc].set_xlim(t0,t1)
plt.setp( ax[cc].xaxis.get_majorticklabels(), rotation=90,ha='right' )
# # plt.setp( ax[cc].yaxis.get_majorticklabels(), color='k' )        

ly='x'

if ly=='p':
    figname='/Users/jason/Dropbox/CARRA/CARRA rainfall study/Figs/'+'analyse_CARRA_PROMICE_multiplot_w_hourly_'+site+'.svg'
    plt.savefig(figname, bbox_inches='tight')#,dpi=300)