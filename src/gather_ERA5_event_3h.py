#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:31:10 2021

@author: jeb
"""
import cdsapi
import os
import pandas as pd

if os.getlogin() == 'adrien':
    base_path = '/home/adrien/EO-IO/rain_optics_SICE_AWS'
elif os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/rain_optics_SICE_AWS'

os.chdir(base_path)


# varnams=['rf','tp','sf']
# j=0 # select rf or tp

choices=['tcwv','z']

choices=['z']
# choices=['tzuv','tcwv']

choice_index=0
choice=choices[choice_index]

for choice_index,choice in enumerate(choices):

    print(choice_index)
    
    # path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
    
    opath='/Users/jason/0_dat/ERA5/events/'+choice+'/'
    os.system('mkdir -p '+opath)

    
    # os.chdir(path)
    # ofile="./ancil/events/"+varnams[j]+"_events.csv"
    # # read in events df
    # df=pd.read_csv(ofile)
    # print(df.columns)
    
    # path='./Figs/event/'+varnams[j]+'/'
    
    
    # files = sorted(glob(path+"*.png"), reverse=True)
    
    osx=0
    # numpy.savetxt("foo.csv", a, delimiter=",")
    
    # os.system('mkdir -p /Users/jason/0_dat/ERA5/events/')
    # os.system('mkdir -p /Users/jason/0_dat/ERA5/events//')
    
    # print(files)
    
    # if j:osx=1
    
    yearx=[]
    monx=[]
    dayx=[]
    Gt=[]
    
    # n=len(df)
    # for i in range(n):
    for i in range(1):
        for choice in choices:
            # year=df.year[i]
            # month=df.mon[i]
            # day=df.day[i]
            # print(year,month,day)
            year='2017' ; month='9' ; day='14'
            # year='2021' ; month='8' ; day='14'
            # year='2012' ; month='7' ; day='6'
            # year='2022' ; month='9' ; day='23'
            # year='2022' ; month='9' ; day='10'
            
            day_before=str(int(day)-1).zfill(2)
            day_three=str(int(day)+1).zfill(2)
            day_four=str(int(day)+2).zfill(2)
            day_five=str(int(day)+3).zfill(2)
            last_day=str(int(day)+4).zfill(2)
            # day_seven=str(int(day)+5).zfill(2)
            # day_eight=str(int(day)+6).zfill(2)
            # day_nine=str(int(day)+7).zfill(2)
            # last_day=str(int(day)+8).zfill(2)
            print(i,choice,year,month,day_before,day,last_day)
            ofile='/Users/jason/0_dat/ERA5/events/'+choice+'/'+str(year)+str(month).zfill(2)+str(day_before).zfill(2)+'-'+str(last_day).zfill(2)+'_3hourly_'+choice+'.grib'
            print(ofile)
#%%
        
        # #%%
        # for i,file in enumerate(files):
        #     year=files[i].split('/')[-1][4+osx:8+osx]
        #     month=files[i].split('/')[-1][9+osx:11+osx]
        #     day=files[i].split('/')[-1][12+osx:14+osx]
        #     day_before=str(int(day)-1).zfill(2)
        #     day_after=str(int(day)+1).zfill(2)
        #     print(i,year,month,day_before,day,day_after)
        #     Gt.append(str(files[i].split('/')[-1])[0:3+j])
        #     dayx.append(day)
        #     monx.append(month)
        #     yearx.append(year)
            # datex.append(year,month,day)
        
        # a = np.array([yearx,monx,dayx,Gt])
        
        # # a.tofile(ofile,sep=',')
        #
        
        # # from numpy import genfromtxt
        # # my_data = genfromtxt(ofile, delimiter=',',dtype='S')
        
        # df = pd.DataFrame({"year" : yearx, "mon" : monx, "day" : dayx, "mass" : Gt})
        # df.to_csv(ofile, index=False)
        
        #
            # i=0
            # print(str(files[i].split('/')[-1])[0:4])
        
            c = cdsapi.Client()
        
            if choice=='tcwv':
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': 'total_column_water_vapour',
                        'year': str(year),
                        'month': str(month).zfill(2),
                        'day': [
                            str(day_before).zfill(2),str(day).zfill(2),str(day_three).zfill(2),\
                                str(day_four).zfill(2),str(day_five).zfill(2),\
                                str(day_six).zfill(2),str(day_seven).zfill(2),\
                                str(day_eight).zfill(2),str(day_nine).zfill(2),\
                                    str(last_day).zfill(2),
                                    ],
                        'time': [
                            '00:00', '03:00', '06:00',
                            '09:00', '12:00', '15:00',
                            '18:00', '21:00',
                        ],
                        'format': 'grib',
                    },
                    ofile)
            if choice=='tzuv':        
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'grib',
                        'variable': [
                            'temperature', 'u_component_of_wind', 'v_component_of_wind',
                        ],
                        'pressure_level': '850',
                        'year': str(year),
                        'month': str(month).zfill(2),
                        'day': [
                            # str(day_before).zfill(2),str(day).zfill(2),str(day_after).zfill(2),day_four,
                            str(day_before).zfill(2),str(day).zfill(2),str(day_three).zfill(2),\
                                str(day_four).zfill(2),str(day_five).zfill(2),\
                                str(day_six).zfill(2),str(day_seven).zfill(2),\
                                str(day_eight).zfill(2),str(day_nine).zfill(2),\
                                    str(last_day).zfill(2),
                        ],
                        'time': [
                            '00:00', '03:00', '06:00',
                            '09:00', '12:00', '15:00',
                            '18:00', '21:00',
                            ],
                    },
                    ofile)
            if choice=='z':        
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': [
                            'geopotential', #'u_component_of_wind', 'v_component_of_wind',
                        ],        'pressure_level': ['850','500'],
                        'year': str(year),
                        'month': str(month).zfill(2),
                        'day': [
                            # str(day_before).zfill(2),str(day).zfill(2),str(day_after).zfill(2),day_four,
                            str(day_before).zfill(2),str(day).zfill(2),str(day_three).zfill(2),\
                                str(day_four).zfill(2),str(day_five).zfill(2),\
                                # str(day_six).zfill(2),str(day_seven).zfill(2),\
                                # str(day_eight).zfill(2),str(day_nine).zfill(2),\
                                    str(last_day).zfill(2),
                        ],
                        'time': [
                            '00:00',
                            '03:00',
                            '06:00',
                            '09:00',
                            '12:00',
                            '15:00',
                            '18:00',
                            '21:00',
                        ],
                        'format': 'grib',
                    },
                    ofile)
    #%%
    # c = cdsapi.Client()map_ERA5_3h_event
    
    # c.retrieve(
    #     'reanalysis-era5-pressure-levels',
    #     {
    #         'product_type': 'reanalysis',
    #         'variable': [
    #             'geopotential', 'u_component_of_wind', 'v_component_of_wind',
    #         ],        'pressure_level': '500',
    #         'year': '2017',
    #         'month': '09',
    #         'day': [
    #             '13', '14', '15',
    #         # 'year': '2000',
    #         # 'month': '08',
    #         # 'day': [
    #         #     '17', '18', '19',
    #         ],
    #         'time': [
    #             '00:00', '01:00', '02:00',
    #             '03:00', '04:00', '05:00',
    #             '06:00', '07:00', '08:00',
    #             '09:00', '10:00', '11:00',
    #             '12:00', '13:00', '14:00',
    #             '15:00', '16:00', '17:00',
    #             '18:00', '19:00', '20:00',
    #             '21:00', '22:00', '23:00',
    #         ],
    #         'format': 'grib',
    #     },
    #     # '/Users/jason/0_dat/ERA5/20000817-19_hourly_500hPa_UVZW.grib')
    #     '/Users/jason/0_dat/ERA5/20170913-15_hourly_500hPa_UVZW.grib')
    # c.retrieve(
    #     'reanalysis-era5-pressure-levels',
    #     {
    #         'product_type': 'reanalysis',
    #         'variable': 'geopotential',
    #         'pressure_level': '500',
    #         'year': '2017',
    #         'month': '09',
    #         'day': [
    #             '13', '14', '15',
    #         ],
    #         'time': [
    #             '00:00', '01:00', '02:00',
    #             '03:00', '04:00', '05:00',
    #             '06:00', '07:00', '08:00',
    #             '09:00', '10:00', '11:00',
    #             '12:00', '13:00', '14:00',
    #             '15:00', '16:00', '17:00',
    #             '18:00', '19:00', '20:00',
    #             '21:00', '22:00', '23:00',
    #         ],
    #         'format': 'grib',
    #     },
    #     '/Users/jason/0_dat/ERA5/20170913-15_hourly_500hPaZ.grib')