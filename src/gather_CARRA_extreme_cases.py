#!/usr/bin/env python3
# -*- coding: utf-8 -*-
    # '/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvw_1000-300hPa_14_levs.grib')
    # '/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_rh_1000-300hPa_14_levs.grib')
    # '/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_uvwt_3_levs.grib')

"""
Created on Mon Feb 20 11:32:42 2023

@author: jason
cp $HOME/.cdsapirc.CDS $HOME/.cdsapirc
cp $HOME/.cdsapirc $HOME/.cdsapirc.ACS
"""

year='2017' ; month='09' ; day='14' ; casex=year+month+day+'_18' # vert profile

year='2012' ; month='07' ; day='08' ; casex=year+month+day
# year='2012' ; month='07' ; day='09' ; casex=year+month+day
# year='2012' ; month='07' ; day='10' ; casex=year+month+day
# year='2012' ; month='07' ; day='11' ; casex=year+month+day
year='2012' ; month='07' ; day='12' ; casex=year+month+day

# year='2017' ; month='09' ; day='13' ; casex=year+month+day
# year='2017' ; month='09' ; day='14' ; casex=year+month+day
# year='2017' ; month='09' ; day='15' ; casex=year+month+day

year='2021' ; month='08' ; day='13' ; casex=year+month+day
# year='2021' ; month='08' ; day='14' ; casex=year+month+day
# year='2021' ; month='08' ; day='15' ; casex=year+month+day
# year='2021' ; month='08' ; day='16' ; casex=year+month+day

import cdsapi
c = cdsapi.Client()
do_vert=0

if do_vert:
    c.retrieve(
        'reanalysis-carra-pressure-levels',
        {
            'format': 'grib',
            'domain': 'west_domain',
            'variable': [
                'pseudo_adiabatic_potential_temperature',
                # 'relative_humidity',
                'temperature',
                'u_component_of_wind', 'v_component_of_wind', 'geometric_vertical_velocity',
                # 'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 
                # 'specific_cloud_rain_water_content',
                # 'specific_cloud_snow_water_content',
            ],
            'pressure_level': [
                # '10', '20', '30',
                # '50', '70', '100',
                # '150', '200', '250',
                '300', '400', '500',
                '600', '700', '750',
                '800', '825', '850',
                '875', '900', '925',
                '950', '1000',
        ],
            'product_type': 'analysis',
            'time': [
                '00:00', '03:00', '06:00','09:00', '12:00', '15:00','18:00', '21:00',
                # '15:00',
                # '18:00',
            ],
            'year': year,'month': month,'day': day,
        },
        '/Users/jason/0_dat/CARRA/202209_CDS/'+casex+'UTC_CARRA_multivars_1000-300hPa_all_levsx.grib')

#%%

# import cdsapi

# c = cdsapi.Client()

# c.retrieve(
#     'reanalysis-carra-single-levels',
#     {
#         'format': 'grib',
#         'domain': 'west_domain',
#         'level_type': 'surface_or_atmosphere',
#         'variable': [
#             '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
#         ],
#         'product_type': 'analysis',
#         'time': [
#             '00:00', '03:00', '06:00',
#             '09:00', '12:00', '15:00',
#             '18:00', '21:00',
#         ],
#         'year': year,'month': month,'day': day,

#     },
#     '/Users/jason/0_dat/CARRA/202209_CDS/'+casex+'_CARRA_u10v10t2m.grib')

do_horiz=1

import numpy as np

years=['2012','2017','2021','2022']
years=['2022']

for year in years:
    if year=='2012':
        days=np.arange(8,13).astype(str)
        month='07'

    if year=='2017':
        days=np.arange(13,16).astype(str)
        month='09'
        
    if year=='2021':
        days=np.arange(13,16).astype(str)
        month='08'

    if year=='2022':
        days=np.arange(1,6).astype(str)
        month='09'

    if year=='2022':
        days=np.arange(23,25).astype(str)
        month='09'

    for day in days:

        casex=year+month+day.zfill(2)
        
        if do_horiz:
            c.retrieve(
                'reanalysis-carra-single-levels',
                {
                    'format': 'grib',
                    'domain': 'west_domain',
                    'level_type': 'surface_or_atmosphere',
                    'variable': [
                        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                        'surface_latent_heat_flux', 'surface_net_thermal_radiation', 'surface_sensible_heat_flux',
                    ],
                    'product_type': 'forecast',
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                    ],
                    'year': year,'month': month,'day': day,
            
                    'leadtime_hour': '3',
                },
                '/Users/jason/0_dat/CARRA/202209_CDS/'+casex+'_CARRA_u10v10t2mLHF_SHF_LWNet.grib')
# #%%

# import cdsapi

# c = cdsapi.Client()

# c.retrieve(
#     'reanalysis-carra-single-levels',
#     {
#         'format': 'grib',
#         'domain': 'west_domain',
#         'level_type': 'surface_or_atmosphere',
#         'variable': [
#             '2m_temperature',
#         ],
#         'product_type': 'analysis',
#         'time': [
#             '00:00', '03:00', '06:00',
#             '09:00', '12:00', '15:00',
#             '18:00', '21:00',
#         ],
#         'year': '2017',
#         'month': '09',
#         'day': '14',
#     },
#     '/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_t2m.grib')

# #%%

# import cdsapi

# c = cdsapi.Client()

# c.retrieve(
#     'reanalysis-carra-single-levels',
#     {
#         'format': 'grib',
#         'domain': 'west_domain',
#         'level_type': 'surface_or_atmosphere',
#         'variable': [
#             'Time integral of rain flux', 'Time integral of total solid precipitation flux',
#         ],
#         'product_type': 'forecast',
#         'time': [
#             '00:00', '03:00', '06:00',
#             '09:00', '12:00', '15:00',
#             '18:00', '21:00',
#         ],
#         'leadtime_hour': '3',
#         'year': '2017',
#         'month': '09',
#         'day': '14',
#     },
#     '/Users/jason/0_dat/CARRA/202209_CDS/2017914_CARRA_rfsf.grib')