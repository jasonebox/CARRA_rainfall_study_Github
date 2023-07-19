#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:28:44 2021

@author: jeb
"""
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-carra-single-levels',
    {
        'domain': 'west_domain',
        'level_type': 'surface_or_atmosphere',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'albedo', 'skin_temperature', 'surface_latent_heat_flux',
            'surface_net_solar_radiation', 'surface_net_thermal_radiation', 'surface_sensible_heat_flux',
            'surface_thermal_radiation_downwards', 'total_column_integrated_water_vapour', 'total_precipitation',
        ],
        'product_type': 'forecast',
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
        'leadtime_hour': '3',
        'year': '2017',
        'month': '09',
        'day': [
            '13', '14', '15',
        ],
        'format': 'grib',
    },
    '/Users/jason/0_dat/CARRA/20170913-15_3h_SEB_prec.grib')