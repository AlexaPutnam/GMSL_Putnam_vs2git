#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:00:57 2022

@author: alexaputnam
"""

import numpy as np
from netCDF4 import Dataset

import lib_timeseries as lts


def gmsl_noaa():
    # https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/LSA_SLR_timeseries.php
    ds=Dataset('external/slr_sla_gbl_free_txj1j2_90.nc')
    tt = ds['time_tx'][:]
    ht = ds['sla_tx'][:]
    t1 = ds['time_j1'][:][ds['time_j1'][:]>np.nanmax(tt)]
    h1 = ds['sla_j1'][:][ds['time_j1'][:]>np.nanmax(tt)]
    t2 = ds['time_j2'][:][ds['time_j2'][:]>np.nanmax(t1)]
    h2 = ds['sla_j2'][:][ds['time_j2'][:]>np.nanmax(t1)]
    t3 = ds['time_j3'][:][ds['time_j3'][:]>np.nanmax(t2)]
    h3 = ds['sla_j3'][:][ds['time_j3'][:]>np.nanmax(t2)]
    t = np.hstack((tt,np.hstack((t1,np.hstack((t2,t3))))))
    gia_cor = ((t-t[0])*0.3) #0.3 mm/yr
    #gia_cor[-np.size(t3):]=0
    gmsl = np.hstack((ht,np.hstack((h1,np.hstack((h2,h3))))))#+gia_cor
    gmsl_smooth = lts.smooth(t,gmsl)
    gmsl_ds_nr,ratei,acceli,errR,errA = lts.deseason_reg(t,gmsl)
    #ratei,acceli = lts.get_rate_accel_w_seasons(t,gmsl)
    rate,accel = np.round(ratei,1),np.round(acceli,3)
    TIT = 'NOAA Rate: '+str(rate)+' mm/yr \n NOAA Acceleration: '+str(accel)+' mm/yr$^2$ \n NOAA Website: 3.0 +/- 0.4 mm/yr'
    print(TIT)
    return t,gmsl_smooth,rate,accel
    
def gmsl_goddard():
    text_array = np.loadtxt('external/gmsl_goddard.txt',delimiter=' ',dtype=object) #108,323,355
    df = np.array([np.float(x) for x in text_array[:, 0]])#np.array(text_array[:, 0])
    cyc = np.array([np.float(x) for x in text_array[:, 1]])#np.array(text_array[:, 1])
    t = np.array([np.float(x) for x in text_array[:, 2]])#np.array(text_array[:, 2])
    Nobs = np.array([np.float(x) for x in text_array[:, 3]])
    Nwobs = np.array([np.float(x) for x in text_array[:, 4]])
    sla_noGia = np.array([np.float(x) for x in text_array[:, 5]])
    sd_sla_noGia = np.array([np.float(x) for x in text_array[:, 6]])
    sla_smth_noGia = np.array([np.float(x) for x in text_array[:, 7]])
    sla = np.array([np.float(x) for x in text_array[:, 8]])
    sd_sla = np.array([np.float(x) for x in text_array[:, 9]])
    sla_smth = np.array([np.float(x) for x in text_array[:, 10]])
    gmsl_smooth = np.array([np.float(x) for x in text_array[:, 11]])
    
    gmsl_ds_nr,ratei,acceli,errR,errA = lts.deseason_reg(t,gmsl_smooth)
    #ratei,acceli = lts.get_rate_accel_w_seasons(t,gmsl_smooth)
    rate,accel = np.round(ratei,1),np.round(acceli,3)
    TIT = 'Goddard Rate: '+str(rate)+' mm/yr \n Goddard Acceleration: '+str(accel)+' mm/yr$^2$ \n Goddard Website: 3.4 +/- 0.4 mm/yr'
    print(TIT)
    return t,gmsl_smooth,rate,accel

