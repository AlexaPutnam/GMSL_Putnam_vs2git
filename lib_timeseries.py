#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:18:14 2022

@author: alexaputnam
"""
import xarray as xr
import numpy as np

def tdm_out(FN,MEAN=True,MINCYC=[]):
    ds = xr.open_dataset(FN,decode_times=False)
    dsla = ds.sla_gia.data[:,0] #!!! sla or sla_gia # output = (N,4). Columns are mean, std, min, max 
    nr = ds.nr.data
    cycle = ds.cycle.data
    time = ds.time.data
    if MEAN == False:
        if np.size(MINCYC)==0:
            return cycle,dsla*1000.
        else:
            imc = np.where(~np.isnan(dsla))[0]
            LIM = np.size(imc)-MINCYC
            return cycle[imc[LIM:]],dsla[imc[LIM:]]*1000.
    else:
        if np.size(MINCYC)==0:
            bias,sd = np.nanmean(dsla),np.nanstd(dsla)
        else:
            imc = np.where(~np.isnan(dsla))[0]
            LIM = np.size(imc)-MINCYC
            bias = np.nanmean(dsla[imc[LIM:]])
            sd = np.nanstd(dsla[imc[LIM:]])
        return bias*1000.,sd*1000.


def IMBias_relTX():
    FN1 = 'radsstat4/txj1_stat.nc'
    FN2 = 'radsstat4/j1j2_stat.nc'
    FN3 = 'radsstat4/j2j3_stat.nc'
    FN4 = 'radsstat4/j3s6_stat.nc'
    dc = 18
    btj,stj = tdm_out(FN1,MEAN=True,MINCYC=dc)
    b12,s12 = tdm_out(FN2,MEAN=True,MINCYC=dc)
    b23,s23 = tdm_out(FN3,MEAN=True,MINCYC=dc)
    bjs,sjs = tdm_out(FN4,MEAN=True,MINCYC=dc)
    imb_tj = np.copy(btj)
    imb_12 = np.copy(imb_tj)+b12
    imb_23 = np.copy(imb_12)+b23
    imb_js = np.copy(imb_23)+bjs
    dd=1
    imb_tj,imb_12,imb_23,imb_js = np.round(imb_tj,dd),np.round(imb_12,dd),np.round(imb_23,dd),np.round(imb_js,dd)
    dd=2
    sd_tj,sd_12,sd_23,sd_js = np.round(stj,dd),np.round(s12,dd),np.round(s23,dd),np.round(sjs,dd)
    
    
    ctjC,StjC = tdm_out(FN1,MEAN=False,MINCYC=dc) #MINCYC=21-dc)
    c12C,S12C = tdm_out(FN2,MEAN=False,MINCYC=dc) #MINCYC=20-dc)
    c23C,S23C = tdm_out(FN3,MEAN=False,MINCYC=dc) #MINCYC=23-dc)
    cjsC,SjsC = tdm_out(FN4,MEAN=False,MINCYC=dc) #MINCYC=51-dc)
    
    ctj,Stj = tdm_out(FN1,MEAN=False)
    c12,S12 = tdm_out(FN2,MEAN=False)
    c23,S23 = tdm_out(FN3,MEAN=False)
    cjs,Sjs = tdm_out(FN4,MEAN=False)
    
    from matplotlib import pyplot as plt
    colors = ['tomato','dodgerblue','goldenrod','green','darkviolet']
    dp = 10
    fig=plt.figure(figsize=(10, 6))
    plt.subplots_adjust(top=0.8,bottom=0.13,hspace=0.4)
    plt.suptitle('Intermission bias estimation \n red = full tandem phase \n blue = cycles used to find IMB')
    plt.subplot(221)
    plt.plot(ctj,Stj,'.-',color=colors[0])
    plt.plot(ctjC,StjC,'.-',color=colors[1])
    plt.ylim(btj-dp,btj+dp)
    plt.grid()
    plt.title('TX/J1. IMB = '+str(imb_tj)+'$\pm$'+str(sd_tj)+' mm')
    plt.xlabel('Cycle number of successor')
    plt.ylabel('dMSL [mm]')
    plt.subplot(222)
    plt.plot(c12,S12,'.-',color=colors[0])
    plt.plot(c12C,S12C,'.-',color=colors[1])
    plt.ylim(b12-dp,b12+dp)
    plt.grid()
    plt.title('J1/J2. IMB = '+str(imb_12)+'$\pm$'+str(sd_12)+' mm')
    plt.xlabel('Cycle number of successor')
    plt.ylabel('dMSL [mm]')
    plt.subplot(223)
    plt.plot(c23,S23,'.-',color=colors[0])
    plt.plot(c23C,S23C,'.-',color=colors[1])
    plt.ylim(b23-dp,b23+dp)
    plt.grid()
    plt.title('J2/J3. IMB = '+str(imb_23)+'$\pm$'+str(sd_23)+' mm')
    plt.xlabel('Cycle number of successor')
    plt.ylabel('dMSL [mm]')
    plt.subplot(224)
    plt.plot(cjs,Sjs,'.-',color=colors[0])
    plt.plot(cjsC,SjsC,'.-',color=colors[1])
    plt.ylim(bjs-dp,bjs+dp)
    plt.grid()
    plt.title('J3/S6A. IMB = '+str(imb_js)+'$\pm$'+str(sd_js)+' mm')
    plt.xlabel('Cycle number of successor')
    plt.ylabel('dMSL [mm]')
    fig.savefig('validation/gmsl_2022rel3_rads_col_diff.png')
    return imb_tj,imb_12,imb_23,imb_js

def biases():
    # previous: im_biases = [88.4,-13.9,-21.2,-11.0] # [mm] un-lat-weighted collinear, radius=3000 m
    im_biases = [86.7, -14.2, -21.8, -11.5] #[86.5, -14.6, -22.1, -11.4] # Biases for 2022 
    # To compute new biases, run;
    #xsim_biases= IMBias_relTX()
    return im_biases

def append_ts(MNS,TDM=False):
    FN1 = 'radsstat4/tx_stat.nc'
    FN2 = 'radsstat4/j1_stat.nc'
    FN3 = 'radsstat4/j2_stat.nc'
    FN4 = 'radsstat4/j3_stat.nc'
    FN5 = 'radsstat4/6a_stat.nc'
    nr_min = 300000.0
    bias4 = biases()
    print('biases [tx-j1, j1-j2, j2-j3, j3-s6a: '+str(bias4))
    sample_cutoff = 0
    for MN in MNS:
        if MN == 'tx':
            # TX must be defined first to set arrays
            t = []
            sla_noC = []
            tag = []
            nr = []
            cog = []
            cal1 = []
            ku = []
            cycle = []
            # Call Topex data
            ds = xr.open_dataset(FN1,decode_times=False)
            ds_temp = xr.open_dataset('extras/tx_CoG.nc')
            tx_cog = (ds_temp.sla.data[:,0] - ds_temp.sla_nocg.data[:,0])
            cyc_cog = ds_temp.cycle.data
            ds_temp = xr.open_dataset('extras/tx_ptr_cal1.nc')
            cal1_array = ds_temp.drange_cal1.data[:,0]
            ku_array = ds_temp.drange_ku.data[:,0]
            cyc_ptr = ds_temp.cycle.data
            if TDM==False:
                cyclim = [1,343] # J1 1 = TX 344 (TX = J1+343)
            else:
                cyclim=[1,364]
        elif MN == 'j1':
            ds = xr.open_dataset(FN2,decode_times=False)
            TAG=1
            bias = bias4[0]
            if TDM==False:
                cyclim=[1,239] # J2 1 = J1 240 (J1 = J2+239)
            else:
                cyclim=[1,259]
        elif MN == 'j2':
            ds = xr.open_dataset(FN3,decode_times=False)
            TAG=2
            bias = bias4[1]
            if TDM==False:
                cyclim=[1,284] # J3 1 = J2 281 (J2 = J3+280)
            else:
                cyclim=[1,303]
        elif MN == 'j3':
            ds = xr.open_dataset(FN4,decode_times=False)
            TAG=3
            bias = bias4[2]
            if TDM==False:
                cyclim = [4,227] #190 (S6 = J3+175)
            else:
                cyclim=[1,227]
        elif MN== 's6a': # S6 considered reference orbit starting at cycle 52
            ds = xr.open_dataset(FN5,decode_times=False)
            TAG=4
            bias = bias4[3]
            if TDM==False:
                cyclim = [52,400]
            else:
                cyclim=[1,400]
        else:
            raise('Mission not used in timeseries')
        print('MISSION --> '+MN)
        for i in range(0,len(ds.time.data)):
            if ds.nr.data[i] < sample_cutoff: continue

            SELECT = ds.cycle.data[i]>=cyclim[0] and ds.cycle.data[i]<=cyclim[1]
            if SELECT == True and ds.nr.data[i]>=nr_min:
                print(ds)
                time = get_date_precise(ds.time.data[i])
                t.append(time)
                cycle.append(ds.cycle.data[i])
                nr.append(ds.nr.data[i])
                if MN == 'tx':
                    t0 = 1999 + (40/365.25)
                    if  time <= t0:
                        sla_noC.append((1000.*ds.sla_gia.data[i,0]) - 1.9) #!!! sla or sla_gia
                        tag.append(-1)
                    else:
                        sla_noC.append((1000.*ds.sla_gia.data[i,0])) #!!! sla or sla_gia
                        tag.append(0)
                    cog.append(1000.*tx_cog[cyc_cog == ds.cycle.data[i]][0])
                    cal1.append(1000.*cal1_array[cyc_ptr == ds.cycle.data[i]][0])
                    ku.append(1000.*ku_array[cyc_ptr == ds.cycle.data[i]][0])
                else:
                    cog.append(0)
                    cal1.append(0)
                    ku.append(0)
                    sla_noC.append((1000.*ds.sla_gia.data[i,0]) - bias) #!!! sla or sla_gia
                    tag.append(TAG)
    t = np.asarray(t)
    tag = np.asarray(tag)
    cycle = np.asarray(cycle)
    nr = np.asarray(nr)
    cog = np.asarray(cog)
    cal1 = np.asarray(cal1)
    ku = np.asarray(ku)
    sla = np.asarray(sla_noC) - cog - cal1 + ku
    rate,accel = get_rate_accel_w_seasons(t,sla)
    print(np.nanmean(cog))
    print(np.nanmean(cal1))
    print(np.nanmean(ku))
    print(np.nanmean(np.asarray(sla_noC)))
    print(np.round(rate,5),np.round(accel,5))
    return t,cycle,nr,sla,tag,bias4

def get_date_precise(sec_since_1985):
    yr = int(sec_since_1985/(60*60*24*365.24))
    time = 1985
    for i in range(0,yr):
        time += 1
        if ((1985+i)%4) != 0:
            sec_since_1985 -= (60*60*24*365)
        else:
            sec_since_1985 -= (60*60*24*366)
    
    if ((1985+yr)%4) != 0:
        time += (sec_since_1985/(60*60*24*365))
    else: 
        time += (sec_since_1985/(60*60*24*366))
#    print(time)
    return time

def smooth(t,gmsl,window=7):
    buff = int(np.floor(window/2))
    gmsl_smooth = np.zeros(len(t))
    for i in range(0,len(t)):
        gmsl_smooth[i] = np.nanmean(gmsl[i-buff:i+buff+1])
    return gmsl_smooth

def lstsqrs(t,timeseries,FULL=7,ERR=[]):
    if FULL==7:
        G2 = np.ones((len(t),FULL))
        G2[:,1] = t # !!!!!! G2[:,1] = 0;
        G2[:,2] = 0.5*(t**2) # !!!!! G2[:,2] = 0 #t**2
        G2[:,3] = np.sin(2*t*np.pi)
        G2[:,4] = np.cos(2*t*np.pi)
        G2[:,5] = np.sin(4*t*np.pi)
        G2[:,6] = np.cos(4*t*np.pi)
    elif FULL==5:
        G2 = np.ones((len(t),FULL))
        G2[:,1] = t # !!!!!! G2[:,1] = 0;
        G2[:,2] = 0.5*(t**2) # !!!!! G2[:,2] = 0 #t**2
        G2[:,3] = np.sin(2*t*np.pi)
        G2[:,4] = np.cos(2*t*np.pi)
    elif FULL==3:
        G2 = np.ones((len(t),FULL))
        G2[:,1] = t # !!!!!! G2[:,1] = 0;
        G2[:,2] = 0.5*(t**2) # !!!!! G2[:,2] = 0 #t**2
    elif FULL==2:
        G2 = np.ones((len(t),FULL))
        G2[:,1] = t # !!!!!! G2[:,1] = 0;
    if np.size(ERR)==0:
        G2i = np.linalg.inv(G2.T.dot(G2))
        C_trend = G2i.dot(G2.T.dot(timeseries))#np.matmul(np.linalg.pinv(G2),timeseries)
        c = np.asarray(C_trend)
        err = np.empty(7)*np.nan
    elif np.size(ERR)==1:
        G2i = np.linalg.inv(G2.T.dot(G2))
        C_trend = G2i.dot(G2.T.dot(timeseries))#np.matmul(np.linalg.pinv(G2),timeseries)
        c = np.asarray(C_trend)
        err = np.sqrt(ERR*np.diag(G2i))
    else:
        G2i = np.linalg.inv(G2.T.dot(G2))
        G2E = np.linalg.inv(G2.T.dot(np.diag(ERR).dot(G2))) #G2.T.dot(G2)).dot((G2.T.dot(ERR.dot(G2))).dot(np.linalg.inv(G2.T.dot(G2))))
        err = np.sqrt(np.diag(G2E))
        C_trend = G2i.dot(G2.T.dot(timeseries))#np.matmul(np.linalg.pinv(G2),timeseries)
        c = np.asarray(C_trend)
    return c,err

def reg_gmsl(lat,lon,time,sla):
    Nphi = np.size(lat)
    Nlam = np.size(lon)
    sla_ds = np.empty(np.shape(sla))*np.nan
    rate = np.empty((Nphi,Nlam))*np.nan
    accel = np.empty((Nphi,Nlam))*np.nan
    errR = np.empty((Nphi,Nlam))*np.nan
    errA = np.empty((Nphi,Nlam))*np.nan
    for ii in np.arange(Nphi):
        for jj in np.arange(Nlam):
            sla_ds[:,ii,jj],rate2,accel2,errR[ii,jj],errA[ii,jj] = deseason_reg(time,sla[:,ii,jj])
            rate[ii,jj],accel[ii,jj] = get_rate_accel_w_seasons(time,sla[:,ii,jj])
    return sla_ds,rate,accel,errR,errA


def deseason_reg(t,timeseries_orig):
    # t,timeseries_orig = time,sla[:,ii,jj]
    #timeseries_ds,rate,accel=deseason_fromEOF(t,timeseries_orig)
    inn = np.where(~np.isnan(timeseries_orig))[0]
    if np.size(t)/2.0:#np.size(t)-2:
        timeseries =  timeseries_orig[inn]#np.matrix(timeseries_orig[inn]).T
        t_nonan = t[inn]
        dt = t_nonan - np.mean(t_nonan)
        c,err = lstsqrs(t_nonan,timeseries)
        fit = c[0] + (c[1]*t_nonan) + (0.5*c[2]*t_nonan**2)
        '''
        from matplotlib import pyplot as plt
        plt.figure()
        #plt.plot(t,timeseries,label='timeseries')
        #plt.plot(t,fit,label='fit')
        plt.plot(t,fit-timeseries,label='fit-timeseries')
        plt.legend()
        '''
        mse = np.nanmean((timeseries-fit)**2)
        print('RMSE: '+str(np.round(np.sqrt(mse),3))+ 'mm')
        c_rate,err_rate = lstsqrs(dt,timeseries,ERR=mse)
        errR,errA = err_rate[1],err_rate[2]
        rate,accel = c_rate[1],c_rate[2]
        # least squares w/ only trend, offset, and seasons        
        rebuild = (c[3]*np.sin(2*t*np.pi)) + (c[4]*np.cos(2*t*np.pi)) + (c[5]*np.sin(4*t*np.pi)) + (c[6]*np.cos(4*t*np.pi))
        timeseries_ds = timeseries_orig - np.squeeze(rebuild)
    else:
        timeseries_ds,rate,accel,errR,errA = np.empty(np.shape(t))*np.nan,np.nan,np.nan,np.nan,np.nan
    return timeseries_ds,rate,accel,errR,errA

def get_rate_accel_w_seasons(t,timeseries):
    t = t-np.mean(t)
    # f = a + bt + dsin() + ecos() + fsin() + gcos()
    # least squares w/ only trend, offset, and seasons
    c,err = lstsqrs(t,timeseries,FULL=7)
    return c[1],c[2]


def fit_curve(t,gmsl,string):
    p = np.polyfit(t,gmsl,2)
    #print(string+' quadratic coefficients: [c,b,a]',p*2)
    return np.polyval(p,t)

