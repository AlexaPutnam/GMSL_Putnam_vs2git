#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:13:00 2022

@author: alexaputnam
"""
import sys
import numpy as np
from matplotlib import pyplot as plt

import lib_timeseries as lts
import lib_external as lex

LOCDIR = '/Users/alexaputnam/GMSL/web_gmsl/'

#################################################################
#################################################################
#################################################################
# User input
# List all missions that should be included in the GMSL trend map
MNS = ['tx','j1','j2','j3','s6a']
# Indicate whether to display the validation results (True = display)
VALIDATE = True


#################################################################
#################################################################
#################################################################
# plotting functions
def web_plot(t,gmsl,gmsl_smooth,tag,RATE,RATE_ERR,ACCEL,ACCEL_ERR,S6A,DESEASON=True):
    ##################################################
    # Assign tags
    cond1 = np.logical_or(tag == -1,tag == 0)
    cond2 = tag == 1
    cond3 = tag == 2
    cond4 = tag == 3
    cond5 = tag == 4
    ##################################################
    ## trend segment colots
    colors = ['tomato','dodgerblue','goldenrod','green','darkviolet']
    ##################################################
    ## Create figure
    plt.figure(figsize=(8.5,5.5))
    ax = plt.subplot(1,1,1)
    p1 = ax.scatter(t[cond1],gmsl[cond1],c=colors[0],marker = 'o',s = 15,zorder = 10,label='TOPEX')
    p2 = ax.scatter(t[cond2],gmsl[cond2],c=colors[1],marker = 'o',s = 15,zorder = 10,label='Jason-1')
    p3 = ax.scatter(t[cond3],gmsl[cond3],c=colors[2],marker = 'o',s = 15,zorder = 10,label='Jason-2')
    p4 = ax.scatter(t[cond4],gmsl[cond4],c=colors[3],marker = 'o',s = 15,zorder = 10,label='Jason-3')
    if S6A is True:
        p5 = ax.scatter(t[cond5],gmsl[cond5],c=colors[4],marker = 'o',s = 15,zorder = 10,label='Sentinel-6 MF')
    p6, = ax.plot(t,gmsl_smooth,'.25',linewidth = 2.5,zorder=11,label = '60-day smoothed')
    p7, = ax.plot(t,lts.fit_curve(t,gmsl,'sig_removed'),linewidth = 2,c='mediumblue',zorder = 12, \
                                label = 'Trend: '+RATE+' $\pm$ '+RATE_ERR+' mm/y')
    p8, = ax.plot(t,gmsl,alpha = 0,linewidth = 0,label=  'Acceleration: '+ACCEL+' $\pm$ '+ACCEL_ERR+' mm/y$^2$')
    p9, = ax.plot(t,gmsl,alpha = 0,linewidth = 0,label=  'Acceleration: '+ACCEL+' $\pm$ '+ACCEL_ERR+' mm/y$^2$')
    if S6A is True:
        leg = ax.legend((p1,p2,p3,p4,p5,p6,p7,p8,p9), \
                  ('TOPEX','Jason-1','Jason-2','Jason-3','Sentinel-6 MF','60-day smoothed','Quadratic Fit','Average Rate: '+RATE+' $\pm$ '+RATE_ERR+' mm/y','Acceleration: '+ACCEL+' $\pm$ '+ACCEL_ERR+' mm/y$^2$'),
                  frameon=True,fontsize='medium')
    else:
        leg = ax.legend((p1,p2,p3,p4,p6,p7,p8,p9), \
                  ('TOPEX','Jason-1','Jason-2','Jason-3','60-day smoothed','Quadratic Fit','Average Rate: '+RATE+' $\pm$ '+RATE_ERR+' mm/y','Acceleration: '+ACCEL+' $\pm$ '+ACCEL_ERR+' mm/y$^2$'),
                  frameon=True,fontsize='medium')
    ax.set_ylabel('MSL [mm]',fontsize=13,fontweight = 'bold')
    ax.set_xlabel('Year',fontsize=13,fontweight='bold')
    ax.set_facecolor('.88')
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='1.75', color='white')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color='white')
    
    leg.get_frame().set_edgecolor('.4')
    leg.get_frame().set_linewidth(1)
    
    if DESEASON is True:
        ax.annotate('University of Colorado 2022_rel1\nSeasonal Signals Removed',(2022,-40),fontsize=8,fontweight='bold',ha='right')
        plt.savefig('output_4_website/gmsl_2022rel1_seasons_rmvd.pdf',dpi=400)
        plt.savefig('output_4_website/gmsl_2022rel1_seasons_rmvd.eps',dpi=400)
        plt.savefig('output_4_website/gmsl_2022rel1_seasons_rmvd.png',dpi=400)
    elif DESEASON is False:
        ax.annotate('University of Colorado 2022_rel1\nSeasonal Signals Retained',(2022,-42),fontsize=8,fontweight='bold',ha='right')
        plt.savefig('output_4_website/gmsl_2022rel1_seasons_retained.pdf',dpi=400)
        plt.savefig('output_4_website/gmsl_2022rel1_seasons_retained.eps',dpi=400)
        plt.savefig('output_4_website/gmsl_2022rel1_seasons_retained.png',dpi=400)
    
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.show()

def data2web(t,gmsl,tag,c_file,im_biases,nr):
    if 4 in tag:
        S6A = True
    else:
        S6A = False
    gmsl_ds_nr,rate,accel,errR,errA = lts.deseason_reg(t,gmsl)
    gmsl_ds = np.round(gmsl_ds_nr,3)
    gmsl_smooth = lts.smooth(t,gmsl,window=7)
    gmsl_ds_smooth = lts.smooth(t,gmsl_ds,window=7)
    
    ##################################################
    # Print rate and acceleration
    if S6A is True:
        IN = 'with'
    else:
        IN = 'without'
    print('----------------------------------')
    print('Rate '+IN+' s6: '+str(np.round(rate,1))+' +/- '+str(np.round(errR,2))+' mm/y')
    print('Acceleration '+IN+' s6: '+str(np.round(accel,3))+' +/- '+str(np.round(errA,4))+' mm/y2')
    print('----------------------------------')
    ##################################################
    # Save Text Files
    ## Figure without seasonal signals
    file1 = open('output_4_website/gmsl_2022rel1_seasons_rmvd.txt','w')
    file1.write('# Date      2022_rel1 w/ seasonal signals and GIA removed (mm)')
    file1.write('\n')
    for i in range(0,len(t)):
        file1.write(str(t[i]) + '      ' + str(gmsl_ds[i])+'\n')
    file1.close()
    
    ## Figure with seasonal signals
    file2 = open('output_4_website/gmsl_2022rel1_seasons_retained.txt','w')
    file2.write('# Date      2022_rel1 w/ GIA removed (mm)')
    file2.write('\n')
    for i in range(0,len(t)):
        file2.write(str(t[i]) + '      ' + str(np.round(gmsl[i],2))+'\n')
    file2.close()
    
    ##################################################
    ## Set values for figures
    RATE = str(np.round(rate,1))#'3.3'
    RATE_ERR = '0.4'
    ACCEL = str(np.round(accel,3))#'0.099' #0.097 with no seasons?
    ACCEL_ERR = '0.025'
    
    web_plot(t,gmsl,gmsl_smooth,tag,RATE,RATE_ERR,ACCEL,ACCEL_ERR,S6A,DESEASON=False)
    web_plot(t,gmsl_ds,gmsl_ds_smooth,tag,RATE,RATE_ERR,ACCEL,ACCEL_ERR,S6A,DESEASON=True)
    
    
def validate_cu_noaa_goddard(FN):
    file = np.load('radsstat4/gmsl_rads_appended.npz')
    t,gmsl,tag,c_file = file.f.t,file.f.gmsl,file.f.tag,file.f.cycle
    im_biases = file.f.biases
    t_c = np.round(t,3)
    
    gmsl_ds_nr,rate_c,acc_c,errR,errA = lts.deseason_reg(t,gmsl)
    gmsl_ds = np.round(gmsl_ds_nr,3)
    #gmsl_smooth = lts.smooth(t,gmsl,window=7)
    gmsl_ds_smooth = lts.smooth(t,gmsl_ds,window=7)

    t_g,gmsl_g,rate_g,acc_g = lex.gmsl_goddard()
    t_n,gmsl_n,rate_n,acc_n = lex.gmsl_noaa()
    TIT_g = 'NASA Rate: '+str(rate_g)+' mm/yr, Acceleration: '+str(acc_g)+' mm/yr$^2$, Website: 3.4 +/- 0.4 mm/yr'
    TIT_n = 'NOAA Rate: '+str(rate_n)+' mm/yr, Acceleration: '+str(acc_n)+' mm/yr$^2$, Website: 3.0 +/- 0.4 mm/yr'
    TIT_c = 'CU Rate: '+str(np.round(rate_c,1))+' mm/yr, Acceleration: '+str(np.round(acc_c,3))+' mm/yr$^2$, Website: 3.3 +/- 0.4 mm/yr'
    lev_g = np.round(gmsl_g[~np.isnan(gmsl_g)][0]-gmsl_ds_smooth[~np.isnan(gmsl_ds_smooth)][0],1)
    lev_n = np.round(gmsl_n[~np.isnan(gmsl_n)][0]-gmsl_ds_smooth[~np.isnan(gmsl_ds_smooth)][0],1)
    print(TIT_c)
    fig=plt.figure(figsize=(10,8))
    plt.subplots_adjust(top=0.80,bottom=0.13,hspace=0.4)
    plt.title(TIT_g+'\n'+TIT_n+'\n'+TIT_c)
    plt.plot(t_g,gmsl_g-lev_g,'-',label='NASA (level bias removed = '+str(lev_g)+' mm)',linewidth=3)
    plt.plot(t_n,gmsl_n-lev_n,'-',label='NOAA with GIA correction applied (level bias removed = '+str(lev_n)+' mm)',linewidth=3,alpha=0.7)
    plt.plot(t_c,gmsl_ds_smooth,'-',label='CU',linewidth=3,alpha=0.5)
    plt.xlabel('years')
    plt.ylabel('gmsl [mm]')
    plt.grid()
    plt.legend()
    fig.savefig(FN)

def find_overlap(t,tag,tg1,tg2):
    if tg1==0:
        i1t = np.where(tag<=tg1)[0]
    else:
        i1t = np.where(tag==tg1)[0]
    i2t = np.where(tag==tg2)[0]
    i1c = np.where(t[i1t]>=np.nanmin(t[i2t]))[0]
    i2c = np.where(t[i2t]<=np.nanmax(t[i1t]))[0]
    return i1t[i1c],i2t[i2c]
    
    
def validate_tandem_match(t,sla,tag,FN):
    biases = lts.biases()
    
    #t0 = 1999 + (30/365.25)#(40/365.25)
    t1 = 2002 +  (1/365.25)#(11/365.25)
    t2 = 2008 + (177/365.25)#(187/365.25)
    t3 = 2016 + (37/365.25)#(47/365.25)
    t4 = 2020 + (320/365.25)#(180/365.25)
    
    colors = ['tomato','dodgerblue','goldenrod','green','darkviolet']
    
    itjA,itjB = find_overlap(t,tag,0,1)
    i12A,i12B = find_overlap(t,tag,1,2)
    i23A,i23B = find_overlap(t,tag,2,3)
    ijsA,ijsB = find_overlap(t,tag,3,4)

    
    fig=plt.figure(figsize=(10, 6))
    plt.subplots_adjust(top=0.9,bottom=0.13,hspace=0.4)
    plt.suptitle('Zoom-in on tandem region')
    plt.subplot(221)
    plt.plot((t[itjA]-t[itjA][0])*365.25,sla[itjA],color=colors[0],label='TOPEX')
    plt.plot((t[itjB]-t[itjA][0])*365.25,sla[itjB],color=colors[1],label='Jason-1')
    plt.axvline(x=((t[itjA]-t[itjA][0])*365.25)[-1]-180,color='black')
    plt.grid()
    plt.title('TX/J1. TDM bias = '+str(biases[0])+' mm')
    plt.xlabel('Number of tandem days in zoomed area')
    plt.ylabel('MSL [mm]')
    plt.legend()
    plt.subplot(222)
    plt.plot((t[i12A]-t[i12A][0])*365.25,sla[i12A],color=colors[1],label='Jason-1')
    plt.plot((t[i12B]-t[i12A][0])*365.25,sla[i12B],color=colors[2],label='Jason-2')
    plt.axvline(x=((t[i12A]-t[i12A][0])*365.25)[-1]-180,color='black')
    plt.grid()
    plt.title('J1/J2. TDM bias = '+str(biases[1])+' mm')
    plt.xlabel('Number of tandem days in zoomed area')
    plt.ylabel('MSL [mm]')
    plt.legend()
    plt.subplot(223)
    plt.plot((t[i23A]-t[i23A][0])*365.25,sla[i23A],color=colors[2],label='Jason-2')
    plt.plot((t[i23B]-t[i23A][0])*365.25,sla[i23B],color=colors[3],label='Jason-3')
    plt.axvline(x=((t[i23A]-t[i23A][0])*365.25)[-1]-180,color='black')
    plt.grid()
    plt.title('J2/J3. TDM bias = '+str(biases[2])+' mm')
    plt.xlabel('Number of tandem days in zoomed area')
    plt.ylabel('MSL [mm]')
    plt.legend()
    plt.subplot(224)
    plt.plot((t[ijsA]-t[ijsA][0])*365.25,sla[ijsA],color=colors[3],label='Jason-3')
    plt.plot((t[ijsB]-t[ijsA][0])*365.25,sla[ijsB],color=colors[4],label='Sentinel-6 MF')
    plt.axvline(x=((t[ijsA]-t[ijsA][0])*365.25)[-1]-180,color='black')
    plt.grid()
    plt.title('J3/S6A. TDM bias = '+str(biases[3])+' mm')
    plt.xlabel('Number of tandem days in zoomed area')
    plt.ylabel('MSL [mm]')
    plt.legend()
    fig.savefig(FN)
    

#################################################################
#################################################################
#################################################################
## Make timeseries for website
t,cycle,nr,sla,tag,bias4 = lts.append_ts(MNS,TDM=False)
np.savez(LOCDIR+'radsstat4/gmsl_rads_appended.npz', t=t,gmsl=sla,tag=tag,cycle=cycle,biases=bias4,nr=nr)
## Make timeseries for tandem analysis (validation)
t2,cycle2,nr2,sla2,tag2,bias42 = lts.append_ts(MNS,TDM=True)
np.savez(LOCDIR+'radsstat4/gmsl_rads_appended_tandem.npz', t=t2,gmsl=sla2,tag=tag2,cycle=cycle2,biases=bias42,nr=nr2)


#################################################################
#################################################################
#################################################################
# Analyze timeseries
file = np.load(LOCDIR+'radsstat4/gmsl_rads_appended.npz')
t,gmsl,tag,c_file = file.f.t,file.f.gmsl,file.f.tag,file.f.cycle
im_biases,nr = file.f.biases,file.f.nr
t = np.round(t,3)
data2web(t,gmsl,tag,c_file,im_biases,nr)


#################################################################
#################################################################
#################################################################
## Validate
if VALIDATE == True:
    
    file = np.load(LOCDIR+'radsstat4/gmsl_rads_appended_tandem.npz')
    t,gmsl,tag,c_file = file.f.t,file.f.gmsl,file.f.tag,file.f.cycle
    im_biases,nr = file.f.biases,file.f.nr
    t = np.round(t,3)

    inn = np.where((~np.isnan(gmsl))&(t<=2021))[0]
    timeseries =  gmsl[inn]#np.matrix(timeseries_orig[inn]).T
    t_nonan = t[inn]
    dt = t_nonan - np.mean(t_nonan)
    c2,err2 = lts.lstsqrs(t_nonan,timeseries,FULL=2)
    c5,err5 = lts.lstsqrs(t_nonan,timeseries,FULL=5)
    fit2 = c2[0] + (c2[1]*t_nonan) 
    rebuild5 = c5[0] + (c5[1]*t_nonan) + (0.5*c5[2]*t_nonan**2)+(c5[3]*np.sin(2*t_nonan*np.pi)) + (c5[4]*np.cos(2*t_nonan*np.pi))#+(c5[5]*np.sin(4*t_nonan*np.pi)) + (c5[6]*np.cos(4*t_nonan*np.pi))

    r2 = 1-(np.var(timeseries-fit2)/np.var(timeseries))
    r5 = 1-(np.var(timeseries-rebuild5)/np.var(timeseries))

    plt.figure()
    plt.plot(t_nonan,timeseries,'-',label='data',color='grey')
    plt.plot(t_nonan,fit2,'-',label='fit',color='darkorange')
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel('Mean Sea Level [mm]')
    plt.title('Global mean sea level \n linear regression')
    plt.legend()

    plt.figure()
    plt.plot(t_nonan,timeseries-fit2,'.',label='data - fit',color='darkorange')
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel('Residuals [mm]')
    plt.title('Global mean sea level \n residuals from linear regression')
    plt.legend()
    plt.ylim(-20,20)

    plt.figure()
    plt.plot(t_nonan,timeseries,'-',label='data',color='grey')
    plt.plot(t_nonan,rebuild5,'-',label='fit',color='seagreen')
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel('Mean Sea Level [mm]')
    plt.title('Global mean sea level \n multivariate regression')
    plt.legend()

    plt.figure()
    plt.plot(t_nonan,timeseries-rebuild5,'.',label='data - fit',color='seagreen')
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel('Residuals [mm]')
    plt.title('Global mean sea level \n residuals from multivariate regression')
    plt.legend()
    plt.ylim(-20,20)

    plt.figure()
    plt.hist(timeseries-fit2,label='data - fit(linear)',color='darkorange')
    plt.hist(timeseries-rebuild5,label='data - fit(multivariate)',color='seagreen')
    plt.grid()
    plt.xlabel('Occurences')
    plt.xlabel('Residuals [mm]')
    plt.title('Global mean sea level \n histogram of residuals')
    plt.legend()



    gmsl_ds_nr,rate,accel,errR,errA = lts.deseason_reg(t,gmsl)
    gmsl_ds = np.round(gmsl_ds_nr,3)
    gmsl_smooth = lts.smooth(t,gmsl,window=7)
    gmsl_ds_smooth = lts.smooth(t,gmsl_ds,window=7)
    
    validate_cu_noaa_goddard(LOCDIR+'validation/gmsl_2022rel1_seasons_retained_external_comparison.png')
    validate_tandem_match(t,gmsl,tag,LOCDIR+'validation/gmsl_2022rel1_seasons_retained_tandem_phase.png')

    
'''

    
plt.figure(figsize=(15,4))
plt.scatter(t,gmsl_ds_smooth,c=nr/100000,cmap='jet',vmin=3,vmax=6)
plt.title('GMSL timeseries and number of observations')
cb=plt.colorbar()

plt.figure(figsize=(15,4))
plt.scatter(t,gmsl_ds_smooth,c=nr/100000,cmap='jet',vmin=3,vmax=6)
plt.title('measurements per cycle / 100000')
plt.colorbar()

plt.figure(figsize=(6,3))
plt.title('Time step between data points')
plt.plot(t[:-1],np.diff(t*365.25),'k')
plt.grid()
plt.ylabel('[days]')
plt.savefig('time_steps.png',dpi=200)
plt.show()

'''

'''
plt.figure(figsize=(6,3))
cond = np.logical_and(t<=2016,t>=1996)
plt.title('GMSL')
plt.plot(t[cond],gmsl_ds[cond],'k')
plt.grid()
plt.ylabel('[days]')
plt.savefig('gmsl_ltd.png',dpi=200)
plt.show()
'''

