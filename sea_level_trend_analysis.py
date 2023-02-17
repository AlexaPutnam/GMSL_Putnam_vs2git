import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from glob import glob
from scipy.interpolate import griddata


from scipy.interpolate import NearestNDInterpolator as nn_interp
from scipy.ndimage import uniform_filter
import cartopy.crs as ccrs
import cartopy.feature as ft
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import netCDF4 as nc
from matplotlib import colors

import lib_external as lex


###########################################################################  
## Functions
###########################################################################
def conv_datetime(dtime):
    temp = str(dtime)
    
    yr = int(temp[0:4])
    month = int(temp[5:7])
    day = int(temp[8:10])
    return yr + ((month-1)/12) + (day/365.25)
    
def deseason(t,timeseries):
    timeseries = np.matrix(timeseries).T
    G2 = np.empty((len(t),6))
    G2[:,0] = 1                   # constant
    G2[:,1] = t                   # trend
    G2[:,2] = np.sin(2*t*np.pi)   # sin annual
    G2[:,3] = np.cos(2*t*np.pi)   # cos annual
    G2[:,4] = np.sin(4*t*np.pi)   # sin semiannual
    G2[:,5] = np.cos(4*t*np.pi)   # cos semiannual
    # least squares
    c = np.asarray(np.matmul(np.linalg.pinv(np.matrix(G2)),timeseries))
    # rebuild
    rebuild = (c[2]*np.sin(2*t*np.pi)) \
        + (c[3]*np.cos(2*t*np.pi)) \
        + (c[4]*np.sin(4*t*np.pi)) \
        + (c[5]*np.cos(4*t*np.pi))
    return np.squeeze(np.asarray(timeseries)) - np.squeeze(rebuild)

def mask_out_land(xref_mesh,yref_mesh):
    ### make mask using elev.1-deg.nc to remove land###
    ds = xr.open_dataset('extras/elev.1-deg.nc',decode_times=False)
    mask_vals = ds.data.data.squeeze()
    mask_lon = ds.lon.data
    mask_lat = ds.lat.data
    mlonmesh,mlatmesh = np.meshgrid(mask_lon,mask_lat)
    # flatten masks
    mask_vals_flat = np.reshape(mask_vals,np.size(mask_vals))
    mask_lon_flat = np.reshape(mlonmesh,np.size(mlonmesh))
    mask_lat_flat = np.reshape(mlatmesh,np.size(mlatmesh))
    # Make sure data is centered at 0
    mask_lon_flat[mask_lon_flat >= 180] -= 360
    # Set NaN to all land 
    mask = griddata((mask_lon_flat,mask_lat_flat),mask_vals_flat,(xref_mesh,yref_mesh),'linear')
    mask[mask >= 0] = np.NaN
    mask[np.isfinite(mask)] = 1
    return mask

def mask_out_land_4plot(xref_mesh,yref_mesh):
    ### make mask using elev.1-deg.nc to remove land###
    ds = xr.open_dataset('extras/elev.1-deg.nc',decode_times=False)
    mask_vals = ds.data.data.squeeze()
    mask_lon = ds.lon.data
    mask_lat = ds.lat.data
    mlonmesh,mlatmesh = np.meshgrid(mask_lon,mask_lat)
    # flatten masks
    mask_vals_flat = np.reshape(mask_vals,np.size(mask_vals))
    mask_lon_flat = np.reshape(mlonmesh,np.size(mlonmesh))
    mask_lat_flat = np.reshape(mlatmesh,np.size(mlatmesh))
    # Make sure data is centered at 0
    # Set NaN to all land 
    mask = griddata((mask_lon_flat,mask_lat_flat),mask_vals_flat,(xref_mesh,yref_mesh),'linear')
    mask[mask >= 0] = np.NaN
    mask[np.isfinite(mask)] = 1
    mask[np.abs(yref)>66,:] = np.NaN
    return mask


def create_spatial_timeseries(filenames):
    from netCDF4 import Dataset
    Nf = np.shape(filenames)[0]
    ds_temp = xr.open_dataset('grids4trend/tx_grids_15.nc')
    xref = ds_temp.lon.data
    yref = ds_temp.lat.data
    xref_mesh,yref_mesh = np.meshgrid(xref,yref)
    ### Create land mask
    mask = mask_out_land(xref_mesh,yref_mesh)
    # skip these cycles
    skip_name1 = 'grids4trend/j2_grids_174.nc'
    skip_name2 = 'grids4trend/j2_grids_175.nc'
    skip_name3 = 'grids4trend/j3_grids_197.nc'
    # find gridded timeseries
    mission_temp = []
    cycle_temp = []
    t = np.zeros(Nf)*np.NaN
    sla_temp = np.zeros((Nf,len(ds_temp.lat.data),len(ds_temp.lon.data)))
    for ii in range(0,Nf):
        if ii%25 == 0: 
            print(np.round(100*ii/Nf,2))
        # open SL file
        ds = Dataset(filenames[ii])
        ds = xr.open_dataset(filenames[ii])
        igd = np.where(~np.isnan(ds.sla_mean.data))
        # make sure file is valid
        if np.isnan(np.nanmin(ds.time_mean.data)) or np.size(igd)==0: 
            mission_temp.append(999)
            cycle_temp.append(999)
            continue
        # compute time
        tmin = conv_datetime(np.nanmin(ds.time_mean.data[igd]))
        tmax = conv_datetime(np.nanmax(ds.time_mean.data[igd]))
        t[ii] = (tmin + tmax)/2
        # save mission name and cycle number
        mission_temp.append(ds.title[12:14])
        cycle_temp.append(int(ds.title[21:-3]))
        # mIf the file needs to be skipped then set to NaN
        if np.logical_or(np.logical_or(filenames[ii] == skip_name1,filenames[ii] == skip_name2),filenames[ii]==skip_name3): #######
            sla_temp[ii,:,:] = np.NaN                                             #######
            xr.Dataset.close(ds)                                                 #######
            continue                                                             #######
        # remove GIA and save SLA
        sla_temp[ii,:,:] = mask*(ds.sla_mean.data.T)# + ds.gia_mean.data.T)
        xr.Dataset.close(ds)
    # Make sure the data is in chronological order
    args = np.argsort(t)
    timeT = t[args]
    timeT[timeT<1992]=np.nan
    sla = sla_temp[args,:,:]
    mission = np.asarray(mission_temp)[args]
    cycle = np.asarray(cycle_temp)[args]
    # Remove all infinate numbers
    cond = np.isfinite(timeT)
    cycle = cycle[cond]
    mission = mission[cond]
    sla = sla[cond,:,:]
    timeT = timeT[cond]
    # Remove intermission biases [86.7, -14.2, -21.8, -11.5] #
    sla[timeT < (1999 + (40/365.25))] -= 1.9/1000
    sla[mission == 'j1'] -= 86.7/1000
    sla[mission == 'j2'] -= -14.2/1000
    sla[mission == 'j3'] -= -21.8/1000
    sla[mission == '6a'] -= -11.5/1000
    return cycle,mission,sla,timeT,xref,yref

def txA_corrections(cycle,mission,sla):
    # CoG
    ds_temp = xr.open_dataset('extras/tx_CoG.nc')
    tx_cog = (ds_temp.sla.data[:,0] - ds_temp.sla_nocg.data[:,0])
    cyc_cog = ds_temp.cycle.data
    # CAL1
    ds_temp = xr.open_dataset('extras/tx_ptr_cal1.nc')
    cal1_array = ds_temp.drange_cal1.data[:,0]
    ku_array = ds_temp.drange_ku.data[:,0]
    cyc_ptr = ds_temp.cycle.data
    # Corrected SLA for Topex-SideA only
    for ii in range(0,len(cycle)):
        if mission[ii] != 'tx': continue
        if np.sum(cycle[ii] == cyc_cog) == 0:
            sla[ii,:,:] = np.NaN
            continue
        if np.sum(cycle[ii] == cyc_ptr) == 0:
            sla[ii,:,:] = np.NaN
            continue
        sla[ii,:,:] += (-tx_cog[cyc_cog==cycle[ii]] - cal1_array[cyc_ptr == cycle[ii]] + ku_array[cyc_ptr == cycle[ii]])
    return sla

def sla2trends_gridded(xref,yref,sla,timeT,t_limit=2023):
    xref_mesh,yref_mesh = np.meshgrid(xref,yref)
    trends = np.zeros(np.shape(xref_mesh))*np.NaN
    for ii in range(0,len(yref)):
        for jj in range(0,len(xref)):
            ts = sla[:,ii,jj]*1000
            if np.sum(np.isfinite(ts)) < (.3*len(timeT[timeT<t_limit])): continue
    
            cond = np.logical_and(np.isfinite(ts),timeT<t_limit)
            ts_ds = deseason(timeT[cond],ts[cond])
            trends[ii,jj] = np.polyfit(timeT[cond],ts_ds,1)[0]
    trends_rolled = np.roll(trends,(int(len(xref)/2)+1),1)
    # test: global trend
    scale = np.cos(np.deg2rad(yref_mesh))
    #print(np.nansum(scale*trends)/np.nansum(scale[np.isfinite(trends)]))
    return trends_rolled

def smooth(x,y,data,mask,smooth,xmesh,ymesh):
    interp = nn_interp((x,y),data)
    new_map = interp(xmesh,ymesh)
    smoothed_map = uniform_filter(new_map,smooth)
    return smoothed_map*mask

def plot_sl_trends(xref,yref,rads,t,vmin=-4,vmax=8,cmap='jet',TITLE=[],clab='mm/year'):
    plt.figure(figsize=(10,6)) #(8.5, 5.4))
    ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude = 180))
    if np.size(TITLE)==0:
        plt.title('Sea Level Trends\n\n',fontsize=14)
        plt.text(.5,88.5,str(np.round(np.nanmin(t),2))+' - '+str(np.round(np.nanmax(t),2))+'\nTOPEX/Poseidon, Jason-1, Jason-2, Jason-3, Sentinel-6 Michael Freilich',fontsize=12, ha='center')
    else:
        plt.title(TITLE+'\n mean = '+str(np.round(np.nanmean(rads),3))+' '+clab+', rms = '+str(np.round(np.sqrt(np.nanmean(rads**2)),3)),fontsize=14)
    plt.text(-177,-81,'University of Colorado 2022_rel1',fontsize=9,weight = 'bold')
    plt.ylim((-85,85))
    # data
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    fig1 = ax1.pcolormesh(xref,yref,rads, cmap = cmap, zorder = 1,\
                          transform=ccrs.PlateCarree(),norm=norm)
    # features
    ax1.add_feature(ft.COASTLINE)#,zorder=3)
    ax1.add_feature(ft.LAND,facecolor = 'wheat')#,zorder=2)
    # colorbar
    cbar = plt.colorbar(fig1,orientation='horizontal',pad=.075,aspect=45,extend='both')#,pad=.075,aspect=45, shrink = .9)
    cbar.set_label(clab,weight='bold')
    # x/y axis    
    gl = ax1.gridlines(axes=fig1,crs=ccrs.PlateCarree(central_longitude = 180), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False;    gl.right_labels = False
    gl.xlines = False;         gl.ylines = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180,179,45))
    gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    # save figure
    if np.size(TITLE)==0:
        plt.savefig('output_4_website/cu_sea_level_trends.png',dpi=300,bbox_inches='tight')#,dpi=300, bbox_inches="tight",pad_inches = 0.3)
        plt.savefig('output_4_website/cu_sea_level_trends.pdf',dpi=300)
        plt.savefig('output_4_website/cu_sea_level_trends.eps',dpi=300)
    else:
        plt.savefig('validation/cu_vs_noaa_sea_level_trends.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_sl_trends_simple(xref,yref,rads,t,vmin=-4,vmax=8,cmap='jet',TITLE=[],clab='mm/year'):
    plt.figure(figsize=(10,6)) #(8.5, 5.4))
    ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude = 180))
    if np.size(TITLE)==0:
        plt.title('Sea Level Trends\n\n',fontsize=14)
        plt.text(.5,88.5,str(np.round(np.nanmin(t),2))+' - '+str(np.round(np.nanmax(t),2))+'\nTOPEX/Poseidon, Jason-1, Jason-2, Jason-3, Sentinel-6 Michael Freilich',fontsize=12, ha='center')
    else:
        plt.title(TITLE+'\n mean = '+str(np.round(np.nanmean(rads),3))+' '+clab+', rms = '+str(np.round(np.sqrt(np.nanmean(rads**2)),3)),fontsize=14)
    plt.text(-177,-81,'University of Colorado 2022',fontsize=9,weight = 'bold')
    plt.ylim((-85,85))
    # data
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    fig1 = ax1.pcolormesh(xref,yref,rads, cmap = cmap, zorder = 1,\
                          transform=ccrs.PlateCarree(),norm=norm)
    # features
    ax1.add_feature(ft.COASTLINE)#,zorder=3)
    ax1.add_feature(ft.LAND,facecolor = 'wheat')#,zorder=2)
    # colorbar
    cbar = plt.colorbar(fig1,orientation='horizontal',pad=.075,aspect=45,extend='both')#,pad=.075,aspect=45, shrink = .9)
    cbar.set_label(clab,weight='bold')
    # x/y axis    
    gl = ax1.gridlines(axes=fig1,crs=ccrs.PlateCarree(central_longitude = 180), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False;    gl.right_labels = False
    gl.xlines = False;         gl.ylines = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180,179,45))
    gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    # save figure
    if np.size(TITLE)==0:
        plt.savefig('output_4_website/cu_sea_level_trends.png',dpi=300,bbox_inches='tight')#,dpi=300, bbox_inches="tight",pad_inches = 0.3)
        plt.savefig('output_4_website/cu_sea_level_trends.pdf',dpi=300)
        plt.savefig('output_4_website/cu_sea_level_trends.eps',dpi=300)
    else:
        plt.savefig('validation/cu_vs_noaa_sea_level_trends.png',dpi=300,bbox_inches='tight')
    plt.show()

###########################################################################  
## Create gridded SL trend dataset
###########################################################################
### Create timeseries
filenames = glob('grids4trend/*.nc')
cycle,mission,sla_ucTxA,timeT,xref,yref = create_spatial_timeseries(filenames)
### Define map parameters
xref_mesh,yref_mesh = np.meshgrid(xref,yref)
### Create land mask
mask = mask_out_land(xref_mesh,yref_mesh)
### Apply Topex-sideA corrections
sla = txA_corrections(cycle,mission,sla_ucTxA)
### Deseason and compute trends
trends_rolled = sla2trends_gridded(xref,yref,sla,timeT,t_limit=2023)
# Center longitude to 180
lon = xref
lon[lon<0] += 360
lon = np.roll(lon,int(len(xref)/2)+1)
'''
## remove later (find lagos)
LR = [5.23, 5.0] #6,4
UL = [7.7, 2.0]
ill = np.where((yref_mesh>=LR[0])&(yref_mesh<=UL[0])&(xref_mesh>=UL[1])&(xref_mesh<=LR[1]))
sla_lagos = sla[:,ill]
plt.plot(timeT,sla_lagos)
'''

# save trends
np.savez('rads_trends_uninterp.npz',lon=lon,lat=yref,trends=trends_rolled,t=timeT)

######
# Sanity Checks
######
## Estimate GMSL
t_g,gmsl_g,rate_g,acc_g = lex.gmsl_goddard()
t_n,gmsl_n,rate_n,acc_n = lex.gmsl_noaa()
# Apply latitudinal weight
scale = np.cos(np.deg2rad(yref_mesh))
gmsl = np.zeros(len(timeT))*np.NaN
for i in range(0,len(timeT)):
    gmsl[i] = np.nansum(sla[i,:,:]*scale)/np.nansum(scale[np.isfinite(sla[i,:,:])])
gmsl_ds = deseason(timeT[~np.isnan(gmsl)],gmsl[~np.isnan(gmsl)])
plt.figure(figsize=(10,4))
plt.plot(timeT[~np.isnan(gmsl)],(gmsl_ds-np.nanmean(gmsl_ds))*1000.,'.',label='CU')
plt.plot(t_g,gmsl_g-np.nanmean(gmsl_g),'.',label='NASA')
plt.plot(t_n,gmsl_n-np.nanmean(gmsl_n),'.',label='NOAA')
plt.legend()
plt.grid()
plt.show()

### Look into timeseries ###
# test point 1
xcoord1 = 50
ycoord1 = -64
indx1 = np.argmin(np.abs(lon-xcoord1))
indy1 = np.argmin(np.abs(yref-ycoord1))
ts1 = sla[:,indy1,indx1]*1000
# test point 2
xcoord2 = 125
ycoord2 = -65
indx2 = np.argmin(np.abs(lon-xcoord2))
indy2 = np.argmin(np.abs(yref-ycoord2))
ts2 = sla[:,indy2,indx2]*1000
# test point 3
xcoord3 = 310
ycoord3 = -65
indx3 = np.argmin(np.abs(lon-xcoord3))
indy3 = np.argmin(np.abs(yref-ycoord3))
ts3 = sla[:,indy3,indx3]*1000
# plot
sz1 = 55
sz2 = 15
plt.figure(figsize=(10,14))
plt.subplot(4,1,1)
plt.title('Trends Prior to Interpolation')
plt.pcolormesh(lon,yref,trends_rolled,cmap = 'jet',vmin=-8,vmax=8)
cbar = plt.colorbar()
cbar.set_label('mm/y')
plt.scatter(lon[indx1],yref[indy1],c='k',s=sz1)
plt.scatter(lon[indx2],yref[indy2],c='g',s=sz1)
plt.scatter(lon[indx3],yref[indy3],c='m',s=sz1)
plt.ylim(-70,70)
plt.subplot(4,1,2)
plt.scatter(timeT, ts1,c='k',s=sz2)
plt.grid()
plt.ylabel('Sea Level [mm]')
plt.subplot(4,1,3)
plt.scatter(timeT, ts2,c='g',s=sz2)
plt.grid()
plt.ylabel('Sea Level [mm]')
plt.subplot(4,1,4)
plt.scatter(timeT, ts3,c='m',s=sz2)
plt.grid()
plt.ylabel('Sea Level [mm]')
plt.tight_layout()
plt.show()

###########################################################################  
## SAVE final map and datasets
###########################################################################
# set geographic bins
dx = .5
xref = np.arange((dx/2),360-(dx/2)+dx,dx)
yref = np.arange(-90+(dx/2),90-(dx/2)+dx,dx)
xref_mesh,yref_mesh = np.meshgrid(xref,yref)

# open file created in main.py
file = np.load('rads_trends_uninterp.npz')
rads_temp = file.f.trends
lon = file.f.lon
lat = file.f.lat
rads_lon = lon
rads_lat = lat
timeT=file.f.t
xmesh, ymesh = np.meshgrid(rads_lon,rads_lat)

# land/inclination mask
mask = mask_out_land_4plot(xref_mesh,yref_mesh)
###########

# interpolate to .5 deg grid
xflat = np.reshape(xmesh,(np.size(xmesh)))
yflat = np.reshape(ymesh,(np.size(ymesh)))
data_flat = np.reshape(rads_temp,np.size(xmesh))
rads = mask*griddata((xflat[np.isfinite(data_flat)],yflat[np.isfinite(data_flat)]),data_flat[np.isfinite(data_flat)],(xref_mesh,yref_mesh),'cubic')

# smooth map
rads_flat = np.reshape(rads,np.size(rads))
xref_flat = np.reshape(xref_mesh,np.size(xref_mesh))
yref_flat = np.reshape(yref_mesh,np.size(yref_mesh))
rads = smooth(xref_flat[np.isfinite(rads_flat)],yref_flat[np.isfinite(rads_flat)],rads_flat[np.isfinite(rads_flat)],mask,6,xref_mesh,yref_mesh)

# mask out Caspian Sea
cond1 = np.logical_and(xref_mesh>44,xref_mesh<60)
cond2 = np.logical_and(yref_mesh>35,yref_mesh<50)
cond = np.logical_and(cond1,cond2)
rads[cond] = np.NaN

# final plot (sla:  sea level rise do not include glacial isostatic adjustment effects on the geoid)
plot_sl_trends(xref,yref,rads,timeT)
plot_sl_trends_simple(xref,yref,rads*30.,timeT,TITLE='30 year projection',clab='mm',vmin=-100,vmax=200)

# save map to npz file
np.savez('final_map_data.npz',lat=yref,lon=xref,trends = rads)

# save map to netCDF file 
ds = nc.Dataset('output_4_website/cu_sea_level_trends.nc','w',format='NETCDF4')
ds.description = 'University of Colorado Sea Level Trends 2021_rel1 (1992.96 - 2022.60)'
lat = ds.createDimension('lat',len(yref))
lon = ds.createDimension('lon',len(xref))
lats = ds.createVariable('lat','f4',('lat',))
lons = ds.createVariable('lon','f4',('lon',))
trends = ds.createVariable('trends','f4',('lat','lon'))
trends.units = 'mm/y'
lats[:] = yref
lons[:] = xref
trends[:,:] = rads
ds.close()

############################################################################
### Validation
# cu uses lon[0,360], noaa uses[-180,180] 
ds_noaa = nc.Dataset('external/slr_map_txj1j2.nc')
slt_noaa = ds_noaa['Sea_level_trends'][:].data # (340, 720)
lat_noaa = ds_noaa['lat'][:].data # 340
lon_noaa = ds_noaa['lon'][:].data # 720
#lon_noaa_new = (lon_noaa + 180) % 360 - 180
slt_noaa_sft = np.hstack((slt_noaa[:,lon_noaa>=0],slt_noaa[:,lon_noaa<0]))

irads = np.where((yref>=lat_noaa.min())&(yref<=lat_noaa.max()))
dslt = rads[irads]-slt_noaa_sft
dslt_exp = np.empty(np.shape(rads))*np.nan
dslt_exp[irads]=dslt
dbias = np.nanmean(dslt)
plot_sl_trends(xref,yref,dslt_exp,timeT,vmin=-3,vmax=3,cmap='RdYlBu_r',TITLE='CU SL trends - NOAA SL trends')

