"""
Use real-time RAP analysis fields -- accessed via Unidata's THREDDS server -- to compute and display 1000-500-hPa mean
virtual temperature and geopotential thickness to visualize terms in the hypsometric equation.

This script produces maps for the eastern U.S. and was set up for a wintertime event.

Written by Kim Wood. Script provided as is.
"""
from datetime import datetime
import cartopy.crs as ccrs       # cartopy 0.20.1
import cartopy.feature as cfeature
import matplotlib.pyplot as plt  # matplotlib 3.5.1
import matplotlib.ticker as mticker
import metpy.calc as mpcalc      # metpy 1.2.0
import numpy as np               # numpy 1.22.1
import os
import pyproj                    # pyproj 3.2.1
import xarray as xr              # xarray 0.21.0


def fancymap(im, clev, clabel):
    """
    The custom function "fancymap" improves the appearance of our output map.
    I apply a reduced number of lat/lon labels (compared to lat/lon *lines*) to prevent extraneous labels.
        --> 'gl' is for the *lines*
        --> 'gl2' is for the *labels*
    It is hard-coded for the eastern U.S. region displayed by this script. Update for your desired domain as needed.
        im = the object created by contourf()
        clev = the shading levels used when generating 'im'
        clabel = the label you'd like to display on the colorbar (e.g., 'virtual temperature')
    """
    ## Add lat/lon lines + labels
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.5, linestyle='dashed')
    gl.xlocator = mticker.FixedLocator(np.arange(-170.,-59.,5.))
    gl.ylocator = mticker.FixedLocator(np.arange(16.,53.,4.))
    gl2 = ax.gridlines(crs=pc, draw_labels=True, x_inline=False, linewidth=0.)
    gl2.right_labels, gl2.bottom_labels, gl2.rotate_labels = [False] * 3
    gl2.xlocator = mticker.FixedLocator(np.arange(-105.,-69.,5.))
    gl2.ylocator = mticker.FixedLocator(np.arange(24.,49.,4.))
    gl2.xlabel_style = {'size': 12}  # change font size for longitude labels
    gl2.ylabel_style = {'size': 12}  # change font size for latitude labels
    ## Add colorbar + title
    axpos = ax.get_position()  # get the position of the axes to adapt the size and position of the colorbar
    left = axpos.x1 + 0.01
    bottom = axpos.y0 + 0.005
    width = 0.015
    height = axpos.y1 - 0.005 - bottom
    cax = fig.add_axes([left, bottom, width, height])  # add an axes instance for the colorbar
    cbar = plt.colorbar(im, cax=cax, extendfrac='auto', ticks=clev, orientation='vertical')
    cbar.ax.tick_params(labelsize=10, pad=0.003)
    cbar.set_label(clabel, size=12)


dt = datetime(2022,1,29,15)  # choose your date (year,month,day,hour as integers; hour in UTC)

PNGname = dt.strftime('./RAP_TvirtLayer_thickness_%Y%m%d%H.png')  # output PNG image file name (saves to working folder)
if os.path.exists(PNGname) is True:
    exit('    output PNG image already exists! Exiting...')  # script will not run if the output file already exists
                                                             # comment the if statement as needed

#### Load information from cartopy for later geographic mapping
states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='50m')
borders = cfeature.BORDERS.with_scale('50m')
lakes = cfeature.LAKES.with_scale('50m')
clat = 36.5
#clon = -96.5  # example CONUS center longitude
#extent = [clon-25., clon+25., clat-14.5, clat+14.5]  # example CONUS map extent
clon = -87.5  # eastern U.S.
extent = [clon-14.5, clon+14.5, clat-14.5, clat+14.5]  # eastern U.S.
top = 1.0392156862745099 * extent[-1]  # latitude coordinate for map title ('RAP analysis, [date]')
pc = ccrs.PlateCarree()
proj = ccrs.LambertConformal(central_longitude=clon, central_latitude=clat)

#### Load remote data file via xarray
URL = dt.strftime('https://tds.scigw.unidata.ucar.edu/thredds/dodsC/grib/NCEP/RAP/CONUS_13km/RR_CONUS_13km_%Y%m%d_%H00.grib2')
ds = xr.open_dataset(URL)    # use xarray to access the above URL as defined via 'dt'
ds = ds.sel(time=dt.strftime('%Y-%m-%dT%H:%M:%S'))  # reduce the dataset to the RAP analysis (starting) time from 'dt'

#### Want to list the field names + some info from this xarray Dataset? Uncomment the below 6 lines!
#fieldNames = list(ds.variables.keys())
#for fieldName in fieldNames:
#    try:
#        print('%s -- "%s" [%s]' % (fieldName, ds[fieldName].long_name, ds[fieldName].units))
#    except AttributeError:
#        print(fieldName)

#### Compute 1000-500-hPa geopotential thickness (pressure coordinates are in units of Pa)
thick = ds.Geopotential_height_isobaric.sel(isobaric=50000) - ds.Geopotential_height_isobaric.sel(isobaric=100000)

#### Compute 1000-500-hPa pressure-weighted mean virtual temperature
MR = mpcalc.mixing_ratio_from_relative_humidity(ds.isobaric, ds.Temperature_isobaric, ds.Relative_humidity_isobaric)
Tvirt = mpcalc.virtual_temperature(ds.Temperature_isobaric, MR, molecular_weight_ratio=0.6219569100577033)
layer = slice(50000,100000)  # 1000-500-hPa layer
dp = np.zeros(ds.isobaric.sel(isobaric=layer).size, dtype=float)  # NumPy array to store dp, the depth in Pa of each layer
dp[:-1] = np.abs(np.ediff1d(ds.isobaric.sel(isobaric=layer).values))
dp[-1] = dp[-2]  # np.ediff1d produces an array 1 smaller than the input, so make an assumption here for final value
dp = xr.DataArray(data=dp.copy(), dims=['isobaric'], coords={'isobaric':ds.isobaric.sel(isobaric=layer)},
                  attrs={'units': ds.isobaric.units})
TvirtLayer = np.sum(Tvirt.sel(isobaric=layer)*ds.isobaric.sel(isobaric=layer)*dp, axis=0) / \
             np.sum(ds.isobaric.sel(isobaric=layer)*dp, axis=0)

#### Convert x/y coordinates to latitude & longitude (note: steps are NOT via xarray as I'm debugging that approach)
dsProjDict = ds.LambertConformal_Projection.attrs  # obtain the source data map projection
R = dsProjDict['earth_radius']
dsProj = ccrs.LambertConformal(central_longitude=dsProjDict['longitude_of_central_meridian']-360.,
                               central_latitude=dsProjDict['latitude_of_projection_origin'],
                               standard_parallels=[dsProjDict['standard_parallel']],
                               globe=ccrs.Globe(semimajor_axis=R, semiminor_axis=R))
RAPproj = pyproj.Proj(dsProj.proj4_init)
xx, yy = np.meshgrid(ds.x.values*1000., ds.y.values*1000.)  # km --> m
lon, lat = RAPproj(xx, yy, inverse=True)  # use pyproj to convert x/y to lon/lat

#### Define a subregion to improve contour label frequency by restricting the extent of mapped data
rows, cols = np.where((lat>=extent[2]-1.)&(lat<=extent[3]+1.)&(lon>=extent[0]-4.)&(lon<=extent[1]+1.))
r0, r1, c0, c1 = [min(rows), max(rows)+1, min(cols), max(cols)+1]  # limited domain --> improves contour label frequency
contours = np.arange(np.ceil(thick.min()/75.)*75., np.floor(thick.max()/75.)*75.+1., 75.)  # define thickness contours

#### Finally, create the map!
TvirtLevels = np.arange(235.,281.,5.)  # shading levels for the layer-mean virtual temperature in units of K
#fig = plt.figure(figsize=(13.5,9))  # CONUS
fig = plt.figure(figsize=(9,9))  # eastern U.S.
ax = plt.axes(projection=proj)
ax.set_extent(extent, crs=pc)
cf = ax.contourf(lon, lat, TvirtLayer.values, TvirtLevels, cmap=plt.get_cmap('plasma'), extend='both', transform=pc)
fancymap(cf, TvirtLevels, '1000-500-hPa mean virtual temperature [K]')
cl = ax.contour(lon[r0:r1,c0:c1], lat[r0:r1,c0:c1], thick[r0:r1,c0:c1], contours, colors='k', transform=pc)
ax.clabel(cl, contours, inline=True, fmt='%d', fontsize=10)
ax.coastlines(resolution='50m', color='k', linewidth=1)
ax.add_feature(lakes, edgecolor='k', facecolor='none')
ax.add_feature(states, edgecolor='k', facecolor='none')
ax.add_feature(borders, edgecolor='k')
ax.text(-101.8, 21.45, 'contours: 1000-500-hPa geopotential thickness [gpm]', size=11, weight='bold', ha='left', 
        va='top', transform=pc)
title = dt.strftime('RAP analysis, %H UTC %d %b %Y')
ax.text(clon, top, title, size=12, weight='bold', ha='center', va='bottom', transform=pc)
ax.spines['geo'].set_zorder(100)  # ensure map frame border is visible on top of displayed data
#plt.show()  # uncomment if you'd like Python to display an interactive figure window when making this map
plt.savefig(PNGname, bbox_inches='tight', pad_inches=0.05)  # save the map to a PNG file with a small white border
plt.close()


