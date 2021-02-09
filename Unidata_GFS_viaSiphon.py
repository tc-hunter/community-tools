"""
EXAMPLE SCRIPT to accomplish the following tasks:
    1) Use siphon to access real-time GFS forecast fields from Unidata THREDDS
    2) Find the variable for temperature on pressure levels
    3) Create a subregion for the U.S. and download ONLY data within that subregion for 200-1000 hPa levels
    4) Plot 500-hPa temperature via cartopy
"""
from cartopy.feature import NaturalEarthFeature
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sys


#### Function to add grid lines and labels showing latitude and longitude
def addlines():
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.35)
    gl.xlocator = mticker.FixedLocator(np.arange(-130.,-59.,5.))
    gl.ylocator = mticker.FixedLocator(np.arange(20.,53.,4.))
    gl2 = ax.gridlines(crs=pc, draw_labels=True, linewidth=0., alpha=0.)
    gl2.rotate_labels = False  # disable rotated lat/lon labels
    gl2.top_labels = False     # turn off labels along top x-axis
    gl2.right_labels = False   # turn off labels along right y-axis
    gl2.xlocator = mticker.FixedLocator(np.arange(-130.,-59.,5.))
    gl2.ylocator = mticker.FixedLocator(np.arange(20.,51.,4.))
    gl2.xlabel_style = {'size': 12}  # change font size for longitude labels
    gl2.ylabel_style = {'size': 12}  # change font size for latitude labels


#### Function to add a colorbar that is sized appropriately for the output map
def addcolorbar():
    axes_bbox = ax.get_position()
    left = axes_bbox.x1 + 0.015
    bottom = axes_bbox.y0
    width = 0.015
    height = axes_bbox.y1 - bottom
    cax = fig.add_axes([left, bottom, width, height])
    cbar = plt.colorbar(im, cax=cax, ticks=clevs, orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('500-hPa temperature [K]', size=11)  # MODIFY THIS for other fields!!


"""
Code begins below; modify 'ymdh' as needed for experimenting (since Unidata THREDDS only keeps a few days of forecasts)
"""
ymdh = '2021020812'  # YYYYMMDDHH; user can change to a command-line input with the line ymdh = sys.argv[1]

#### Access GFS forecast initialized at the time given by 'ymdh'
init_time = datetime.strptime(ymdh, '%Y%m%d%H')
catalogURL = 'https://tds.scigw.unidata.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/' + \
             init_time.strftime('GFS_Global_0p25deg_%Y%m%d_%H00.grib2/catalog.xml')
catalog = TDSCatalog(catalogURL)
dataset = catalog.datasets[init_time.strftime('GFS_Global_0p25deg_%Y%m%d_%H00.grib2')].remote_access()
fieldList = list(dataset.variables)  # list of all variable names in 'dataset' (helpful when developing or debugging)

#### Get attribute info for temperature on pressure levels
temperature = dataset.variables['Temperature_isobaric']
for dim in temperature.dimensions:
    print(dim, dataset.variables[dim].units)  # print the dimension name and its units
hoursName, levName, latName, lonName = temperature.dimensions
hours = dataset.variables[hoursName][:]  # NumPy array of forecast hours (hours since 'init_time')
plevs = dataset.variables[levName][:]    # NumPy array of pressure levels (units of Pa)
lat = dataset.variables[latName][:]      # NumPy array of latitudes
lon = dataset.variables[lonName][:]      # NumPy array of longitudes (0-360 format)

#### Get data for desired pressure level range and geographic region (below approach ensures dim order doesn't matter)
hour = 24  # 24-hour forecast
minLev, maxLev = [20000., 100000.]  # 200 & 1000 hPa
minLat, maxLat, minLon, maxLon = [20., 50., 230., 300.]
l0, l1 = [min(np.where((plevs>=minLev)&(plevs<=maxLev))[0]), max(np.where((plevs>=minLev)&(plevs<=maxLev))[0])+1]
r0, r1 = [min(np.where((lat>=minLat)&(lat<=maxLat))[0]), max(np.where((lat>=minLat)&(lat<=maxLat))[0])+1]
c0, c1 = [min(np.where((lon>=minLon)&(lon<=maxLon))[0]), max(np.where((lon>=minLon)&(lon<=maxLon))[0])+1]
temperatureArray = temperature[np.where(hours==hour)[0][0],l0:l1,r0:r1,c0:c1]

#### Plot 500-hPa temperature on a cartopy map
clevs = np.arange(256.,271.)
glon, glat = np.meshgrid(lon[c0:c1], lat[r0:r1])  # create 2-D grids of longitude and latitude for cartopy use
subsetLev = plevs[l0:l1]
data = temperatureArray[subsetLev==50000.,:,:].reshape(lat[r0:r1].size,lon[c0:c1].size)
pc = ccrs.PlateCarree()
states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection=ccrs.Mercator(central_longitude=180., min_latitude=22., max_latitude=50.))
ax.set_extent([-127.,-65.,22.,50.], crs=pc)
ax.coastlines('50m', linewidth=1.5)
ax.add_feature(states, linewidth=1.5, edgecolor='black')
im = ax.contourf(glon, glat, data, clevs, extend='both', transform=pc)
addcolorbar()
addlines()
ax.set_title('GFS F%03d'%hour + (init_time+timedelta(hours=hour)).strftime(' valid at %H UTC %d %b %Y'), loc='left')
ax.set_title(init_time.strftime('(initialized %H UTC %d %b %Y)'), loc='right')
plt.show()

