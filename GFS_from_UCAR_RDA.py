"""
This is a 'docstring.' You can use the triple quotation marks to provide general information about the entire script, 
individual functions within the script, and/or instructions on how to use (run) the script.

For proper docstring use, consult this guide: https://www.python.org/dev/peps/pep-0257/
"""
from cartopy.feature import NaturalEarthFeature
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap, from_levels_and_colors
from metpy.units import units
from siphon.catalog import TDSCatalog
from siphon.http_util import session_manager
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import metpy.calc as mpcalc
import numpy as np
import os
import sys


# ----------------------------------------------------------------------------------------------------------------------
# IMPORTANT:
# You need to insert *your* UCAR RDA username and password in the below line instead of the word username and password.
# Because they're strings, they need to be within the single quotes!
# ----------------------------------------------------------------------------------------------------------------------
session_manager.set_session_options(auth=('username', 'password'))


def fancymap():
    """
    The custom function "fancymap()" improves the appearance of our output map.
    """
    axes_bbox = ax.get_position()
    left = axes_bbox.x1 + 0.015
    bottom = axes_bbox.y0
    width = 0.015
    height = axes_bbox.y1 - bottom
    cax = fig.add_axes([left, bottom, width, height])
    cbar = plt.colorbar(im, cax=cax, ticks=clevs, orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.35)
    gl.xlocator = mticker.FixedLocator(np.arange(220.,360.,10.))
    gl.ylocator = mticker.FixedLocator(np.arange(20.,61.,5.))
    gl2 = ax.gridlines(crs=pc, draw_labels=True, linewidth=0., alpha=0.)
    gl2.rotate_labels = False  # disable rotated lat/lon labels
    gl2.top_labels = False     # turn off labels along top x-axis
    gl2.right_labels = False   # turn off labels along right y-axis
    gl2.xlocator = mticker.FixedLocator(np.arange(-140.,0.,10.))
    gl2.ylocator = mticker.FixedLocator(np.arange(20.,61.,5.))
    gl2.xlabel_style = {'size': 12}  # change font size for longitude labels
    gl2.ylabel_style = {'size': 12}  # change font size for latitude labels


def customize_field(ff, lev):
    """
    The custom function "customize_field()" selects labels, colors, etc appropriate for the requested field.
    Preset fields are provided here; the user is encouraged to modify/add fields as desired.
    """
    nn = ''  # ensure colormap norm variable is an empty string if otherwise undefined below
    if lev == 'surface':
        ll = np.arange(948.,1051.,4.)  # levels for MSLP (hPa)
        if ff == 'precip':
            l = [0.1,0.25,0.5,1.,1.5,2.,3.,4.,5.,6.,7.,8.,9.,10.]  # levels for 6-h average precip rate
            cmap0 = plt.get_cmap('Greys')(0)
            cmap1 = plt.get_cmap('Greens')(np.linspace(0.2,1.,9))
            cmap2 = plt.get_cmap('gist_rainbow_r')(np.linspace(0.81,1.,7))
            colors = np.vstack((cmap0,cmap1,cmap2))
            cm = LinearSegmentedColormap.from_list('my_colormap',colors)
            cl = l.copy()
            ex = 'both'
            ft = '6-h average precipitation rate (mm/h) and MSLP (hPa)'
        if ff == 'snow':
            l = np.concatenate((np.array([0.1,0.25]),np.arange(0.5,12.1,0.5)))  # levels for snowfall accumulation
            cmap0 = plt.get_cmap('BuPu')(np.linspace(0.,1.,14))
            cmap1 = plt.get_cmap('Greens')(np.linspace(0.4,1.,13))
            colors = np.vstack((cmap0,cmap1))
            cm, nn = from_levels_and_colors(l, colors, extend='both')
            cl = np.concatenate((np.array([0.1,1.]),np.arange(2.,12.1,1.)))
            ex = 'both'
            ft = '6-hourly accumulated snowfall (10:1 ratio; inches) and MSLP (hPa)'
        if ff == 'srh':
            l = np.arange(0.,801.,10.)
            cm = plt.get_cmap('hot_r')
            cl = np.arange(0.,801.,50.)
            ex = 'both'
            ft = '0-1.5km storm-relative helicity ($\mathregular{m^{2} s^{-2}}$) and MSLP (hPa)'
        if ff == 'gust':
            l = np.arange(0.,101.,2.)  # levels for wind gust magnitude
            cmap0 = plt.get_cmap('BuPu')(np.linspace(0.,1.,26))
            cmap1 = plt.get_cmap('YlOrRd')(np.linspace(0.1,1.,26))
            colors = np.vstack((cmap0,cmap1))
            cm, nn = from_levels_and_colors(l, colors, extend='both')
            cl = np.arange(0.,101.,10.)
            ex = 'max'
            ft = 'surface wind gust (kt) and MSLP (hPa)'
        if ff == 'cape':
            l = np.arange(0.,4001.,100.)  # levels for mixed-layer CAPE
            cm = plt.get_cmap('gnuplot2_r')
            cl = np.arange(0.,4001.,500.)
            ex = 'max'
            ft = '0-90mb mixed-layer CAPE (J/kg) and CIN (J/kg)'
            ll = np.arange(-1000.,-99.,100.)  # levels for mixed-layer CIN
    if ff == 'wind':
        cm = plt.get_cmap('BuPu')
        if lev == '300':
            l = np.arange(50.,171.,20.)  # levels for wind speed (kt)
            ll = np.arange(8000.,10001.,100.)  # levels for heights (m)
        if lev == '500':
            l = np.arange(30.,131.,20.)  # levels for wind speed (kt)
            ll = np.arange(5000.,6101.,50.)  # levels for heights (m)
        cl = l.copy()
        ex = 'max'
        ft = '%s-hPa wind (kt) and geopotential height (m)' % lev
    if ff == 'vort':  # and lev == '500':
        cm = plt.get_cmap('hot_r')
        l = np.arange(1.,51.)  # levels for vorticity (10^-5 s^-1)
        cl = np.arange(5.,51.,5.)
        ex = 'max'
        if lev == '500':
            ll = np.arange(5000.,6101.,50.)  # levels for heights (m)
        if lev == '300':
            ll = np.arange(8000.,10001.,100.)  # levels for heights (m)
        ft = '%s-hPa wind, geopotential height (m), and relative vorticity ($\mathregular{10^{-5} s^{-1}}$)' % lev
    if ff == 'temp' and lev == '700':
        cm = plt.get_cmap('BrBG')
        l = np.arange(10.,91.,5.)  # levels for relative humidity (%)
        cl = np.arange(10.,91.,10.)
        ex = 'both'
        ll = np.arange(-30.,31.,3.)  # levels for temperature (deg C)
        ft = '700-hPa wind, relative humidity (%), and temperature ($^\circ$C)'
    return cm, l, cl, ex, ll, ft, nn  # colormap, shaded contour levels, shown contour levels, how colorbar is extended, 
                                      # line contour levels, plot title, and colormap norm


# ----------------------------------------------------------------------------------------------------------------------
# The main script is below.
# 
# This script expects command-line inputs on execution.
# 
# Example execution to DISPLAY a map of 300-hPa wind and heights for the 72-hour forecast valid at 00 UTC 1 Jan 2020:
#     python GFS_from_UCAR_RDA.py wind 300 2020010100 72
# Example execution to SAVE the above map (the added "1" at the end means a map will be SAVED instead of DISPLAYED):
#     python GFS_from_UCAR_RDA.py wind 300 2020010100 72 1
# 
# Map combination list:
#     wind 300 / vort 300 / wind 500 / vort 500 / temp 700
#     precip surface / snow surface / srh surface / gust surface / cape surface
# ----------------------------------------------------------------------------------------------------------------------
field = sys.argv[1].lower()  # options: wind, vort, temp, precip, snow, srh, gust, cape
plev = sys.argv[2].lower()   # options: 300, 500, 700, surface
dt = sys.argv[3]             # the requested valid time in YYYYMMDDHH format
ForecastHour = sys.argv[4].zfill(3)  # the forecast hour we want
if len(sys.argv) > 5:
    pname = '%s%s_%s_%sh.png' % (field,plev,dt,ForecastHour)
    if os.path.exists(pname) is True:
        print('% already exists! Exiting...' % pname)
        exit()
    else:
        print('Generating %s...' % pname)

# ----------------------------------------------------------------------------------------------------------------------
# Figure out needed GFS file based on command-line input and define some helpful Cartopy variables
# ----------------------------------------------------------------------------------------------------------------------
DesiredTime = datetime.strptime(dt,'%Y%m%d%H')  # This is a 'datetime' variable for the desired map date and time
ModelInit = DesiredTime - timedelta(hours=int(ForecastHour))  # Get the model init time for DesiredTime and ForecastHour
pc = ccrs.PlateCarree()
states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_shp')
catUrl = ModelInit.strftime('https://thredds.rda.ucar.edu/thredds/catalog/files/g/d084001/%Y/%Y%m%d/catalog.xml')
datasetName = ModelInit.strftime('gfs.0p25.%Y%m%d%H.f') + ForecastHour + '.grib2'

# ----------------------------------------------------------------------------------------------------------------------
# Access file on RDA server and define CONUS region (to avoid downloading the global GFS field)
# ----------------------------------------------------------------------------------------------------------------------
catalog = TDSCatalog(catUrl)  # We're using 'TDSCatalog' from the 'siphon' Python package to access the above URL
ds = catalog.datasets[datasetName]  # Let's get the 24-hour forecast file
dataset = ds.remote_access()  # Calls remote_access() -- this gives us access to the file
latitude = dataset.variables['lat'][:]  # Get 1-D array with latitude values
rows = np.where((latitude>=20.)&(latitude<=55.))[0]  # List of locations within 20-55N
r0, r1 = [min(rows), max(rows)+1]
longitude = dataset.variables['lon'][:]  # Get 1-D array with longitude values
cols = np.where((longitude>=220.)&(longitude<=300.))[0]  # List of locations within 120-60W (0-360 longitude format)
c0, c1 = [min(cols), max(cols)+1]
glon, glat = np.meshgrid(longitude[c0:c1], latitude[r0:r1])  # Create 2-D latitude and longitude grids for cartopy use

# ----------------------------------------------------------------------------------------------------------------------
# Request fields from remote file depending on the selected map of interest
# ----------------------------------------------------------------------------------------------------------------------
var_list = list(dataset.variables)  # list of all variable names in 'dataset' (helpful when developing or debugging)
bcolor = 'blue'  # wind barb color
ccolor = 'black'  # contour color (the lines for geopotential height, etc.)
lw = 1.5  # line width for contours
if field == 'wind' or field == 'vort' or field == 'temp':
    ## get zonal wind field
    uwind = dataset.variables['u-component_of_wind_isobaric']
    coords = uwind.coordinates.split()
    levels = dataset.variables[coords[2]]
    lindex = np.where(levels[:]==float(plev)*100.)[0][0]
    uwind = (uwind[0,lindex,r0:r1,c0:c1] * units('m/s')).to(units('kt'))  # pull subregion for desired level
    ## get meridional wind field
    vwind = dataset.variables['v-component_of_wind_isobaric']
    coords = vwind.coordinates.split()
    levels = dataset.variables[coords[2]]
    lindex = np.where(levels[:]==float(plev)*100.)[0][0]
    vwind = (vwind[0,lindex,r0:r1,c0:c1] * units('m/s')).to(units('kt'))  # pull subregion for desired level
    ## get geopotential height field
    ght = dataset.variables['Geopotential_height_isobaric']
    coords = ght.coordinates.split()
    levels = dataset.variables[coords[2]]
    lindex = np.where(levels[:]==float(plev)*100.)[0][0]
    datal = ght[0,lindex,r0:r1,c0:c1]  # get CONUS subregion and desired level
if field == 'vort':
    ## compute relative vorticity from absolute vorticity
    absv = dataset.variables['Absolute_vorticity_isobaric']
    coords = absv.coordinates.split()
    levels = dataset.variables[coords[2]]
    lindex = np.where(levels[:]==float(plev)*100.)[0][0]
    absv = absv[0,lindex,r0:r1,c0:c1]  # this is ABSOLUTE vorticity
    planetary = 2.*(7.2921*10.**-5.)*np.sin(np.radians(glat))  # compute PLANETARY vorticity
    data = (absv - planetary) * 10**5.  # now compute RELATIVE vorticity
if field == 'wind' and int(plev) <= 500:
    ## only compute wind speed for 500- and 300-hPa 'wind' maps
    data = mpcalc.wind_speed(uwind, vwind).m
if plev == '700':
    ## get relative humidity field
    data = dataset.variables['Relative_humidity_isobaric']
    coords = data.coordinates.split()
    levels = dataset.variables[coords[2]]
    lindex = np.where(levels[:]==float(plev)*100.)[0][0]
    data = data[0,lindex,r0:r1,c0:c1]
    ## get temperature field
    datal = dataset.variables['Temperature_isobaric']
    coords = datal.coordinates.split()
    levels = dataset.variables[coords[2]]
    lindex = np.where(levels[:]==float(plev)*100.)[0][0]
    datal = datal[0,lindex,r0:r1,c0:c1] - 273.15  # convert from Kelvin to degrees Celsius
    bcolor = 'black'
if plev == 'surface':
    datal = dataset.variables['MSLP_Eta_model_reduction_msl']  # do NOT use 'Pressure_reduced_to_MSL_msl'!!!
    datal = datal[0,r0:r1,c0:c1] / 100.  # convert from Pa to hPa
    lw = 1.
    if field == 'precip':
        ## get 6-hour-averaged precipitation rate
        data = dataset.variables['Precipitation_rate_surface_6_Hour_Average']
        data = data[0,r0:r1,c0:c1] * (60.*60.)  # convert from per second to per hours
        ccolor = 'blue'
    if field == 'snow':
        ## get 6-hour accumulated snowfall
        prcp = dataset.variables['Precipitation_rate_surface_6_Hour_Average']  # units: kg m**-2 s**-1
        prcp = prcp[0,r0:r1,c0:c1] * (10.*60.*60.*6.) / 25.4
        snow_cat = dataset.variables['Categorical_Snow_surface_6_Hour_Average']
        snow_cat = snow_cat[0,r0:r1,c0:c1]
        data = prcp * snow_cat  # keep only the values categorized as snow
    if field == 'srh':
        ## get storm-relative helicity (0-1500m)
        data = dataset.variables['Storm_relative_helicity_height_above_ground_layer']
        data = data[0,0,r0:r1,c0:c1]
    if field == 'gust':
        ## get wind gust field at the surface (I assume that's 10 m)
        data = dataset.variables['Wind_speed_gust_surface']
        data = data[0,r0:r1,c0:c1] * 1.943844  # convert from m/s to kt
    if field == 'cape':
        ## get mixed-layer CAPE and CIN
        data = dataset.variables['Convective_available_potential_energy_pressure_difference_layer']
        data = data[0,0,r0:r1,c0:c1]  # pressure_difference_layer2 options: 90mb (0) or 127.5mb (1)
        datal = dataset.variables['Convective_inhibition_pressure_difference_layer']
        datal = datal[0,0,r0:r1,c0:c1]

# ----------------------------------------------------------------------------------------------------------------------
# Plot the desired map with a custom title for 1) the plotted fields and 2) the displayed time + forecast hour
# ----------------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(14,10))
ax = plt.axes(projection=ccrs.Mercator(central_longitude=180., min_latitude=21., max_latitude=51.))
ax.set_extent([232.,295.,21.,51.], crs=pc)  # set region bounds for map
ax.coastlines('50m', linewidth=1.)
ax.add_feature(states, linewidth=1., edgecolor='black')
ax.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='k', facecolor='none')
cmap, levs, clevs, extend, lines, field_title, norm = customize_field(field, plev)
if norm != '':
    im = ax.contourf(glon, glat, data, levs, cmap=cmap, norm=norm, extend=extend, transform=pc)
else:
    im = ax.contourf(glon, glat, data, levs, cmap=cmap, extend=extend, transform=pc)  # 'data' is the shaded variable
cs = ax.contour(glon, glat, datal, lines, colors=ccolor, linewidths=lw, transform=pc)  # 'datal' is for contours
plt.clabel(cs, fmt='%d')
if plev != 'surface':
    ax.barbs(glon, glat, uwind.to('kt').m, vwind.to('kt').m, pivot='middle', color=bcolor, regrid_shape=16, 
             transform=pc)
fancymap()
ax.set_title(field_title, loc='left', fontsize=14)
ax.set_title('%d-h GFS forecast valid '%int(ForecastHour) + DesiredTime.strftime('%H UTC %d %b %Y'), loc='right')
if len(sys.argv) == 5:
    plt.show()
else:
    plt.savefig(pname, bbox_inches='tight', pad_inches=0.03)  # Save map and get rid of large whitespace borders
    plt.close()

