####
#    Example script to plot AHPS daily precipitation data (1200 UTC day before to 1200 UTC day of) for a fixed region 
#    related to tropical cyclone (TC) landfalls. The resulting map displays rainfall, in inches, with the TC track 
#    overlaid. The track line is labeled each time there is a HURDAT2 entry, usually every 6 hours plus special entries 
#    for landfalls, peak intensity, etc.
#    
#    It assumes the AHPS netCDF files are already downloaded and stored in your working directory.
#    Source for the netCDF files: https://water.weather.gov/precip/downloads/
#    Documentation for AHPS: https://water.weather.gov/precip/about.php
#    
#    It includes custom functions to obtain HURDAT2 track data and ensure consistency in the resulting maps.
#    The custom HURDAT2 function 'onetrack' assumes the HURDAT2 file(s) are in your working directory. Change the path
#    to those files if you have them saved elsewhere on your machine.
#    
#    Tested with Python 3.9.7
#    Example usage:
#    >>> python Mapping_AHPS_precipitation_for_TCs.py AL 2020 Laura
#    
#    Example output map: http://arashi.geosci.msstate.edu/python/2020Laura_rainfall_1day_20200828.png
####
from datetime import datetime, timedelta
from glob import glob
from metpy.plots import USCOUNTIES            # metpy 1.1.0
import cartopy.crs as ccrs                    # cartopy 0.20.1
import cartopy.feature as cfeature
import matplotlib.patheffects as PathEffects  # matplotlib 3.4.2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np                            # numpy 1.19.5
import os
import pyproj                                 # pyproj 3.0.1
import sys
import xarray as xr                           # xarray 0.19.0


def onetrack(season, basin, identifier, printalert=True):
    """
    Custom function to obtain the HURDAT2 best-track for a single TC.
    - 'season' is the 4-digit season during which the TC occurred
    - 'basin' is the basin in which the TC occurred (AL = North Atlantic, EP = eastern North Pacific, CP = 
      central North Pacific
    - 'identifier' is the TC number (01, 02, etc.) or the TC name (Ten, Elsa, etc.)
    
    Output is a tuple with two fields: the TC name (a string) and the track (a list of strings)
    Can be changed to output the track as a NumPy array of strings instead; see commented line at end of function
    """

    def latlonfmt(bad):
        """
        Custom function to transform HURDAT2-formatted latitude and longitude to float values.
        """
        bad = bad.strip()  # Remove whitespace in string 'bad' before continuing
        good = float(bad[0:len(bad)-1])  # Remove E/W/N/S letter before converting string to float
        if bad.find('W') > -1 or bad.find('S') > -1:  # Make value negative if west or south
            good = -1.*good
        return '%.1f' % good  # Output is a string
    
    #### Check for valid basin input and ensure correct types for each input value
    b2 = basin.upper()
    if 'AL EP CP'.find(b2) == -1:
        print('bad basin input: '+b2)  # Print an error because an invalid basin was provided
        exit()
    season = str(season)
    identifier = str(identifier)
    if len(identifier) == 1:
        identifier = identifier.zfill(2)  # ensure zero-padding if needed ("1" --> "01")
    
    #### Load desired HURDAT2 database
    if b2 == 'AL':
        ## North Atlantic HURDAT2 file name (updated for 2020)
        FileName = '/RaiuData/datasets/tracks/hurdat2-1851-2020-052921.txt'
        if os.path.exists(FileName) is False:
            FileName = 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2020-052921.txt'
    else:
        ## eastern and central North Pacific HURDAT2 file name (updated for 2020)
        FileName = '/RaiuData/datasets/tracks/hurdat2-nepac-1949-2020-043021a.txt'
        if os.path.exists(FileName) is False:
            FileName = 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2020-043021a.txt'
    hurdat2 = np.genfromtxt(FileName, dtype=str, delimiter='\n')
    
    #### Locate the desired TC record
    LineLength = np.array([len(line) for line in hurdat2])
    IDrows = np.where(LineLength==37)[0]
    for row in IDrows:  # there's likely a more efficient way to find the TC identifying info, but this approach works!
        if hurdat2[row][4:8] == season:
            if len(identifier) == 2:
                ## Approach when the TC number was provided
                TC_ID = b2+identifier+season
                if hurdat2[row][0:8] == TC_ID:
                    StartRow = row + 1
                    info = hurdat2[row].split(',')
                    EndRow = StartRow + int(info[2].strip())
                    TCname = info[1].strip().capitalize()
                    #print(StartRow, EndRow, TCname, TC_ID)
                    break
            else:
                ## Approach when the TC name was provided
                if hurdat2[row].find(identifier.upper()) > -1:
                    StartRow = row + 1
                    info = hurdat2[row].split(',')
                    TC_ID = info[0]
                    EndRow = StartRow + int(info[2].strip())
                    TCname = info[1].strip().capitalize()
                    #print(StartRow, EndRow, TCname, TC_ID)
                    break
    
    #### Extract the desired TC record
    if printalert is True:
        print('ALERT: Output dates are in YYYYMMDDHHmm format!')
    track = []
    for r in range(StartRow, EndRow):
        data = hurdat2[r].split(', ')
        date = data[0] + data[1]  # format: YYYYMMDDHHmm
        lat = latlonfmt(data[4].strip())
        lon = latlonfmt(data[5].strip())  # ensures western longitudes are NEGATIVE
        wind = data[6].strip()
        pres = data[7].strip()
        status = data[3].strip()
        track.append([date,lat,lon,wind,pres,status])
    return TCname, track  # 'TCname' is a string; 'track' is a list
    #return TCname, np.array(track)  # 'TCname' is a string; 'track' is a NumPy array


def addgridlines(minlat, maxlat, minlon, maxlon):
    """
    Custom function to add reasonable latitude and longitude lines to a geographic map via Cartopy.
    """
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=1., color='gray', linestyle='--', zorder=80)
    gl.xlocator = mticker.FixedLocator(np.arange(-180.,180.,1.))  # draws longitude lines every 1 degree
    gl.ylocator = mticker.FixedLocator(np.arange(-89.,90.,1.))  # draws latitude lines every 1 degree
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl2 = ax.gridlines(crs=pc, draw_labels=True, x_inline=False, linewidth=0., color='gray', alpha=0., linestyle='--', 
                      zorder=0)
    gl2.xlocator = mticker.FixedLocator(np.arange(minlon,maxlon+0.1))
    gl2.ylocator = mticker.FixedLocator(np.arange(minlat,maxlat+0.1))
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.rotate_labels = False


def addcolorbar(figure, axes, imagedata, colorlevels, barlabel=None):
    """
    Custom function to add a well-proportioned colorbar to the map.
    """
    axes_bbox = axes.get_position()
    left = axes_bbox.x1 + 0.015
    bottom = axes_bbox.y0
    width = 0.015
    height = axes_bbox.y1 - bottom
    cax = figure.add_axes([left, bottom, width, height])
    cbar = plt.colorbar(imagedata, cax=cax, ticks=colorlevels, orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    if barlabel is not None:
        cbar.set_label(barlabel, rotation=90)


basin = sys.argv[1].upper()
season = sys.argv[2]
tc = sys.argv[3].capitalize()

#### Load the NHC HURDAT2 best-track for the desired TC via the 'onetrack' function
track = np.array(onetrack(season, basin, tc, printalert=False)[1])
tcdates = np.array([datetime.strptime(track[t,0],'%Y%m%d%H%M') for t in range(track.shape[0])])

#### Declare fixed variables for mapping purposes
mlon, xlon, mlat, xlat = [-97., -85., 27., 35.]  # fixed region centered on Louisiana/Mississippi
lat = None
pc = ccrs.PlateCarree()
counties = USCOUNTIES.with_scale('5m')
scl = '10m'
borders = cfeature.BORDERS.with_scale(scl)
lakes = cfeature.LAKES.with_scale(scl)
coast = cfeature.COASTLINE.with_scale(scl)
states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale=scl,
                                  facecolor='none')
levs = np.arange(0.,12.1,0.5)  # will shade rainfall values every 0.5 inches from 0 to 12
cmap = plt.get_cmap('gist_heat_r')  # colormap for mapping rainfall values
mapProj = ccrs.LambertConformal(central_longitude=(xlon-mlon)/2.+mlon, central_latitude=(xlat-mlat)/2.+mlat)

#### Iterate through available AHPS netCDF files to create 24-h rainfall maps
flist = sorted(glob('nws_precip_1day_*.nc'))  # Create a list of already-downloaded AHPS netCDF files
fdays = np.array([datetime.strptime(fname[16:24],'%Y%m%d') for fname in flist])
tindices = np.where((fdays>=tcdates[0])&(fdays<=tcdates[-1]))[0]
for t in tindices:
    date = fdays[t]
    pname = season + tc + date.strftime('_rainfall_1day_%Y%m%d.png')
    if os.path.exists(pname) is True:
        continue  # skip generating the image if the file already exists
    print(pname)
    ncname = date.strftime('nws_precip_1day_%Y%m%d_conus.nc')
    rows = np.where((tcdates>=date-timedelta(hours=12))&(tcdates<=date+timedelta(hours=12)))[0]
    tclat = np.array(track[rows,1], dtype=float)
    tclon = np.array(track[rows,2], dtype=float)
    date24h = tcdates[rows].copy()
    ds = xr.open_dataset(ncname)
    if lat is None:
        dataproj = pyproj.Proj(ds.crs.proj4)
        xx, yy = np.meshgrid(ds.x.values, ds.y.values) # create 2-D arrays for x and y coordinates
        lon, lat = dataproj(xx, yy, inverse=True) # inverse=True --> extracts lon and lat!!
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(1, 1, 1, projection=mapProj)
    ax.set_extent([mlon, xlon, mlat, xlat], crs=pc)
    ax.add_feature(coast, linewidth=1., zorder=5)
    ax.add_feature(borders, linewidth=1., zorder=5)
    ax.add_feature(lakes, linewidth=0.5, edgecolor='k', facecolor='none', zorder=5)
    ax.add_feature(states, linewidth=1., edgecolor='k', zorder=5)
    ax.add_feature(counties, edgecolor='gray', facecolor='none', linewidth=0.4, zorder=5)
    im = ax.pcolormesh(lon, lat, ds.observation.values, cmap=cmap, shading='nearest', vmin=0, vmax=12, transform=pc, 
                       zorder=1)
    addgridlines(mlon, xlon, mlat, xlat)
    addcolorbar(fig, ax, im, levs[::2])  # label colorbar ticks using every other shaded interval
    ax.spines['geo'].set_zorder(100)  # ensure figure's map border is visible over other geographic lines
    ax.set_title(date.strftime('24-hour rainfall [inches] ending 12 UTC %d %b %Y'), loc='right')
    ax.set_title(tc, weight='bold', loc='left')
    ax.plot(tclon, tclat, c='k', lw=2., transform=pc, zorder=25)
    ax.scatter(tclon, tclat, s=50, transform=pc, zorder=50)
    for i in range(tclon.size):
        if tclat[i] > xlat or tclon[i] > xlon or tclon[i] < mlon or tclat[i] < mlat:
            continue
        text = ax.text(tclon[i], tclat[i], date24h[i].strftime('%H UTC'), size=10, weight='bold', ha='left', 
                       va='center', transform=pc, zorder=50)
        text.set_path_effects([PathEffects.withStroke(linewidth=1.,foreground='w')])
    plt.savefig(pname, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    ds.close()

