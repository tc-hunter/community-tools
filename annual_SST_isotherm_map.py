####
#    Example script to compute an average SST per year over a range of dates and plot a particular isotherm from that
#    average on a map, with each year earning its own color from a colormap.
#
#    The SST data are provided by NOAA's Physical Sciences Laboratory (PSL), which spans late 1981 to the present:
#    https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
#    and are remotely accessible via xarray through their THREDDS server.
#
#    Example output map: https://twitter.com/DrKimWood/status/1461801164173942797
####
from datetime import datetime, timedelta
from matplotlib.ticker import FixedLocator
import cartopy.crs as ccrs            # cartopy 0.20.1
import cartopy.feature as cfeature
import matplotlib as mpl              # matplotlib 3.4.2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np                    # numpy 1.19.5
import xarray as xr                   # xarray 0.19.0


def addgridlines(minlat, maxlat, minlon, maxlon, latint=1., lonint=1.):
    """
    Custom function to add reasonable latitude and longitude lines to a geographic map via Cartopy.
    'latint' and 'lonint' are the intervals between latitude lines and longitude lines, respectively.
    """
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=1., color='gray', linestyle='--', zorder=80)
    if latint == 1.:
        gl.ylocator = FixedLocator(np.arange(-90.,90.,latint))
    else:
        minlat = np.ceil(minlat/latint) * latint
        maxlat = np.floor(maxlat/latint) * latint
        gl.ylocator = FixedLocator(np.arange(minlat-(latint*4.),90.,latint))
    if lonint == 1.:
        gl.xlocator = FixedLocator(np.arange(-180.,180.,lonint))
    else:
        minlon = np.ceil(minlon/lonint) * lonint
        maxlon = np.floor(maxlon/lonint) * lonint
        gl.xlocator = FixedLocator(np.arange(minlon-(lonint*4.),180.,lonint))
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl2 = ax.gridlines(crs=pc, draw_labels=True, x_inline=False, linewidth=0., color='gray', alpha=0., linestyle='--', 
                       zorder=0)
    gl2.xlocator = FixedLocator(np.arange(minlon,maxlon+0.1,lonint))
    gl2.ylocator = FixedLocator(np.arange(minlat,maxlat+0.1,latint))
    gl2.xlabel_style = {'size': 13}
    gl2.ylabel_style = {'size': 13}
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.rotate_labels = False


#### Declare fixed variables for mapping purposes
mlon, xlon, mlat, xlat = [-100., -75., 18., 34.]  # longitude and latitude bounds
latslice = slice(mlat, xlat)
lonslice = slice(mlon+360., xlon+360.+5.)  # pull extra data due to Lambert Conformal projection
lat = None
pc = ccrs.PlateCarree()
scl = '10m'
borders = cfeature.BORDERS.with_scale(scl)
lakes = cfeature.LAKES.with_scale(scl)
coast = cfeature.COASTLINE.with_scale(scl)
states = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale=scl,
                                  facecolor='none')
mapProj = ccrs.LambertConformal(central_longitude=(xlon-mlon)/2.+mlon, central_latitude=(xlat-mlat)/2.+mlat)

#### Set up the map and iterate through the desired years to compute each annual average (see 'timeslice' variable)
year1 = 1982  # start year
year2 = 2020  # end year
cmap = plt.get_cmap('cool')  # define colormap
colors = cmap(np.linspace(0.,1.,year2-year1+1))
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1, 1, 1, projection=mapProj)
ax.set_extent([mlon, xlon, mlat, xlat], crs=pc)
ax.add_feature(coast, linewidth=1., zorder=5)
ax.add_feature(borders, linewidth=1., zorder=5)
ax.add_feature(lakes, linewidth=0.5, edgecolor='k', facecolor='none', zorder=5)
ax.add_feature(states, linewidth=1., edgecolor='k', zorder=5)
data = None
for year in range(year1,year2+1):
    ds = xr.open_dataset('https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.%d.nc'%year)
    timeslice = slice(np.datetime64('%d-08-01'%year), np.datetime64('%d-10-31'%year))  # 1 Aug to 31 Oct
    average = ds.sst.sel(time=timeslice,lat=latslice,lon=lonslice).mean(dim='time')
    if data is None:
        ## Store annual average SST data in an array in case you want it later
        data = average.values.reshape(1,average.shape[0],average.shape[1])
    else:
        data = np.concatenate((data,average.values.reshape(1,average.shape[0],average.shape[1])), axis=0)
    ## Print once per iteration to monitor progress and see minimum and maximum values per year
    print(year, np.nanmin(data[year-year1,:,:]), np.nanmax(data[year-year1,:,:]))
    hex_str = mcolors.to_hex(colors[year-year1])  # convert array of color values to a hex string
    ax.contour(average.lon.values-360., average.lat.values, average.values, [28.5], colors=hex_str, transform=pc)
    ds.close()

#### Add gridlines and a title to the map
addgridlines(mlat, xlat, mlon, xlon, latint=4., lonint=5.)
title = 'location of the average August-October 28.5\N{DEGREE SIGN}C (83.3\N{DEGREE SIGN}F) isotherm, %d-%d' % (year1,year2)
ax.set_title(title, weight='bold', size=14, loc='left')
#### The below 9 lines choose the colorbar placement and generate the shading for the desired year range
axes_bbox = ax.get_position()
left = axes_bbox.x1 + 0.015
bottom = axes_bbox.y0
width = 0.015
height = axes_bbox.y1 - bottom
cax = fig.add_axes([left, bottom, width, height])
norm = mcolors.Normalize(vmin=year1, vmax=year2)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
cbar.ax.tick_params(labelsize=12)
plt.show()
#plt.savefig('map28.5Ccontour.png', bbox_inches='tight', pad_inches=0.05)  # uncomment this line to save the figure
#plt.close()
