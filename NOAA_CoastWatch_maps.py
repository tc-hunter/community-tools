"""
Use NOAA CoastWatch THREDDS server to plot various ocean fields.

Successfully ran with Python 3.9.10.
"""
from datetime import datetime
import cartopy.crs as ccrs             # cartopy 0.20.2
import cartopy.feature as cfeature
import matplotlib.pyplot as plt        # matplotlib 3.4.2
import matplotlib.ticker as mticker
import numpy as np                     # numpy 1.22.4
import os
import xarray as xr                    # xarray 2022.3.0


def addcolorbar(ff, axes, ticks, cbar_label='26\N{DEGREE SIGN}C isotherm depth [m]'):
    """
    Custom function to add a vertical colorbar on the right side scaled to the height of the displayed map.
    Includes the ability to change the font size of the tick marks.
    """
    axes_bbox = axes.get_position()
    left = axes_bbox.x1 + 0.015
    bottom = axes_bbox.y0
    width = 0.015
    height = axes_bbox.y1 - bottom
    cax = ff.add_axes([left, bottom, width, height])
    cbar = plt.colorbar(im, cax=cax, ticks=ticks, orientation='vertical', extendfrac='auto')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(cbar_label)


map_extent = [-99.,-9.99,2.,45.01]  # change this to whatever domain you want (minlon,maxlon,minlat,maxlat)
pc = ccrs.PlateCarree()
merc = ccrs.Mercator()
land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='#edd1a1')
cmap = plt.get_cmap('viridis')  # colormap to display the data
clev = np.arange(0,151,10)  # shading levels for colormap

ds = xr.open_dataset('https://www.star.nesdis.noaa.gov/thredds/dodsC/OHCNADailyAgg2022')  # remote dataset for 2022
# print(list(ds.keys))  # print a list of all fields
iso26C = ds.iso26C  # other fields are available -- see above (commented) line
tt = iso26C.time.values[-1]  # pulls most recent day -- change this as needed
dt = datetime.strptime(str(tt)[0:10], '%Y-%m-%d')  # convert np.datetime64 value to datetime value
pngname = dt.strftime('NOAACoastWatch_26Cisotherm_NorthAtlantic_%Y%m%d.png')
if os.path.exists(pngname) is True:
    ## Quit at this point if the output PNG already exists
    exit('    Latest map already exists! Exiting...')

#### This code uses NumPy arrays for plotting.
#### Modify as desired to use xarray plotting instead.
latest = iso26C.sel(time=tt)
dataX = iso26C.longitude.values
dataY = iso26C.latitude.values
data = latest.values

#### Create the map.
fig = plt.figure(figsize=(10,7.5))
ax2 =  plt.axes(projection=merc)
ax2.set_extent(map_extent, crs=pc)
im = ax2.contourf(dataX, dataY, data, clev, cmap=cmap, extend='max', transform=pc)
ax2.add_feature(land, zorder=25)
ax2.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='k', facecolor='none', zorder=50)
ax2.spines['geo'].set_zorder(100)
addcolorbar(fig, ax2, clev)
gl = ax2.gridlines(crs=pc, draw_labels=True, linewidth=0.5, zorder=10)
gl.rotate_labels, gl.top_labels, gl.right_labels = [False] * 3
gl.xlocator = mticker.FixedLocator(np.arange(-90.,-9.,10.))
gl.ylocator = mticker.FixedLocator(np.arange(5.,46.,5.))
ax2.text(-10., 45.2, dt.strftime('%d %b %Y'), size=13, weight='bold', va='bottom', ha='right', transform=pc)
# plt.show()  # Uncomment this line if you'd like the script to display the map in a window
plt.savefig(pngname, bbox_inches='tight', pad_inches=0.05, dpi=150)
plt.close()
