
import pandas as pd
import netCDF4 as nc
import numpy as np
import os
from pyproj import Proj, transform, Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def read_csv(path, dtindex=True):
    """
    Reads in a csv file to pandas dataframe.

    Parameters
    ----------
    path : string
        Path to the file (folder + name of file + .csv). File has to have the 
        following structure: 'DateTime', Station1, Station2, ...
    dtindex : boolean, optional
        Indication if the index of the dataframe should be the datetimes or
        not. The default is True. In general dtindex is only put to False when
        handling raw data, where duplicated timestamps are possible.

    Returns
    -------
    csvfile : pandas dataframe
        Pandas dataframe with DateTime and observations of each stations in a
        separate column. 

    """
    
    csvfile = pd.read_csv(path, sep=',', parse_dates=['DateTime'])
    if dtindex:
        csvfile.set_index(csvfile.columns[0], inplace=True)
    print('COMMENT: The file ' + os.path.basename(path) + ' is sucessfully read in.')
    return csvfile


def from_epsg_to_latlon(x,y):
    """
    Converts x- and y-coordinates from EPSG3067 coordinate system to latitude
    and longitude.

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.

    Returns
    -------
    lat : float
        latitude.
    lon : float
        longtiude.

    """
    
    # Define the ETRS-TM35FIN and WGS84 coordinate systems
    etrs_tm35fin = Proj(proj='utm', zone=35, ellps='GRS80')  # ETRS-TM35FIN EPSG code
    wgs84 = Proj(proj='latlong', datum='WGS84')          # WGS84 EPSG code
    
    # Perform the coordinate transformation
    lon, lat = transform(etrs_tm35fin, wgs84, x, y)
    
    return lat, lon


def add_latlon_columns(row):
    """
    Selects the x- and y-coordinate and converts them to latitude and 
    longitude.

    Parameters
    ----------
    row : pandas row
        Row must contain a column named 'X-coorinate (EPSG3067)' and a column
        named 'Y-coordinate (EPSG3067)'.

    Returns
    -------
    pandas series
        series with row named 'Latitude' with latitude value and a row named 
        'Longitude' with the longitude value.

    """
    
    lat, lon = from_epsg_to_latlon(row['X-coorinate (EPSG3067)'], row['Y-coordinate (EPSG3067)'])
    
    return pd.Series({'Latitude': lat, 'Longitude': lon})


def Plot_position_gaps(dataframe):
    """
    Makes a plot that visualises the positions of the missing values.
    Each station is visualised below each other, and at the bottom an overall picture is given 
    (for which timestamps at least one station has a missing value)

    Parameters
    ----------
    dataframe : pandas dataframe
        The pandas dataframe with the data of all stations.
        The stations are given in separate columns.
        The index of the dataframe are the datetime stamps.

    Returns
    -------
    None.

    """
    
    # Create figure with subplots
    fig, ax = plt.subplots(2,1, sharex=False, figsize=(10, 4), height_ratios=[3,1])
    # Change the space between subplots (0 for plots side by side)
    fig.subplots_adjust(hspace=0.05) 
    
    # Extract years from DatetimeIndex and exclude the first year
    years = dataframe.index.year.unique()[1:]
    
    # Create positions for the ticks corresponding to January 1st of each year and middle of year
    y_positions = [dataframe.index.get_loc(pd.Timestamp(f'{year}-01-01')) for year in years]
    y_midpoints = [(y_positions[i] + y_positions[i + 1]) / 2 for i in range(len(y_positions) - 1)]
    # y_midpoints.append(y_positions[-1] + (y_positions[-1] - y_positions[-2]) / 2)
    
    # Choose colormap
    cmap = ListedColormap(['white', 'darkred'])
    
    
    # PLOT 1
    plt.subplot(2,1,1)
    sns.heatmap(dataframe.isnull().transpose(), cmap=cmap, cbar=False)
    
    ax[0].get_xaxis().set_visible(False) # Remove this if you want the x-axis to be printed at both subplots
    
    # Set border of plot    
    plt.gca().patch.set(lw=2, ec='k')
    
    text_size = 15
    plt.ylabel('Station', fontsize=text_size)
    plt.xlabel('Year', fontsize=text_size)
    plt.title('Missing values', fontsize=text_size)
    
    # PLOT2
    plt.subplot(2,1,2)
    total = pd.DataFrame(dataframe.isnull().any(axis=1), columns=['All'])
    sns.heatmap(total.transpose(), cmap=cmap, cbar=False)
    
    # Set ticks at y_positions
    plt.gca().set_xticks(y_positions)
    plt.gca().set_xticklabels([])
    plt.gca().tick_params(axis='x', which="major", length=5)
    
    # Set tick labels at y_midpoints
    plt.gca().set_xticks(y_midpoints, minor=True)
    plt.gca().set_xticklabels(years, rotation=0, minor=True)
    plt.gca().tick_params(axis='x', which="minor", length=0)
    
    # Set border of plot    
    plt.gca().patch.set(lw=2, ec='k')
    
    
    # GENERAL
    # plt.suptitle('Missing values MOCCA')
    plt.xlabel('Year', fontsize=text_size)
    plt.xticks(fontsize=text_size)
    plt.yticks(fontsize=text_size)
    
    
    plt.show()

def read_netCDF(path, namestations, info=True):
    """
    Reads in a NetCDF file into pandas dataframe.

    Parameters
    ----------
    path : string
        path to the NetCDF file (including .nc).
    namestations : list of strings
        list with names of the stations, which will be the names of the columns of the pandas dataframe..
    info : boolean, optional
        Describes if general information of the netCDF file is printed or not. The default is True.

    Returns
    -------
    df : pandas dataframe
        dataframe with datetimes as index, and with the values for each station in separate column.

    """

    ds = nc.Dataset(path)
    if info==True:
        print('Information about netCDF data:')
        print(ds)
        print('Information about temperature data:')
        print(ds.variables['t2m'])
        print('Information about time data:')
        print(ds['time'])
        
    df_list=list()
    for i in np.arange(len(namestations)):
        df_station = pd.DataFrame(ds.variables['t2m'][:][:, i, i])    # Select temperature variable, and select all values. Values are given
                                                    # in dimension (T, lat, lon).
        df_list.append(df_station)
        
    df = pd.concat(df_list, axis=1)
    indices = np.array(ds['time'][:], dtype='datetime64[s]') # Convert time in seconds from 1970 to actual date and time
    df=df.set_index(indices)                    # Indices of df = datetime
    df=df.set_axis(namestations, axis=1)        # Colum names = names of stations
    df=df.applymap(lambda l: l-273.15)          # Convert from K to °C
    
    print(df)
    
    return df