# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:50:30 2024

@author: ambjacob
"""




#%%
import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation") #%% IMPORT ALL THE NEEDED PACKAGES AND FILES

#%%
from Read_file import *

path_main = r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation"
path = os.path.join(path_main, 'Data\ERA5_geopotential_Turku.nc')


#%%
ds = nc.Dataset(path)

print('Information about netCDF data:')
print(ds)
    
#%% For lat and lon
df = pd.DataFrame(ds.variables['lat_data'][:][:, :])    # Select temperature variable, and select all values. Values are given
                                            # in dimension (T, lat, lon). Value of lat is enough to distinguish
                                            # between stations.
print(df)

#%% For geopotential height
df = pd.DataFrame(ds.variables['z'][:][0, :, :])    # Select temperature variable, and select all values. Values are given
                                            # in dimension (T, lat, lon). Value of lat is enough to distinguish
                                            # between stations.
print(df)

df = df /9.80665

print(df)
print(ds.variables['lat'][:])
print(ds.variables['lon'][:])


#%% For temperature (ERA5_wholegrid)

df = pd.DataFrame(ds.variables['t2m'][:][0, :, :])    # Select temperature variable, and select all values. Values are given
                                            # in dimension (T, lat, lon). Value of lat is enough to distinguish
                                            # between stations.
                                            
print('Information about netCDF data:')
print(ds)
print('Information about temperature data:')
print(ds.variables['t2m'])
print('Information about time data:')
print(ds['time'])
print('Information about realization')
print(ds['realization'])
                                            
# indices = np.array(ds['time'][:], dtype='datetime64[s]') # Convert time in seconds from 1970 to actual date and time
# df=df.set_index(indices)                    # Indices of df = datetime
# df=df.set_axis(namestations, axis=1)        # Colum names = names of stations
df=df.applymap(lambda l: l-273.15)          # Convert from K to °C

print(df)

print(ds.variables['lat'][:])
print(ds.variables['lon'][:])

#%% (ERA5_point1and2)

df = pd.DataFrame(ds.variables['t2m'][:][0, :, :])    
df=df.applymap(lambda l: l-273.15)  
print(df)
print(ds.variables['lat'][:])
print(ds.variables['lon'][:])

#%% (ERA5_Turku_2021)

df = pd.DataFrame(ds.variables['t2m'][:][0, :, :])    
df=df.applymap(lambda l: l-273.15)  
print(df)
print(ds.variables['lat'][:])
print(ds.variables['lon'][:])