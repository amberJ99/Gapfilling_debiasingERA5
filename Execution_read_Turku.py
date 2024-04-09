#%%

import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation")

#%% IMPORT ALL THE NEEDED PACKAGES AND FILES

from Read_file import *
import os

#%% DEFINE PATHS

path_main = r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation"
path_data = os.path.join(path_main, "Data")
path_datamade = os.path.join(path_main, "Data_made")


#%% TURKU: PUT COORDINATES IN LATITUDE AND LONGITUDE

# Read in Turku cooridnates
Turku_coor = pd.read_excel(os.path.join(path_data, "TURCLIM_observation_site_coordinates.xlsx"),
                      sheet_name = 'Sheet1',
                      skiprows = [0,1])

# Add extra columns with latitude and longitude
Turku_coor[['Latitude', 'Longitude']] = Turku_coor.apply(add_latlon_columns, axis=1)

# Save latitude and longitude
Turku_coor.to_csv(os.path.join(path_datamade, "Turku_coordinates.csv"), columns = ["Site_name", "Latitude", "Longitude"], index=False)
print('COMMENT: The file Turku_coordinates.csv is succesfully saved.')

print(Turku_coor)

#%% TURKU: PUT DATA IN STANDARD FORMAT

# Read in original data (+ put datetime as index)
Turku = pd.read_excel(os.path.join(path_data, "Whole_year_TURCLIM_temperatures_of_the_Turku_area_2012-2021.xlsx"),
                      sheet_name='Sheet1', 
                      # dtype = {'Temp, °C (Ylijoki)':'float64'},
                      parse_dates={'DateTime': ['Date and time (GMT+2)']},
                      )

# Put DateTime as index
Turku['DateTime'] = pd.to_datetime(Turku['DateTime'], format='%d-%m-%Y %H:%M:%S')
Turku.set_index('DateTime', inplace=True)

# Put DateTime in UTC
Turku = Turku.shift(periods=-2, freq='1H')

# Change names of columns + select only usefull columns
new_names = ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']
columns_mapping = {old_name: new_name for old_name, new_name in zip(Turku.columns, new_names)}
Turku = Turku.iloc[:,0:6].rename(columns=columns_mapping)

# Save file
Turku.to_csv(os.path.join(path_datamade, "Turku.csv"), index=True)
print('COMMENT: The file Turku.csv is succesfully saved.')

print(Turku)


#%% TURKU: READ IN DATA

Turku = read_csv(os.path.join(path_datamade, "Turku.csv"))

Turku.info()

#%% TURKU: ANALYSE THE GAPS

# Make hourly
Turku = Turku.asfreq(freq='1H')
print(Turku)

# Number of NaN for each station
Turku_nan = Turku.isna()
NaN_station = Turku_nan.sum()
print(NaN_station)

# Location of NaN --> max consecutive 2 NaN
rows_with_nan = Turku[Turku_nan.any(axis=1)]
print(rows_with_nan)

#%% TURKU: MAKE HOURLY AND COMPLETE
Turku = Turku.asfreq(freq='1H').interpolate()
print(Turku)

# Save file
Turku.to_csv(os.path.join(path_datamade, "Turku_1H_LI.csv"), index=True)
print('COMMENT: The file Turku_1H_LI.csv is succesfully saved.')


#%% TURKU: ERA5 (LAND)

dflist=list()
names = ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']
years = np.arange(2012, 2022, 1)
print(years)

# Put all years together
for year in years:
    df = read_netCDF(os.path.join(path_data, "ERA5_Turku_"+ str(year) +".nc"), names, info=False)
    dflist.append(df)
    #dfERA=dfERA.append(df, verify_integrity=True)
dfERA=pd.concat(dflist)
dfERA.index.name='DateTime'
print(dfERA)

# Save file
dfERA.to_csv(os.path.join(path_datamade, "Turku_ERA5.csv"), index=True)
print('COMMENT: The file Turku_ERA5.csv is succesfully saved.')


