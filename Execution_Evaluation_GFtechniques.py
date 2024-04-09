"""

This file contains the code to perform the evaluation of the gap-filling techniques

"""
#%% SETUP
import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5") #%% IMPORT ALL THE NEEDED PACKAGES AND FILES

import os
from Read_file import *
from Evaluation_GFtechniques import *

#%% DEFINE PATHS

main_path = r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5"
path_data = os.path.join(main_path, "Data", "TURCLIM")
path_results = os.path.join(main_path, "Results")
path_figures = os.path.join(main_path, "Figures")

#%% READ IN TURKU DATA

Turku_obs = read_csv(os.path.join(path_data, "Turku_1H_LI.csv"))
Turku_ERA5 = read_csv(os.path.join(path_data, "Turku_ERA5.csv"))
Turku_ERA5.rename(columns = {old_name: old_name + '_ERA5' for old_name in Turku_ERA5.columns}, inplace=True)
Turku = pd.concat([Turku_obs, Turku_ERA5], axis=1).iloc[2:-2]   # First/last two timestamps are only available for observations/ERA5

city='Turku'


#%% SETUP OF GF

# Dictionary with settings of GF
sv=60
tv=1
positioning = 'both'

dictionary = {"linint": np.nan,
              "fillmodel": np.nan,
              "debmodelReg": [sv, tv, positioning, 1, False],
              "debmodelMeanbias": [sv, tv, positioning, False],
              "debmodelTvar": [sv, tv, positioning]}

# Declare the names of the data and columns
dataset = Turku
stations = ['Betel', 'Kurala', 'Puutori', 'Turola', 'Virastotalo', 'Ylijoki']

# Select the station that is gapfilled
station = stations[5]
ERA5 = station + '_ERA5'

# Choose settings for the repetition and evaluation of the GF
par_slicedates_value = int(sv) # number of days in which no gaps are placed in the beginning and end of dataset
error_value = 'MSE'
gl_min = 1 # minimum gap length (for adiministration purpose only)
gl_step = 0 # not with a regular step (for adiministration purpose only)
gl_max = 336 # maximum gap length (for adiministration purpose only)
gaplengths = (1, 3, 5, 7, 10, 20, 30, 336)
repetitions_value = 10
check_value=250


#%% PERFORM THE GF EVALUATION

df_errors, df_sterr = Test_techniques_differentgaplengths(df=dataset, 
                                                        name_fulldata=station, 
                                                        name_model=ERA5,
                                                        dictionarytechniques=dictionary, 
                                                        par_slicedates=par_slicedates_value, 
                                                        error=error_value,
                                                        range_gaplengths=gaplengths,
                                                        repetitions=repetitions_value, 
                                                        check=check_value, 
                                                        plot=False)

#%% SAVE RESULTS

df_errors.to_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(sv) + 'd_tv' + str(tv) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors.csv"), index=True)
df_sterr.to_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(sv) + 'd_tv' + str(tv) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr.csv"), index=True)



