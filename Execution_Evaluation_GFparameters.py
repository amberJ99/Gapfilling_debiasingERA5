"""

This file contains the code to perform the evaluation of the selection parameters

"""

#%%
import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5") #%% IMPORT ALL THE NEEDED PACKAGES AND FILES

#%% READ IN PACKAGES

from Read_file import *
from Evaluation_GFparameters import *
import os

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
s_val=60
# s_val=[2,5,10,20,30,60,80,150]
tv_val=1
# tv_val = [0, 1, 3, 7, 12]
# pos = 'both'
pos = ['left', 'right', 'both', 'separate']

legend_val = pos


# Declare the names of the data and columns
dataset = Turku
stations = ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']

# Select the station
station = stations[0]
ERA5 = station + '_ERA5'

# Choose settings for the repetition and evaluation of the GF
error_value = 'MSE'
gl_min = 1
gl_step = 0
gl_max = 336
gaplengths = (1, 3, 5, 7, 10, 20, 30, 336)
repetitions_value = 1000
check_value=250


#%% PERFORM THE GF EVALUATION

df_errors, df_sterr = Test_parameters(  df=dataset, 
                                        name_fulldata=station, 
                                        name_model=ERA5,
                                        s=s_val,
                                        tv=tv_val,
                                        positioning = pos,
                                        error='MSE',
                                        gaplengths= gaplengths,
                                        weights=False,
                                        repetitions = repetitions_value,
                                        check = check_value)


#%% SAVE RESULTS

df_errors.to_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_s' + str(s_val) + 'd_tv' + str(tv_val) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors.csv"), index=True)
df_sterr.to_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_s' + str(s_val) + 'd_tv' + str(tv_val) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr.csv"), index=True)

