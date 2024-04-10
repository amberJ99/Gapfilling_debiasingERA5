"""

This file contains the code to perform the evaluation of the gap-filing algorithm.

"""

#%% SETUP
import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5")

from Read_file import *
from Evaluation_GFalgorithm import *
from GFalgorithm import *
import os


#%% DEFINE PATHS

main_path = r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5"
path_data_Turku = os.path.join(main_path, "Data", "TURCLIM")
path_data_MOCCA = os.path.join(main_path, "Data", "MOCCA")
path_results = os.path.join(main_path, "Results")
path_figures = os.path.join(main_path, "Figures")


#%% READ IN TURKU DATA

Turku_obs = read_csv(os.path.join(path_data_Turku, "Turku_1H_LI.csv"))
Turku_ERA5 = read_csv(os.path.join(path_data_Turku, "Turku_ERA5.csv"))
Turku_ERA5.rename(columns = {old_name: old_name + '_ERA5' for old_name in Turku_ERA5.columns}, inplace=True)
Turku = pd.concat([Turku_obs, Turku_ERA5], axis=1).iloc[2:-2]   # First/last two timestamps are only available for observations/ERA5

print(Turku)
city='Turku'

#%% DETERMINE P AND HISTOGRAM (MOCCA)

# Read in MOCCA data
df_list= list()
for station in ['BAS', 'DOC', 'GRM', 'HAP', 'SLP', 'SNZ']:
    df = read_csv(os.path.join(path_data_MOCCA, station + '_temp_QC.csv'), dtindex=True)
    df = df.loc[:,['Temperature']].copy()
    df.rename(columns={'Temperature': station}, inplace=True)
    df_list.append(df)
MOCCA= pd.concat(df_list, axis=1)

# Calculate chance of gaps and distribution of gaplengths
P_begingap, histogram = Calculate_gaps_distribution(MOCCA, ['BAS', 'DOC', 'GRM', 'HAP', 'SLP', 'SNZ'])

distrlabel = 'MOCCA'

#%% OTHER OPTION: CHOOSE P AND HISTOGRAM

P_begingap = 0.002
histvalues = np.array([0.8, 0.1, 0.06, 0.04])
binedges = np.array([1,2,3,4,5])

histogram = [histvalues, binedges]

distrlabel = 'Customsmallgaps'

#%% SETUP GF ALGORITHM

# Parameters
sv = 60
tv = 1
thr_LI = 5
thr_minLP = 30
thr_minLPs = 5
thr_bs = 15

repetitions = 100

# Dictionary with settings of GF algorithm
dictionary = {"seasonal_variation":         sv,
              "time_variation":             tv,
              "threshold_LI":               thr_LI,     # For gaps smaller than treshold_LI (in amount of hours), LI will be applied
              "threshold_minLP":            thr_minLP,  # minimum amount of days needed for LP
              "threshold_minLPoneside":     thr_minLPs, # minimum amount of days needed on one side of the gap to perform meanbias with separate positioning
              "threshold_bs":               thr_bs      # For gaps threshold_bs <= length, separate positioning is used
              }


#%% PERFORM EVALUATION

UHI_obs_calculations, UHI_filled_calculations = Test_multiple_series_of_gaps(Turku, ['Betel', 'Puutori', 'Virastotalo'], 'Ylijoki', P_begingap, histogram, dictionary, repetitions)


#%% SAVE RESULT

idx = pd.IndexSlice
for station in ['Betel', 'Puutori', 'Virastotalo']:
    # Save the original UHI
    UHI_obs_mean = UHI_obs_calculations.loc[idx[:,station], idx[:,'mean']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    UHI_obs_sterr = UHI_obs_calculations.loc[idx[:,station], idx[:,'std']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    
    UHI_obs_mean.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_mean.csv"), index=True)
    UHI_obs_sterr.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel  + '_rep' + str(repetitions) + '_' +city + "_" + station + "_OBS_sterr.csv"), index=True)

    # Save the estimated UHI
    UHI_filled_mean = UHI_filled_calculations.loc[idx[:,station], idx[:,'mean']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    UHI_filled_sterr = UHI_filled_calculations.loc[idx[:,station], idx[:,'std']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    
    UHI_filled_mean.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_mean.csv"), index=True)
    UHI_filled_sterr.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' +city + "_" + station + "_FILLED_sterr.csv"), index=True)






