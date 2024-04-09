# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:27:32 2024

@author: ambjacob
"""

#%%
import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation") #%% IMPORT ALL THE NEEDED PACKAGES AND FILES

#%% READ IN PACKAGES

from Read_file import *
from GF_algorithm_evaluation import *
from GF_algorithm import *
import os


#%% DEFINE PATHS

path_main = r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation"
path_data = os.path.join(path_main, "Data")
path_datamade = os.path.join(path_main, "Data_made")
path_results = os.path.join(path_main, "Results")
path_figures = os.path.join(path_main, "Figuren")


#%% READ IN TURKU DATASET
# IDEA: for each station have a separate column with ERA5, because now you have to specifiy which ERA5 to use for each station (with idea: ERA5 = namestation_ERA5)

Turku_obs = read_csv(os.path.join(path_datamade, "Turku_1H_LI.csv"))
Turku_ERA5 = read_csv(os.path.join(path_datamade, "Turku_ERA5.csv"))
Turku_ERA5.rename(columns = {old_name: old_name + '_ERA5' for old_name in Turku_ERA5.columns}, inplace=True)
Turku = pd.concat([Turku_obs, Turku_ERA5], axis=1).iloc[2:-2]   # First/last two timestamps are only available for observations/ERA5

city='Turku'

print(Turku)

#%% DETERMINE P AND HISTOGRAM (MOCCA)

# Read in MOCCA data
df_list= list()
for station in ['BAS', 'DOC', 'GRM', 'HAP', 'SLP', 'SNZ']:
    df = read_csv(os.path.join(path_data, station + '_temp_QC.csv'), dtindex=True)
    df = df.loc[:,['Temperature']].copy()
    df.rename(columns={'Temperature': station}, inplace=True)
    df_list.append(df)
MOCCA= pd.concat(df_list, axis=1)
print(MOCCA)

# Calculate chance of gaps and distribution of gaplengths
P_begingap, histogram = Calculate_gaps_distribution(MOCCA, ['BAS', 'DOC', 'GRM', 'HAP', 'SLP', 'SNZ'])
print(histogram)

distrlabel = 'MOCCA'

#%% CHOOSE P AND HISTOGRAM

P_begingap = 0.002
histvalues = np.array([0.8, 0.1, 0.06, 0.04])
binedges = np.array([1,2,3,4,5])

histogram = [histvalues, binedges]

distrlabel = 'Customsmallgaps'

print(histogram)


#%% SETUP THE GF ALGORITHM

print(P_begingap)

# Parameters
sv = 60
tv = 1
thr_LI = 5
thr_minLP = 30
thr_minLPs = 5
thr_bs = 15

repetitions = 100

# Dictionary with settings of GF algorithm
dictionary = {"seasonal_variation":      sv,
              "time_variation":          tv,
              "threshold_LI":           thr_LI,  # For gaps smaller than treshold_LI (in amount of hours), LI will be applied
              "threshold_minLP":        thr_minLP,  # minimum amount of days needed for LP
              "threshold_minLPoneside": thr_minLPs,  # minimum amount of days needed on one side of the gap to perform meanbias with separate positioning
              "threshold_bs":           thr_bs   # For gaps threshold_bs <= length, separate positioning is used
              }

#%% PERFORM EVALUATION SINGLE TIME

Test_one_series_of_gaps(Turku, ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo'], 'Ylijoki', P_begingap, histogram, dictionary)


#%% PERFORM EVALUATION MUTLIPLE TIMES

# ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo']

UHI_obs_calculations, UHI_filled_calculations = Test_multiple_series_of_gaps(Turku, ['Betel'], 'Ylijoki', P_begingap, histogram, dictionary, repetitions)


#%% Save results for each station separately

# ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo']

idx = pd.IndexSlice
for station in ['Betel']:
    # Save the original UHI
    UHI_obs_mean = UHI_obs_calculations.loc[idx[:,station], idx[:,'mean']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    UHI_obs_sterr = UHI_obs_calculations.loc[idx[:,station], idx[:,'std']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    
    UHI_obs_mean.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_mean.csv"), index=True)
    UHI_obs_sterr.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel  + '_rep' + str(repetitions) + '_' +city + "_" + station + "_OBS_sterr.csv"), index=True)

    # Save the filled UHI
    UHI_filled_mean = UHI_filled_calculations.loc[idx[:,station], idx[:,'mean']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    UHI_filled_sterr = UHI_filled_calculations.loc[idx[:,station], idx[:,'std']].reset_index(level=1, drop=True).T.reset_index(level=1, drop=True)
    
    UHI_filled_mean.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_mean.csv"), index=True)
    UHI_filled_sterr.to_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' +city + "_" + station + "_FILLED_sterr.csv"), index=True)



#%% READ RESULTS AND MAKE PLOT OF RESULTS (without errorbars)


for station in ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo']:
    # Select the data and put it into one dataframe for every station
    UHI_obs_mean = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_mean.csv"),
                               index_col=0)
    UHI_obs_sterr = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_sterr.csv"),
                               index_col=0)
    UHI_filled_mean = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_mean.csv"),
                               index_col=0)
    UHI_filled_mean.rename(columns= {'autumn': 'autumn_filled', 'spring': 'spring_filled', 'summer': 'summer_filled', 'winter': 'winter_filled'}, inplace=True)
    print(UHI_filled_mean)
    UHI_filled_sterr = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_sterr.csv"),
                               index_col=0)
    

    # Set up the subplots
    fig, ax = plt.subplots(1)
    
    # Plot the UHI
    for count, df in enumerate([UHI_obs_mean, UHI_filled_mean]):
        df.loc[:,'Hour'] = df.index
        dfm = df.melt('Hour', var_name='Season', value_name='UHI')
        if count==0:
            sns.lineplot(ax=ax, data=dfm, x= 'Hour', y='UHI', hue='Season', linestyle='solid').set(title='UHI '+station)
            # plt.legend()
        else:
            sns.lineplot(ax=ax, data=dfm, x='Hour', y='UHI', hue='Season', linestyle='dashed').set(title='UHI '+station)
            # plt.legend()


    # Settings for plot
    ax.set_xlabel('Hour')
    ax.set_ylabel('UHI (°C)')

    handles, labels = ax.get_legend_handles_labels()
    for i in range(4,8):
        handles[i].set_linestyle('--')

    ax.legend(bbox_to_anchor=(1, 1))
    
    plt.show()


#%% READ RESULTS AND MAKE PLOT OF RESULTS (with errorbars and dashed)


for station in ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo']:
    # Select the data and put it into one dataframe for every station
    UHI_obs_mean = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_mean.csv"),
                               index_col=0)
    UHI_obs_sterr = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_sterr.csv"),
                               index_col=0)
    UHI_filled_mean = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_mean.csv"),
                               index_col=0)
    UHI_filled_mean.rename(columns= {'autumn': 'autumn_filled', 'spring': 'spring_filled', 'summer': 'summer_filled', 'winter': 'winter_filled'}, inplace=True)
    UHI_filled_sterr = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_sterr.csv"),
                               index_col=0)
    UHI_filled_sterr.rename(columns= {'autumn': 'autumn_filled', 'spring': 'spring_filled', 'summer': 'summer_filled', 'winter': 'winter_filled'}, inplace=True)
    

    # Set up the subplots
    fig, ax = plt.subplots(1)

    color_map = ['royalblue', 'darkorange', 'forestgreen', 'red']
    
    # Plot the UHI
    for count, df in enumerate([UHI_obs_mean, UHI_filled_mean]):
        for count_season, column in enumerate(df.columns):
            if count==0:
                ax.errorbar(df.index, df[column], yerr=UHI_obs_sterr[column].mul(1.96), capsize=3, markersize=1, marker='.', label= column, color = color_map[count_season])
            else:
                plotdash = ax.errorbar(df.index, df[column], yerr=UHI_filled_sterr[column].mul(1.96), capsize=3, markersize=1, marker='.', label = column, linestyle = 'dashed', color = color_map[count_season])
                plotdash[-1][0].set_linestyle('dashed')

    # Settings for plot
    ax.set_title('UHI '+ station)
    ax.set_xlabel('Hour')
    ax.set_ylabel('UHI (°C)')

    ax.legend(bbox_to_anchor=(1, 1))
    
    plt.show()



#%% READ RESULTS AND MAKE PLOT OF RESULTS (with errorbars and different colors)


for station in ['Betel', 'Kurala', 'Puutori', 'Turola', 'Virastotalo']:
    # Select the data and put it into one dataframe for every station
    UHI_obs_mean = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_mean.csv"),
                               index_col=0)
    UHI_obs_sterr = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_OBS_sterr.csv"),
                               index_col=0)
    UHI_filled_mean = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_mean.csv"),
                               index_col=0)
    UHI_filled_mean.rename(columns= {'autumn': 'autumn_filled', 'spring': 'spring_filled', 'summer': 'summer_filled', 'winter': 'winter_filled'}, inplace=True)
    UHI_filled_sterr = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_sterr.csv"),
                               index_col=0)
    UHI_filled_sterr.rename(columns= {'autumn': 'autumn_filled', 'spring': 'spring_filled', 'summer': 'summer_filled', 'winter': 'winter_filled'}, inplace=True)
    

    # Set up the subplots
    fig, ax = plt.subplots(1)
    
    color_map = ['royalblue', 'darkorange', 'forestgreen', 'red']
    
    # Plot the UHI
    for count, df in enumerate([UHI_obs_mean, UHI_filled_mean]):
        for count_season, column in enumerate(df.columns):
            if count==0:
                ax.errorbar(df.index, df[column], yerr=UHI_obs_sterr[column].mul(1.96), capsize=3, markersize=1, marker='.', label= column, color = color_map[count_season])
            else:
                ax.errorbar(df.index, df[column], yerr=UHI_filled_sterr[column].mul(1.96), capsize=3, markersize=1, marker='.', label = column, color=color_map[count_season], alpha=0.6)


    # Settings for plot
    ax.set_title('UHI '+ station)
    ax.set_xlabel('Hour')
    ax.set_ylabel('UHI (°C)')

    ax.legend(bbox_to_anchor=(1, 1))
    
    plt.show()
    
    plt.savefig(os.path.join(path_results, 'Test' + station + '.pdf'))
    
    

    
#%% READ RESULTS AND MAKE PLOT OF RESULTS (with errorbars and dashed + one correct original UHI)

# ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo']

# Calculate correct original UHI
UHI_obs, count_obs = Calculate_UHI_onekind(Turku.loc[:,['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']], ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo'], 'Ylijoki', False)

# Select the data and put it into one dataframe for every station
for station in ['Betel', 'Puutori', 'Virastotalo']:
    UHI_obs_mean = pd.DataFrame(index= UHI_obs[0].index, columns = ['autumn', 'spring', 'summer', 'winter'])
    number_season = [3, 1, 2, 0]  # Sequence of seasons is ['winter', 'spring', 'summer', 'autumn']
    for number, season in enumerate(['autumn', 'spring', 'summer', 'winter']):
        UHI_obs_mean.loc[:, season]=UHI_obs[number_season[number]].loc[:,station]
    
    UHI_filled_mean = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_mean.csv"),
                                index_col=0)
    # UHI_filled_mean.rename(columns= {'autumn': 'autumn_filled', 'spring': 'spring_filled', 'summer': 'summer_filled', 'winter': 'winter_filled'}, inplace=True)
    UHI_filled_sterr = pd.read_csv(os.path.join(path_results, "TestGFalgorithm_sv" + str(sv) + 'd_tv' + str(tv) + 'h_thrLI' + str(thr_LI) + 'h_thrminLP' + str(thr_minLP) + 'd_thrminLPs' + str(thr_minLPs) + 'd_thrbs' + str(thr_bs) + 'd_Pgap' + str(round(P_begingap,6)) + '_distr' + distrlabel + '_rep' + str(repetitions) + '_' + city + "_" + station + "_FILLED_sterr.csv"),
                                index_col=0)
    # UHI_filled_sterr.rename(columns= {'autumn': 'autumn_filled', 'spring': 'spring_filled', 'summer': 'summer_filled', 'winter': 'winter_filled'}, inplace=True)
    
    # Set up the subplots
    fig, ax = plt.subplots(1)

    color_map = ['royalblue', 'darkorange', 'forestgreen', 'red']
    
    # Plot the UHI
    for count, df in enumerate([UHI_obs_mean, UHI_filled_mean]):
        for count_season, column in enumerate(df.columns):
            if count==0:
                ax.errorbar(df.index, df[column], yerr=None, markersize=1, marker='.', label= column, linestyle = 'dashed', color = color_map[count_season])
            else:
                plotdash = ax.errorbar(df.index, df[column], yerr=UHI_filled_sterr[column].mul(1.96), capsize=3, markersize=1, marker='.', label = column, color = color_map[count_season])
                # plotdash[-1][0].set_linestyle('dashed')

    # Settings for plot
    ax.set_title('UHI '+ station)
    ax.set_xlabel('Hour')
    ax.set_ylabel('UHI (°C)')

    # ax.legend(bbox_to_anchor=(1, 1), title='Original UHI')

    # To get legends:
    lines, labels = ax.get_legend_handles_labels()
    # ax.legend([lines[i] for i in [0,1,2,3]], [labels[i] for i in [0,1,2,3]], title='Original UHI')
    ax.legend([lines[i] for i in [4,5,6,7]], [labels[i] for i in [4,5,6,7]], title='Estimated UHI')
    
    # plt.show()
    
    fig.savefig(os.path.join(path_figures, 'Evaluation_algorithm_' + station + 'estimatedUHI.png'), format='png', dpi=1200)


#%% Plot temperature

df=Turku
df.loc[:,'Hour']=df.index.hour
name_seasons=['autumn', 'spring', 'summer', 'winter']

print(df)

winter = df.loc[(df.index.map(lambda x: x.month in (1, 2, 12)))]
spring = df.loc[(df.index.map(lambda x: x.month in (3, 4, 5)))]
summer = df.loc[(df.index.map(lambda x: x.month in (6, 7, 8)))]
autumn = df.loc[(df.index.map(lambda x: x.month in (9, 10, 11)))]


season_hour_list=list()
for season in [autumn, spring, summer, winter]:
    season_hour = season.groupby('Hour').mean()
    season_hour_list.append(season_hour)
    print(season_hour)
    
    
for count_season, season in enumerate([autumn, spring, summer, winter]):
    fig, ax = plt.subplots(1)
    for station in ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']:
        ax.errorbar(season_hour_list[count_season].index, season_hour_list[count_season].loc[:,station], yerr=None, markersize=1, marker='.', label= station)
    plt.legend()
    plt.title(name_seasons[count_season])
    
    plt.show()


#%% READ RESULTS AND MAKE PLOT OF RESULTS (with errorzone)


# https://stackoverflow.com/questions/56203420/how-to-use-custom-error-bar-in-seaborn-lineplot
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html


#%% Check if MOCCA is hourly --> OK

# Method 1
print(pd.infer_freq(MOCCA.index))

# Method 2
reference_index = pd.date_range(start=MOCCA.index.min(), end=MOCCA.index.max(), freq='H')
print(reference_index)
print(MOCCA.index)
missing_timestamps = reference_index[~reference_index.isin(MOCCA.index)]
missing_timestamps = MOCCA.index[~MOCCA.index.isin(reference_index)]
print("\nMissing Timestamps:")
print(missing_timestamps)

# Method 3
Data_diff = pd.DataFrame()
Data_diff['MOCCA']=MOCCA.index
Data_diff['referentie']=reference_index
Data_diff['gelijk']=(Data_diff['MOCCA']==Data_diff['referentie'])
print(Data_diff)
print(Data_diff.gelijk.sum())


#%% CREATE GAPS IN TURKU DATASET: Based on exact locations in MOCCA

# Perform shift on MOCCA

extra_index = pd.date_range(start='2012-01-01', end='2016-07-01', freq='H', inclusive='left')
pd_extra = pd.DataFrame(index=extra_index, columns=MOCCA.columns)
MOCCA_extended = pd.concat([pd_extra, MOCCA])
MOCCA_shift = MOCCA_extended.shift(-39432).iloc[:-39432,:]

print(MOCCA_shift)
print(pd.infer_freq(MOCCA_shift.index))


# Create gapped data based on MOCCA
# !!! Only time period which is both present at Turku and MOCCA are selected !!!
Turku_gapped_1 = Make_gaps(Turku[['Betel', 'Kurala', 'Puutori', 'Turola', 'Virastotalo', 'Ylijoki']], MOCCA, 'SNZ')
Turku_gapped_2 = Make_gaps(Turku[['Betel', 'Kurala', 'Puutori', 'Turola', 'Virastotalo', 'Ylijoki']], MOCCA_shift, 'DOC')

Turku_gapped = pd.concat([Turku_gapped_2.loc[:'2016-06-30',:], Turku_gapped_1])
Turku_gapped = pd.concat([Turku_gapped, Turku[['Betel_ERA5', 'Kurala_ERA5', 'Puutori_ERA5', 'Turola_ERA5', 'Virastotalo_ERA5', 'Ylijoki_ERA5']]], 
                          join= 'inner', 
                          axis=1)

print(Turku_gapped)


#%% CREATE GAPS IN TURKU DATASET: check results after making gaps based on distribution

# See if chances are alike
print(P_begingap)
print(histogram)

P_begingap_T, histogram_T = Calculate_gaps_distribution(Turku_gapped, ['Betel'])
print('For Turku:')
print(P_begingap_T)
print(histogram_T)


# Check if distributions are alike
results = pd.DataFrame({'MOCCA':histogram[0][0:histogram_T[1][:-1].size], 'Turku': histogram_T[0]}, index = histogram_T[1][:-1])
results['x'] = results.index
results = results.melt(id_vars='x')

plt.figure()
sns.barplot(x='x', y='value', hue='variable', data=results)
plt.xlim(-1,100)
plt.show()


# Check position of gaps
plt.figure()
cmap = ListedColormap(['white', 'darkred'])
sns.heatmap(Turku_gapped.isnull().transpose(), cmap=cmap, cbar=False)
plt.show()




#%% READ IN NOVI SAD DATASET

NoviSad = read_csv(os.path.join(path_datamade, "NoviSad_all.csv"))
NoviSad_ERA5 = read_csv(os.path.join(path_datamade, "NoviSad_ERA5Land.csv"))
NoviSad_ERA5.rename(columns = {old_name: old_name + '_ERA5' for old_name in NoviSad_ERA5.columns}, inplace=True)
NoviSad = pd.concat([NoviSad, NoviSad_ERA5], axis=1)

city='NoviSad'

print(NoviSad)



#%% CREATE GAPS IN NOVI SAD DATASET

# Read in MOCCA data
df_list= list()
for station in ['BAS', 'DOC', 'GRM', 'HAP', 'SLP', 'SNZ']:
    df = read_csv(os.path.join(path_data, station + '_temp_QC.csv'), dtindex=True)
    df = df.loc[:,['Temperature']].copy()
    df.rename(columns={'Temperature': station}, inplace=True)
    df_list.append(df)
MOCCA= pd.concat(df_list, axis=1)
print(MOCCA)

# Create gapped data based on MOCCA
# !!! Only time period which is both present at NoviSad and MOCCA are selected !!!
NoviSad_gapped = Make_gaps(NoviSad[['s2-2', 's2-3', 's3-2', 's5-2', 's5-3', 's5-4', 's5-5', 's5-6', 's6-4', 's6-8', 's6-9', 's8-1', 'sA-1']], MOCCA, 'DOC')
NoviSad_gapped = pd.concat([NoviSad_gapped, NoviSad[['s2-2_ERA5', 's2-3_ERA5', 's3-2_ERA5', 's5-2_ERA5', 's5-3_ERA5', 's5-4_ERA5', 's5-5_ERA5', 's5-6_ERA5', 's6-4_ERA5', 's6-8_ERA5', 's6-9_ERA5', 's8-1_ERA5', 'sA-1_ERA5']]], 
                         join= 'inner', 
                         axis=1)

print(NoviSad_gapped)
print(NoviSad_gapped.isna().sum())




#%% EVALUATE BY LOOKING AT THE UHI





