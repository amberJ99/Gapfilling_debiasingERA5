# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:14:03 2024

@author: ambjacob
"""

#%%
import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation") #%% IMPORT ALL THE NEEDED PACKAGES AND FILES

#%% READ IN PACKAGES

from Read_file import *
from GF_evaluation import *
import os


#%% DEFINE PATHS

path_main = r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation"
path_data = os.path.join(path_main, "Data")
path_datamade = os.path.join(path_main, "Data_made")
path_results = os.path.join(path_main, "Results")
path_figures = os.path.join(path_main, "Figuren")

#%% Read in Turku dataset
# IDEA: for each station have a separate column with ERA5, because now you have to specifiy which ERA5 to use for each station (with idea: ERA5 = namestation_ERA5)

Turku_obs = read_csv(os.path.join(path_datamade, "Turku_1H_LI.csv"))
Turku_ERA5 = read_csv(os.path.join(path_datamade, "Turku_ERA5.csv"))
Turku_ERA5.rename(columns = {old_name: old_name + '_ERA5' for old_name in Turku_ERA5.columns}, inplace=True)
Turku = pd.concat([Turku_obs, Turku_ERA5], axis=1).iloc[2:-2]   # First/last two timestamps are only available for observations/ERA5

city='Turku'

print(Turku)

#%% Read in Novi Sad dataset

NoviSad = read_csv(os.path.join(path_datamade, "NoviSad_all.csv"))
NoviSad_ERA5 = read_csv(os.path.join(path_datamade, "NoviSad_ERA5.csv"))
NoviSad_ERA5.rename(columns = {old_name: old_name + '_ERA5' for old_name in NoviSad_ERA5.columns}, inplace=True)
NoviSad = pd.concat([NoviSad, NoviSad_ERA5], axis=1)

city='NoviSad'

print(NoviSad)


#%% Set up the GF

# Dictionary with settings of GF
# IDEA: give each element for each GF-technique also a name (similar as settings toolkit Thomas)
LP=60
tw=1
positioning = 'both'

dictionary = {"linint": np.nan,
              "fillmodel": np.nan,
              "debmodelReg": [LP, tw, positioning, 1, False],
              "debmodelMeanbias": [LP, tw, positioning, False],
              "debmodelTvar": [LP, tw, positioning]}

# Declare the names of the data and columns
dataset = Turku
stations = ['Betel', 'Kurala', 'Puutori', 'Turola', 'Virastotalo', 'Ylijoki']
station = stations[5]
ERA5 = station + '_ERA5'

# Choose settings for the repetition and calculation of the GF
par_slicedates_value = int(LP) # /2+1 when chosing both or separate, without /2+1 for left or right
error_value = 'MSE'
gl_min = 1
gl_step = 0
gl_max = 336
# gaplengths = list(range(gl_min, gl_max, gl_step))
gaplengths = (1, 3, 5, 7, 10, 20, 30, 336)
repetitions_value = 1000
check_value=250


#%% Perform the GF testing

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

#%% Save results

df_errors.to_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors.csv"), index=True)
df_sterr.to_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr.csv"), index=True)


#%% Read results

df_errors = pd.read_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors.csv"), index_col=0)
df_sterr = pd.read_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr.csv"), index_col=0)

print(df_errors)
#%% Make simple plot of results

df_errors.plot(marker='o', yerr=df_sterr, ecolor='red')
plt.xlabel('Gaplength (hour)')
plt.ylabel(error_value)
if error_value == 'MBE':
    plt.axhline(color='k')
plt.show()

#%% Make plot (original)

# Choose zoom
ymin=0.8
ymax=4

# Make plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6,7))
for col in df_errors.columns:
    ax1.errorbar(df_errors.index, df_errors[col], yerr=df_sterr[col], capsize=3, markersize=5, marker='o')
    ax2.errorbar(df_errors.index, df_errors[col], yerr=df_sterr[col], capsize=3, markersize=5, marker='o')
ax1.set_title('Performance of GF techniques for ' + station, fontsize=14)
ax1.legend(['Linear interpolation', 'Undebiased ERA5', 'Debiased: Linear regression', 'Debiased: Mean bias',
            'Debiased: Temperature variation'], fontsize=13)
plt.xlabel('Length of gap (h)', fontsize=15)
plt.xticks(gaplengths, gaplengths, fontsize=13)
if error_value== 'MBE':
    ax1.set_ylabel('Error (°C)', fontsize=15)
    ax2.set_ylabel('Error (°C)', fontsize=15)
    plt.axhline(color='k')
elif error_value== 'MSE':
    ax1.set_ylabel('Error (°C$^2$)', fontsize=15)
    ax2.set_ylabel('Error (°C$^2$)', fontsize=15)
ax1.tick_params(axis='y', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
ax2.set_ylim((ymin, ymax))
fig.tight_layout()
plt.show()


#%% Make plot (original) (broken axis matplotlib)

# Make figures and axes
fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [4,1]}, figsize=(6,7))
fig.subplots_adjust(hspace=0.15, wspace=0.1)

# Make plot for all axes
for col in df_errors.columns:
    for y in (0,1):
        for x in (0,1):
            axes[y][x].errorbar(df_errors.index, df_errors[col], yerr=df_sterr[col], capsize=3, markersize=5, marker='o')

    
# Set main title and legend
fig.suptitle(station, x=0.5, y=0.93, fontsize=15)
# axes[0][0].legend(['Linear interpolation', 'Undebiased ERA5', 'Debiased: Linear regression', 'Debiased: Mean bias',
#             'Debiased: Temperature variation'], fontsize=13)
axes[0][0].legend(['LI', 'Original ERA5', 'Debiased: LR', 'Debiased: MB',
            'Debiased: WMB'], fontsize=11, loc='upper left')

# Put slashes
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)

# Settings topleft
ax = axes[0][0]
ax.set_xlim(0,32)
ax.set_ylim(-1,32)
ax.spines.right.set_visible(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.plot([1,1], [1,0], transform=ax.transAxes, **kwargs)
ax.set_ylabel('Error (°C$^2$)', fontsize=13)

# Settings topright
ax = axes[0][1]
ax.set_xlim(332,340) # Make sure the ratio of the limits of the x-axis are in line with the 'width_ratios' of the figure !!!
ax.set_ylim(-1,32)
ax.spines.left.set_visible(False)

ax.xaxis.tick_bottom()
ax.yaxis.tick_right()
ax.plot([0,0], [0,1], transform=ax.transAxes, **kwargs)

# Settings bottomleft
ax = axes[1][0]
ax.set_xlim(0,32)
ax.set_ylim(0.5,2)
ax.spines.right.set_visible(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.plot([1,1], [1,0], transform=ax.transAxes, **kwargs)
ax.set_ylabel('Error (°C$^2$)', fontsize=13)
ax.set_xlabel('Gaplength (h)', fontsize=13)
ax.xaxis.set_label_coords(0.5, 0.08, transform=fig.transFigure)

# Settings bottomright
ax = axes[1][1]
ax.set_xlim(332,340)
ax.set_ylim(0.5,2)
ax.spines.left.set_visible(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_right()
ax.plot([0,0], [0,1], transform=ax.transAxes, **kwargs)


# # Set x axis
# ax2.set_xlabel('Length of gap (h)', fontsize=15)
# ax2.set_xticks(gaplengths, gaplengths, fontsize=13)

# # Set y axis
# if error_value== 'MSE':
#     ax1.set_ylabel('Error (°C$^2$)', fontsize=15)
#     ax2.set_ylabel('Error (°C$^2$)', fontsize=15)
    
# ax1.tick_params(axis='y', labelsize=13)
# ax2.tick_params(axis='y', labelsize=13)
# ax2.set_ylim((ymin, ymax))


# fig.tight_layout()
# plt.show()

plt.savefig(os.path.join(path_figures, 'Evaluation_techniques_' + station + '.png'), format='png', dpi=1200)
