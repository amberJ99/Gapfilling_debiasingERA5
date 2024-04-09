# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:34:08 2024

@author: ambjacob
"""

#%%
import sys
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation") #%% IMPORT ALL THE NEEDED PACKAGES AND FILES

#%% READ IN PACKAGES

from Read_file import *
from GF_parameters_evaluation import *
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


#%% Set up the GF

# Dictionary with settings of GF 
# IDEA: give each element for each GF-technique also a name (similar as settings toolkit Thomas)
LP=60
# LP=[2,5,10,20,30,60,80,150]
tw=1
# tw = [0, 1, 3, 7, 12]
# pos = 'both'
pos = ['left', 'right', 'both', 'separate']

legend_val = pos


# Declare the names of the data and columns
dataset = Turku
stations = ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']
station = stations[4]
ERA5 = station + '_ERA5'

# Choose settings for the repetition and calculation of the GF
error_value = 'MSE'
gl_min = 1
gl_step = 0
gl_max = 336
# gaplengths = list(range(gl_min, gl_max, gl_step))
gaplengths = (1, 3, 5, 7, 10, 20, 30, 336)
repetitions_value = 1000
check_value=500


#%% Perform the GF testing

df_errors, df_sterr = Test_parameters(  df=dataset, 
                                        name_fulldata=station, 
                                        name_model=ERA5,
                                        sv=LP,
                                        tv=tw,
                                        positioning = pos,
                                        error='MSE',
                                        gaplengths= gaplengths,
                                        weights=False,
                                        repetitions = repetitions_value,
                                        check = check_value)


#%% SAVE RESULTS

df_errors.to_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors_2.csv"), index=True)
df_sterr.to_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr_2.csv"), index=True)


#%% READ RESULTS

df_errors = pd.read_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors.csv"), index_col=0)
df_sterr = pd.read_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_sv' + str(LP) + 'd_tv' + str(tw) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr.csv"), index_col=0)

print(df_errors)

#%% MAKE PLOT (single parameter)


# Make figures and axes
fig, axes = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios': [4,1]}, figsize=(5,7))
fig.subplots_adjust(hspace=0.15, wspace=0.1)

if legend_val == pos:
    colormap=['peru', 'red', 'darkorange', 'gold']
if legend_val == tw:
    colormap=['lawngreen', 'darkgreen', 'turquoise', 'deepskyblue', 'steelblue']
if legend_val == LP:
    colormap =['black', 'gray', 'darkblue', 'dodgerblue', 'rebeccapurple', 'fuchsia', 'crimson', 'palevioletred']

# Make plot for all axes
for i,col in enumerate(df_errors.columns):
    for x in (0,1):
        axes[x].errorbar(df_errors.index, df_errors[col], yerr=df_sterr[col].mul(1.96), capsize=5, markersize=5, marker='o', color=colormap[i])

    
# Set main title and legend
if legend_val == pos:
    fig.suptitle(station, x=0.5, y=0.93, fontsize=17)
# axes[0][0].legend(['Linear interpolation', 'Undebiased ERA5', 'Debiased: Linear regression', 'Debiased: Mean bias',
#             'Debiased: Temperature variation'], fontsize=13)
if station=='Puutori':
    if legend_val==LP or legend_val==tw:
        columns=2
    else:
        columns=1
    axes[0].legend(legend_val, ncol=columns, fontsize=13, loc='upper right')

# Put slashes
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)

# Settings topleft
ax = axes[0]
ax.set_xlim(0,32)
ax.set_ylim(0.5,2.2)
ax.spines.right.set_visible(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.plot([1,1], [1,0], transform=ax.transAxes, **kwargs)
if station == 'Betel':
    ax.set_ylabel('Error (°C$^2$)', fontsize=15)
if legend_val==tw:
    ax.set_xlabel('Gaplength (h)', x=0.65, fontsize=15)

# Settings topright
ax = axes[1]
ax.set_xlim(332,340) # Make sure the ratio of the limits of the x-axis are in line with the 'width_ratios' of the figure !!!
ax.set_ylim(0.5,2.2)
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

plt.savefig(os.path.join(path_figures, 'Evaluation_parameters_' + str(legend_val) + '_' + station + '.png'), format='png', dpi=1200)



