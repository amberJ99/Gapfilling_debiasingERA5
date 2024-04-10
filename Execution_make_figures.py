"""

This file contains the code to make the figures of the paper.

"""
#%% SETUP

# Import all needed packages and files
import sys
from Visualize_gaps import*

# Define paths
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5")

main_path = r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5"
path_results = os.path.join(main_path, "Results")
path_figures = os.path.join(main_path, "Figures")





#%% FIGURE 1: gaps of MOCCA

# Put the data of all stations in one dataframe
df_allstations = Put_stations_together(main_path)
df_allstations.rename(columns={'GRM': 'Plantentuin', 'SLP': 'Sint-Bavo', 'BAS': 'Provinciehuis', 'SNZ': 'Wondelgem', 'HAP': 'Honda', 'DOC':'Melle'}, inplace=True)

# Make figure of locations missing values
Plot_position_gaps(df_allstations, main_path)

# Make figure of distrubtion of gap lengths
Plot_both_distributions(df_allstations, main_path)





#%% FIGURE 6: RESULTS EVALUATION GF TECHNIQUES 
    
# Select the data that is visualized
sv=60
tv=1
positioning = 'both'
city = 'Turku'
stations = ['Betel', 'Kurala', 'Puutori', 'Turola', 'Virastotalo', 'Ylijoki']
station = stations[5]
error_value = 'MSE'
gl_min = 1 # minimum gap length
gl_step = 0 # not with a regular step
gl_max = 336 # maximum gap length
repetitions_value = 10

# Read data
df_errors = pd.read_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(sv) + 'd_tv' + str(tv) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors.csv"), index_col=0)
df_sterr = pd.read_csv(os.path.join(path_results, "TestGFtechniques_" + positioning + '_sv' + str(sv) + 'd_tv' + str(tv) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr.csv"), index_col=0)


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
axes[0][0].legend(['LI', 'Original ERA5', 'Debiased: LR', 'Debiased: MB',
            'Debiased: WMB'], fontsize=11, loc='upper left')

# Marker for splitting x-axis
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

# Save figure
plt.savefig(os.path.join(path_figures, 'Evaluation_techniques_' + station + '.png'), format='png', dpi=1200)





#%% FIGURE 7: RESULTS EVALUATION SELECTION PARAMETERS

# Select the data that is visualized
sv_val=60
tv_val=1
pos = ['left', 'right', 'both', 'separate']
legend_val = pos
city = 'Turku'
stations = ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']
station = stations[0]
error_value = 'MSE'
gl_min = 1
gl_step = 0
gl_max = 336
repetitions_value = 10

# Read data
df_errors = pd.read_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_sv' + str(sv_val) + 'd_tv' + str(tv_val) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_errors.csv"), index_col=0)
df_sterr = pd.read_csv(os.path.join(path_results, "TestGFparameters_" + str(pos) + '_sv' + str(sv_val) + 'd_tv' + str(tv_val) + "h_" + "gl" + str(gl_min) + "-" + str(gl_max) + "-" + str(gl_step) + "_" + "rep" + str(repetitions_value) + "_" + city + "_" + station + "_" + error_value + "_sterr.csv"), index_col=0)

# Make figures and axes
fig, axes = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios': [4,1]}, figsize=(5,7))
fig.subplots_adjust(hspace=0.15, wspace=0.1)

# Choose colormap
if legend_val == pos:
    colormap=['peru', 'red', 'darkorange', 'gold']
if legend_val == tv_val:
    colormap=['lawngreen', 'darkgreen', 'turquoise', 'deepskyblue', 'steelblue']
if legend_val == sv_val:
    colormap =['black', 'gray', 'darkblue', 'dodgerblue', 'rebeccapurple', 'fuchsia', 'crimson', 'palevioletred']

# Make plot for all axes
for i,col in enumerate(df_errors.columns):
    for x in (0,1):
        axes[x].errorbar(df_errors.index, df_errors[col], yerr=df_sterr[col].mul(1.96), capsize=5, markersize=5, marker='o', color=colormap[i])

    
# Set main title and legend
if legend_val == pos:
    fig.suptitle(station, x=0.5, y=0.93, fontsize=17)
if station=='Puutori':
    if legend_val==sv_val or legend_val==tv_val:
        columns=2
    else:
        columns=1
    axes[0].legend(legend_val, ncol=columns, fontsize=13, loc='upper right')

# Marker for splitting x-axis
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)

# Settings left
ax = axes[0]
ax.set_xlim(0,32)
ax.set_ylim(0.5,2.2)
ax.spines.right.set_visible(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.plot([1,1], [1,0], transform=ax.transAxes, **kwargs)
if station == 'Betel':
    ax.set_ylabel('Error (°C$^2$)', fontsize=15)
if legend_val==tv_val:
    ax.set_xlabel('Gaplength (h)', x=0.65, fontsize=15)

# Settings right
ax = axes[1]
ax.set_xlim(332,340) # Make sure the ratio of the limits of the x-axis are in line with the 'width_ratios' of the figure !!!
ax.set_ylim(0.5,2.2)
ax.spines.left.set_visible(False)

ax.xaxis.tick_bottom()
ax.yaxis.tick_right()
ax.plot([0,0], [0,1], transform=ax.transAxes, **kwargs)


# Save figure
plt.savefig(os.path.join(path_figures, 'Evaluation_parameters_' + str(legend_val) + '_' + station + '.png'), format='png', dpi=1200)


#%% FIGURE 8: RESULTS EVALUATION GF ALGORITHM

# Select the data that is visualized
sv = 60
tv = 1
thr_LI = 5
thr_minLP = 30
thr_minLPs = 5
thr_bs = 15
repetitions = 2
P_begingap = 0.002074
distrlabel = 'MOCCA'
city='Turku'

# Calculate original UHI (using ALL observations, not only the ones corresponding to the location of a gap)
UHI_obs, count_obs = Calculate_UHI_onekind(Turku.loc[:,['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo', 'Ylijoki']], ['Betel', 'Kurala', 'Puutori', 'Tuorla', 'Virastotalo'], 'Ylijoki', False)

# Select the data and put it into one dataframe for every station
for station in ['Betel', 'Puutori', 'Virastotalo']:
    # Prepare dataset of original UHI
    UHI_obs_mean = pd.DataFrame(index= UHI_obs[0].index, columns = ['autumn', 'spring', 'summer', 'winter'])
    number_season = [3, 1, 2, 0]  # Sequence of seasons is ['winter', 'spring', 'summer', 'autumn']
    for number, season in enumerate(['autumn', 'spring', 'summer', 'winter']):
        UHI_obs_mean.loc[:, season]=UHI_obs[number_season[number]].loc[:,station]
    
    # Prepare dataset of estimated UHI
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

    # Settings for plot
    ax.set_title('UHI '+ station)
    ax.set_xlabel('Hour')
    ax.set_ylabel('UHI (°C)')

    # To get legends:
    lines, labels = ax.get_legend_handles_labels()
    # ax.legend([lines[i] for i in [0,1,2,3]], [labels[i] for i in [0,1,2,3]], title='Original UHI')
    # ax.legend([lines[i] for i in [4,5,6,7]], [labels[i] for i in [4,5,6,7]], title='Estimated UHI')
    
    # Save figure
    fig.savefig(os.path.join(path_figures, 'Evaluation_algorithm_' + station + 'estimatedUHI.png'), format='png', dpi=1200)


