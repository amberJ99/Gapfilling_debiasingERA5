from Read_file import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick
import itertools


def Put_stations_together(obs_type):
    """
    Puts the quality controlled temperature data of all MOCCA stations together in one dataframe.

    Returns
    -------
    A pandas dataframe with all MOCCA stations, each column represents one station.

    """
    
    name_list = {'temp' : 'Temperature', 'humid': 'Humidity', 'speed': 'Windspeed'}
    obs_name = name_list[obs_type]
    
    df_list=list()
    for station in ['BAS', 'DOC', 'GRM', 'HAP', 'SLP', 'SNZ']:
        path = r"C:\Users\ambjacob\Documents\Python_projecten\MOCCA_QCandGF\Data_made\MOCCA\\" + station + '_' + obs_type + '_QC.csv'
        df = read_csv(path, dtindex=True)
        df = df.loc[:,[obs_name]].copy()
        df.rename(columns={obs_name: station}, inplace=True)
        df_list.append(df)
        
    df_allstations= pd.concat(df_list, axis=1)
    print(df_allstations)
    
    return df_allstations


def Plot_position_gaps_withoutAll(dataframe, save=False):
    """
    Makes a plot that visualises the positions of the missing values.
    Each station is visualised below each other.

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
    
    
    # DETERMINE X LABELS
    # Extract years from DatetimeIndex and exclude the first year
    years = dataframe.index.year.unique()[1:]
    # Create positions for the ticks corresponding to January 1st of each year and middle of year
    y_positions = [dataframe.index.get_loc(pd.Timestamp(f'{year}-01-01')) for year in years]
    y_midpoints = [(y_positions[i] + y_positions[i + 1]) / 2 for i in range(len(y_positions) - 1)]
    y_midpoints.append(y_positions[-1] + (y_positions[-1] - y_positions[-2]) / 2)
    
    
    # DETERMINE SETTINGS
    cmap = ListedColormap(['white', 'darkred'])
    text_size = 15
    ticks_size = 10

    
    # CREATE FIGURE
    # Make subplots
    fig, ax = plt.subplots(1,1, figsize=(10, 4))
    # plt.title('Location of missing values', fontsize=15)
    
    
    # PLOT 1
    sns.heatmap(dataframe.isnull().transpose(), cmap=cmap, cbar=False)
    
    # Set labels
    # plt.ylabel('Station', fontsize=text_size)
    plt.yticks(fontsize=ticks_size)
    # plt.title('Location of missing values', fontsize=text_size-2)

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
    ax.hlines(np.arange(1,6), *ax.get_xlim(), color='black', linewidth=0.5)
    
    # GENERAL
    # plt.suptitle('Missing values MOCCA')
    plt.xlabel('Year', fontsize=text_size)
    plt.xticks(fontsize=text_size)
    # ax.set_yticks(ticks=ax.get_yticks, which='minor')
    # plt.yticks(fontsize=text_size)
    
    if save:
        path_main = r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation"
        path_figures = os.path.join(path_main, "Figuren")
        plt.savefig(os.path.join(path_figures, 'Gaps_MOCCA_location.png'), format='png', dpi=1200)
    
    plt.show()



def Plot_both_distributions(df, save=False):
    
    # 1. DETERMINE GAPLENGTHS
    # Identify missing values in all stations
    gaps = df.isnull().astype(int)
    
    # Calculate gap lengths
    # Calculate number of NaN in each set (1 set = one range of missing values OR one range of not missing values)
    gap_lengths = gaps.apply(lambda x: x.groupby((x != x.shift()).cumsum()).sum())
            # shift plaats kolom 1 naar onder --> x!=x.shift() geeft True op datetime dat gap begint en datetime net na gap
            # cumsum zorgt ervoor dat elke reeks aan NaN of elke reeks aan not NaN het zelfde getal krijgen (want gaat 1 omhoog elke keer dat deze een begin of einde van gap tegen komt)
            # groupby zorgt ervoor dat elke reeks bij elkaar gegroepeerd wordt (zodat de volgende bewerking enkel binnen de reeks uitgevoerd wordt)
            # sum zorgt ervoor dat binnen elke reeks de som genomen wordt over de waarden in gaps 
            #    --> omdat de reeks zonder NaN de waarde 0 bevatten zal de som 0 zijn + omdat de reeks met NaN(de gaps) de waarde 1 bevatten zal de som het aantal NaN zijn
            # gap_lengths bevat een index die het aantal reeksen aangeeft. In de kolommen staat voor de x-te reeks de lengte (O voor reeks met waarden en lengte voor gap)
    
    # Bring the gap lengths to a single column
    gap_lengths = gap_lengths.melt(var_name='Station', value_name='GapLength').dropna()
    # print(gap_lengths)
    
    # Remove 0's (from sets of not missing values)
    gap_lengths = gap_lengths.loc[gap_lengths['GapLength']!=0]
    
    
    # Sort the dataframe
    gap_lengths = gap_lengths.sort_values(by=['GapLength'])
    # print(gap_lengths)
    
    # Remove very long gap of SLP
    gap_lengths = gap_lengths.iloc[:-1,:]
    print(gap_lengths.iloc[-15:,:])
    
    
    # 2. DETERMINE HISTOGRAMS
    binsize = 24
    if gap_lengths['GapLength'].max()%24==0:
        maxbin = gap_lengths['GapLength'].max()
    else:
        maxbin = gap_lengths['GapLength'].max()+24
        
    # 2A. IN TERMS OF NUMBER OF GAPS
    hist, bin_edges = np.histogram(gap_lengths['GapLength'], bins=np.arange(1, maxbin, binsize))
    print(hist, bin_edges)
    hist_percent = hist/hist.sum()
    # print(bin_edges.size)
    # print(type(bin_edges))
    print(np.arange(1,bin_edges.size))
    print(hist_percent, bin_edges[:-1])
    
    # 2B. IN TERMS OF NUMBER OF MISSING VALUES
    hist_mv, bin_edges_mv = np.histogram(gap_lengths['GapLength'], bins=np.arange(1, maxbin, binsize), weights=gap_lengths['GapLength'])
    print(hist_mv, bin_edges_mv)
    hist_percent_mv = hist_mv/hist_mv.sum()
    # print(bin_edges.size)
    # print(type(bin_edges))
    bin_edges_realmiddle = [bin_edges[i]+binsize/2 for i in np.arange(0,bin_edges.size-1)]
    print(hist_percent_mv, bin_edges_realmiddle)
    bin_edges_whiteinbetween = [bin_edges[i]+(binsize-1)/2 for i in np.arange(0,bin_edges.size-1)]
    print(hist_percent_mv, bin_edges_realmiddle)
    
    
    # 3. MAKE PLOT
    
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,4]}, figsize=(10,4))
    fig.subplots_adjust(hspace=0.05)
    print(axes)
    for i in (0,1):
        # royalblue
        axes[i].bar(bin_edges[:-1], hist_percent, width = 11.5, align = 'edge', color= 'darkorange', edgecolor = 'black', linewidth = 0.5, label='Number of gaps')
        axes[i].bar(bin_edges_whiteinbetween, hist_percent_mv, width = 11.5, align = 'edge', color= 'forestgreen', edgecolor = 'black', linewidth = 0.5, label='Number of missing values')
    

    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    
    axes[0].set_ylim(0.895, 0.955)
    axes[0].spines.bottom.set_visible(False)
    axes[0].plot([0,1], [0,0], transform=axes[0].transAxes, **kwargs)
    axes[0].set_xticks([])
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in axes[0].get_yticks()], fontsize=10)
    # axes[0].set_yticks(fontsize=text_size)

    axes[1].set_ylim(0, 0.24)
    # axes[1].set_ylim(0, 0.19)
    axes[1].spines.top.set_visible(False)
    axes[1].plot([0,1], [1,1], transform=axes[1].transAxes, **kwargs)
    axes[1].set_xticks([0, 500, 1000, 1500, 2000, 2500])
    axes[1].set_xticklabels([0, 500, 1000, 1500, 2000, 2500], fontsize=10)
    axes[1].set_yticklabels(['{:,.0%}'.format(x) for x in plt.gca().get_yticks()], fontsize=10)
    axes[1].set_xlabel('Gap length (h)', fontsize=15)
    
    fig.text(0.055, 0.5, 'Occurrence frequency', ha='center', va='center', rotation = 'vertical', fontsize=15)
    # axes[0].set_title('Distribution of gaplengths MOCCA', fontsize=15)
    
    axes[0].legend()
    
    # axes[0].set_title('Distribution of gap lengths', fontsize=15)

    if save:
        path_main = r"C:\Users\ambjacob\Documents\Python_projecten\GF_evaluation"
        path_figures = os.path.join(path_main, "Figuren")
        plt.savefig(os.path.join(path_figures, 'Gaps_MOCCA_distribution.png'), format='png', dpi=1200)
    
    
    
    
    

