from Read_file import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def Put_stations_together(main_path):
    """
    Puts the quality controlled temperature data of all MOCCA stations together in one dataframe.

    Returns
    -------
    A pandas dataframe with all MOCCA stations, each column represents one station.

    """
    
    df_list=list()
    for station in ['BAS', 'DOC', 'GRM', 'HAP', 'SLP', 'SNZ']:
        path = main_path + "\data\MOCCA\\" + station + '_temp_QC.csv'
        df = read_csv(path, dtindex=True)
        df = df.loc[:,['Temperature']].copy()
        df.rename(columns={'Temperature': station}, inplace=True)
        df_list.append(df)
        
    df_allstations= pd.concat(df_list, axis=1)
    
    return df_allstations


def Plot_position_gaps(dataframe, save):
    """
    Makes a plot that visualizes the positions of the missing values.
    Each station is visualised separately and below each other.

    Parameters
    ----------
    dataframe : pandas dataframe
        The pandas dataframe with the data of all stations.
        The stations are given in separate columns.
        The index of the dataframe are the datetime stamps.
    save : string
        Path to the main folder, to save the figure in folder 'Figures' that is located in the main folder.

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
    # Make figure
    fig, ax = plt.subplots(1,1, figsize=(10, 4))
    sns.heatmap(dataframe.isnull().transpose(), cmap=cmap, cbar=False)
    
    # Set labels
    plt.yticks(fontsize=ticks_size)
    plt.xlabel('Year', fontsize=text_size)
    plt.xticks(fontsize=text_size)

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
    
    
    # SAVE FIGURE
    if save:
        path_figures = os.path.join(save, "Figures")
        plt.savefig(os.path.join(path_figures, 'Gaps_MOCCA_location.png'), format='png', dpi=1200)
    
    plt.show()



def Plot_both_distributions(df, save=False):
    
    # DETERMINE GAP LENGTHS
    # Identify missing values in all stations
    gaps = df.isnull().astype(int)
    
    # Calculate number of NaN in each set (1 set = one range of missing values OR one range of not missing values) for each station separately
    gap_lengths = gaps.apply(lambda x: x.groupby((x != x.shift()).cumsum()).sum())
           
    # Bring the gap lengths to a single column
    gap_lengths = gap_lengths.melt(var_name='Station', value_name='GapLength').dropna()
    
    # Remove 0's (from sets of not missing values)
    gap_lengths = gap_lengths.loc[gap_lengths['GapLength']!=0]
    
    # Sort the dataframe
    gap_lengths = gap_lengths.sort_values(by=['GapLength'])
    
    # Remove very long gap of SLP (not active anymore)
    gap_lengths = gap_lengths.iloc[:-1,:]
    
    
    # DETERMINE HISTOGRAMS
    # Determine binsize
    binsize = 24
    if gap_lengths['GapLength'].max()%24==0:
        maxbin = gap_lengths['GapLength'].max()
    else:
        maxbin = gap_lengths['GapLength'].max()+24
        
    # In terms of number of gaps
    hist, bin_edges = np.histogram(gap_lengths['GapLength'], bins=np.arange(1, maxbin, binsize))
    hist_percent = hist/hist.sum()
    
    # In terms of number of missing values
    hist_mv, bin_edges_mv = np.histogram(gap_lengths['GapLength'], bins=np.arange(1, maxbin, binsize), weights=gap_lengths['GapLength'])
    hist_percent_mv = hist_mv/hist_mv.sum()
    bin_edges_realmiddle = [bin_edges[i]+binsize/2 for i in np.arange(0,bin_edges.size-1)]
    bin_edges_whiteinbetween = [bin_edges[i]+(binsize-1)/2 for i in np.arange(0,bin_edges.size-1)]
    
    # 3. MAKE PLOT
    # Plot histogram
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,4]}, figsize=(10,4))
    fig.subplots_adjust(hspace=0.05)
    for i in (0,1):
        # royalblue
        axes[i].bar(bin_edges[:-1], hist_percent, width = 11.5, align = 'edge', color= 'darkorange', edgecolor = 'black', linewidth = 0.5, label='Number of gaps')
        axes[i].bar(bin_edges_whiteinbetween, hist_percent_mv, width = 11.5, align = 'edge', color= 'forestgreen', edgecolor = 'black', linewidth = 0.5, label='Number of missing values')
    
    # Marker for splitting y-axis
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    
    # Settings for top plot
    axes[0].set_ylim(0.895, 0.955)
    axes[0].spines.bottom.set_visible(False)
    axes[0].plot([0,1], [0,0], transform=axes[0].transAxes, **kwargs)
    axes[0].set_xticks([])
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in axes[0].get_yticks()], fontsize=10)

    # Settings for bottom plot
    axes[1].set_ylim(0, 0.24)
    axes[1].spines.top.set_visible(False)
    axes[1].plot([0,1], [1,1], transform=axes[1].transAxes, **kwargs)
    axes[1].set_xticks([0, 500, 1000, 1500, 2000, 2500])
    axes[1].set_xticklabels([0, 500, 1000, 1500, 2000, 2500], fontsize=10)
    axes[1].set_yticklabels(['{:,.0%}'.format(x) for x in plt.gca().get_yticks()], fontsize=10)
    axes[1].set_xlabel('Gap length (h)', fontsize=15)
    
    # Set label
    fig.text(0.055, 0.5, 'Occurrence frequency', ha='center', va='center', rotation = 'vertical', fontsize=15)
    
    # Include legend
    axes[0].legend()
    
    
    # SAVE FIGURE
    if save:
        path_figures = os.path.join(save, "Figures")
        plt.savefig(os.path.join(path_figures, 'Gaps_MOCCA_distribution.png'), format='png', dpi=1200)
    
    
    
    
    

