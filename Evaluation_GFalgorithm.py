# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:02:56 2024

@author: ambjacob
"""

import pandas as pd
import numpy as np
import random as r
import seaborn as sns
from GF_algorithm import *

def Make_gaps(df, df_referencegaps, station_referencegaps):
    """
    Creates gaps in pandas dataframe based on the occurence of gaps in another data series.
    Only the time period which is present in both the original dataframe and the reference dataframe is selected.

    Parameters
    ----------
    df : pandas dataframe
        Data with datetime index and one or multiple columns in which gaps will be created.
    df_referencegaps : pandas dataframe
        Data with at least one column with the presence of gaps.
    station_referencegaps : string
        Name of the column of the reference dataframe with the gaps.

    Returns
    -------
    df_gaps : pandas dataframe
        Original dataframe, but with gaps created in each column. 

    """

    # Create dataframe for gapped data, but only with overlapping datetimes
    df_gaps = df.copy()
    df_gaps = df_gaps.loc[df_gaps.index.isin(df_referencegaps.index)]

    # Determine rows with nan in reference dataframe
    mask = df_referencegaps.loc[df_gaps.index, station_referencegaps].isnull()

    # Change the values to nan
    df_gaps.loc[mask,:]=np.nan

    return df_gaps

def Calculate_gaps_distribution(df_referencegaps, stations_referencegaps):
    """
    For a given dataframe, determine the chance that a datetime is the beginning of a gap and the distribution of gaplengths (probability).

    Parameters
    ----------
    df_referencegaps : pandas dataframe
        Dataframe for which the chance and probability is calculated. 
        The dataframe can contain multiple columns. 
        In that case, all columns will be placed below each other and one chance/probability is calculated over all columns together.
    stations_referencegaps : list of strings
        Names of the columns to take into account.

    Returns
    -------
    P_begingap : float
        Chance that a random datetime is the beginning of a gap.
    list
        Information about the probability distribution of the gaplengths.
        The first element gives the probability for each bin.
        The second element gives the edges of the bins. Bins include the left edge.

    """
    
    
    # Select all reference stations
    df_ref = df_referencegaps.loc[:,stations_referencegaps].copy()
    
    # Determine gaps of reference df
    gaps = df_ref.isnull().astype(int)
    gap_lengths = gaps.apply(lambda x: x.groupby((x != x.shift()).cumsum()).sum())      # Calculate number of NaN in each set
    gap_lengths = gap_lengths.melt(var_name='Station', value_name='GapLength').dropna() # Put all stations together
    gap_lengths = gap_lengths.loc[gap_lengths['GapLength']!=0]                          # Remove 0's (from sets of not missing values)
    gap_lengths = gap_lengths.sort_values(by=['GapLength'])                             # Sort the dataframe
    gap_lengths = gap_lengths.iloc[:-1,:]                                               # Remove very long gap of SLP
    # Eventueel zeer lang gat SLP (einde van metingen) verwijderen, maar dan moet dit aantal DT ook verwijderd worden !!!
    # print(gap_lengths)
    
    # Determine chance of datetime being the beginning of a gap
    number_of_gaps = gap_lengths.shape[0]
    # print(number_of_gaps)
    number_of_DT = df_ref.melt(var_name='Station', value_name='GapLength').shape[0] - 12631
    # print(number_of_DT)
    P_begingap = number_of_gaps/number_of_DT
    # print(P_begingap)
    
    # Determine distribution of gaplengths
    
    # With bin edges on .5
    # bin_size = 1
    # binvalues = np.arange(-0.5, gap_lengths['GapLength'].max()+1.5, bin_size)
    # histvalues, bin_edges = np.histogram(gap_lengths['GapLength'], bins=binvalues)
    
    # With bin edges as integers (left edge of bin is inclusive, right edge is exclusive except for the last bin)
    bin_size = 1
    binvalues = np.arange(1, gap_lengths['GapLength'].max()+1.5, bin_size)
    histvalues, bin_edges = np.histogram(gap_lengths['GapLength'], bins=binvalues, density=True)
    
    # No control over edges of bins, only the number
    # histvalues, bin_edges = np.histogram(gap_lengths['GapLength'], bins=int(gap_lengths['GapLength'].max()))
    
    # print(histvalues)
    # print(bin_edges)
    
    return P_begingap, [histvalues, bin_edges]
    
def Make_gaps_based_on_distribution(df, P_begingap, histogram, savespace):
    """
    Creates gaps in pandas dataframe based on a chance of the occurrence of a gap and a distribution of gaplengths.
    

    Parameters
    ----------
    df : pandas dataframe
        Data with datetime index and one or multiple columns in which gaps will be created..
    P_begingap : float
        Chance that a random datetime is the beginning of a gap.
    histogram : list of two elements
        Information about the probability distribution of the gaplengths.
        The first element gives the probability for each bin.
        The second element gives the edges of the bins. Bins include the left edge.
    savespace : integer
        In order for the algorithm to always work, the situation must be avoided in which the first gap is placed at the very beginning of the dataset.
        savespace is the number of observations which will be skipped in the beginning of the dataset when making gaps.

    Returns
    -------
    df_gaps : pandas dataframe
        Original dataframe, but with gaps created in each column.
        The occurence of gaps will be the same for each column.

    """
    
    # Make series of gaps
    df_gaps = df.copy()
    
    i=0
    # print(df_gaps.index.size)
    while i < df_gaps.index.size:
        # Determine if it located in savespace:
        if i < savespace:
            i+=1
            
        else:
           # Determine if this datetime is the beginning of a gap
           is_gap = (r.random()<= P_begingap)
           
           if is_gap:
               # print('Gap for index:')
               # print(i)
               # Determine gaplength
               what_length = np.random.choice(len(histogram[1]) - 1, p=histogram[0])
               gaplength = int(histogram[1][what_length])
               # print('With gaplength:')
               # print(gaplength)
               # Make gap
               df_gaps.iloc[i:i+gaplength,:] = np.nan
           
           # Go to next timestamp
           if is_gap:
               i= i+ gaplength + 1     # +1 because two gaps cannot lie side by side
               # print('Next i:')
               # print(i)
           else:
               i+=1
            
    return df_gaps
        



def sliceSeasons(df):
    """
    Splits a dataframe in the four different seasons.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with datetime index.

    Returns
    -------
    winter : pandas dataframe
        Dataframe which contains the rows of the original dataframe df that are located in the winter.
    spring : pandas dataframe
        Dataframe which contains the rows of the original dataframe df that are located in the spring.
    summer : pandas dataframe
        Dataframe which contains the rows of the original dataframe df that are located in the summer.
    autumn : pandas dataframe
        Dataframe which contains the rows of the original dataframe df that are located in the autumn.

    """

    winter = df.loc[(df.index.map(lambda x: x.month in (1, 2, 12)))]
    spring = df.loc[(df.index.map(lambda x: x.month in (3, 4, 5)))]
    summer = df.loc[(df.index.map(lambda x: x.month in (6, 7, 8)))]
    autumn = df.loc[(df.index.map(lambda x: x.month in (9, 10, 11)))]
    
    return winter, spring, summer, autumn



def Calculate_UHI_onekind(df, nameurban, namerural, plotbool):
    """
    Calculates for each seasons separately the UHI values for the urban stations. 
    Also the number of values used for each UHI-value is determined.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with datetime index and hourly temperature observations of urban station(s) and one rural station.
    nameurban : list of strings
        List with the names of the columns with the urban stations.
    namerural : string
        Name of the column with the rural station.

    Returns
    -------
    UHI_seasons : list of dataframes
        List of 4 pandas dataframes, one for each season in the order ['winter', 'spring', 'summer', 'autumn'].
        The dataframe has as index the hours (0-23) and the urban stations as columns, and contains the UHI-values.
    count_seasons : list of series
        List of 4 pandas series, one for each season in the order ['winter', 'spring', 'summer', 'autumn'].
        The dataframe has as index the hours (0-23) and contains the number of values used to calculate the UHI-value (which is the same for each station).

    """

    # Select relevant data
    nameurban = nameurban.copy()
    nameurban.append(namerural)
    df_relevant = df.loc[:, nameurban].copy()
    
    # Split into seasons
    seasons = sliceSeasons(df_relevant)
    
    # Setup mutliple dataframes
    listofseasons = ['winter', 'spring', 'summer', 'autumn']
    UHI_seasons = list()
    count_seasons = list()

    # For each season: calculate UHI
    for count, df_season in enumerate(seasons):
        # Calculate temperature difference and drop column of rural reference station
        dfdiff = df_season.apply(lambda l: l - df_season[namerural])
        dfdiff.drop(columns=[namerural], inplace=True)

        # Calculate mean for every hour
        dfdiff['Hour'] = dfdiff.index.hour
        dfUHI = dfdiff.groupby('Hour').mean()
        dfcount = dfdiff.groupby('Hour').count()
        UHI_seasons.append(dfUHI)
        count_seasons.append(dfcount.iloc[:,0])

    return UHI_seasons, count_seasons


def Test_one_series_of_gaps(df, columns_urban, column_rural, P_begingap, histogram, settings, plot=True):
    """
    Calculates the UHI of a dataframe before and after the gap-filling of a series of gaps.
    The series of gaps is made based on a chance and a distribution of gaplengths.
    The UHI is calculated based on the original observations/ filled values at the location of the gaps.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with datetime index and hourly temperature observations of urban station(s) and one rural station.
    columns_urban : list of strings
        List with the names of the columns with the urban stations.
    column_rural : string
        Name of the column with the rural station.
    P_begingap : float
        Chance that a random datetime is the beginning of a gap.
    histogram : list of two elements
        Information about the probability distribution of the gaplengths.
        The first element gives the probability for each bin.
        The second element gives the edges of the bins. Bins include the left edge.
    settings : dictionary
        Values for the parameters of the gap-filling algorithm.
    plot : boolean, optional
        If plot==True, a plot is made for each urban station separately with the UHI of the observed values and the filled values for each season separately. 
        The default is True.

    Returns
    -------
    UHI_obs : list of dataframes
        UHI values for the observations.
        List of 4 pandas dataframes, one for each season in the order ['winter', 'spring', 'summer', 'autumn'].
        The dataframe has as index the hours (0-23) and the urban stations as columns, and contains the UHI-values.
    UHI_filled : list of dataframes
        UHI values for the filled values.
        List of 4 pandas dataframes, one for each season in the order ['winter', 'spring', 'summer', 'autumn'].
        The dataframe has as index the hours (0-23) and the urban stations as columns, and contains the UHI-values.

    """
    
    df = df.copy()
    list_stations = columns_urban + [column_rural]
    
    # MAKE SERIES OF GAPS
    # Based on chance and distribution, make gaps
    df_gapped = Make_gaps_based_on_distribution(df[list_stations], P_begingap, histogram, settings["threshold_minLP"]*24-1)
    # Add ERA5
    df_gapped = pd.concat([df_gapped, df[[station + '_ERA5' for station in list_stations]]], 
                              join= 'inner', 
                              axis=1)
    
    
    # PERFORM THE GF ALGORITHM
    df_filled = df_gapped
    for station in list_stations:
        df_filled = GF_algorithm(df_filled, station, station +'_ERA5', settings, False)
    
    
    # CALCULATE UHI
    # Calculate the UHI of the original data
    df_onlygaps = df.loc[df_filled.index,:].loc[df_filled[list_stations[0]].isna(),:]
    UHI_obs, count_obs = Calculate_UHI_onekind(df_onlygaps, columns_urban, column_rural, False)

    # Calculate the UHI of the filled data
    df_filled_onlygaps = df_filled.loc[df_filled[list_stations[0]].isna(),:]
    columns_urban_filled = [station + '_FILLED' for station in columns_urban]
    column_rural_filled = column_rural + '_FILLED'
    UHI_filled, count_filled = Calculate_UHI_onekind(df_filled_onlygaps, columns_urban_filled, column_rural_filled, False)
    
    print('Count of datapoints for UHI:')
    print(count_filled)
    
    
    # MAKE PLOT OF RESULTS
    if plot == True:
        for station in columns_urban:
            # Select the data and put it into one dataframe for every station
            df_UHI = pd.DataFrame()
            df_UHI.loc[:, 'winter'] = UHI_obs[0].loc[:,station]
            df_UHI.loc[:, 'winter_FILLED'] = UHI_filled[0].loc[:, station+'_FILLED']
            df_UHI.loc[:, 'spring'] = UHI_obs[1].loc[:, station]
            df_UHI.loc[:, 'spring_FILLED'] = UHI_filled[1].loc[:, station + '_FILLED']
            df_UHI.loc[:, 'summer'] = UHI_obs[2].loc[:, station]
            df_UHI.loc[:, 'summer_FILLED'] = UHI_filled[2].loc[:, station + '_FILLED']
            df_UHI.loc[:, 'autumn'] = UHI_obs[3].loc[:, station]
            df_UHI.loc[:, 'autumn_FILLED'] = UHI_filled[3].loc[:, station + '_FILLED']

            df_UHI_obs= df_UHI.loc[:,['winter', 'spring', 'summer', 'autumn']]
            df_UHI_FILLED = df_UHI.loc[:, ['winter_FILLED', 'spring_FILLED', 'summer_FILLED', 'autumn_FILLED']]

            df_count = pd.DataFrame()
            df_count.loc[:, 'winter'] = count_obs[0]
            df_count.loc[:, 'spring'] = count_obs[1]
            df_count.loc[:, 'summer'] = count_obs[2]
            df_count.loc[:, 'autumn'] = count_obs[3]

            # Set up the subplots
            fig, (ax1, ax2) = plt.subplots(2, sharex=True,gridspec_kw={'height_ratios': [3, 1]})

            # Plot the UHI
            for count, df in enumerate([df_UHI_obs, df_UHI_FILLED]):
                df.loc[:,'Hour'] = df.index
                dfm = df.melt('Hour', var_name='Season', value_name='UHI')
                if count==0:
                    sns.lineplot(ax=ax1, data=dfm, x= 'Hour', y='UHI', hue='Season', linestyle='solid').set(title='UHI '+station)
                    # plt.legend()
                else:
                    sns.lineplot(ax=ax1, data=dfm, x='Hour', y='UHI', hue='Season', linestyle='dashed').set(title='UHI '+station)
                    # plt.legend()

            # Plot the number of missing values
            ax2.plot(df_count, marker='o')
            ax2.set_ylabel('$N$')

            # Settings for plot
            ax2.set_xlabel('Hour')
            ax1.set_ylabel('UHI (°C)')

            pos1 = ax1.get_position()
            ax1.set_position([pos1.x0, pos1.y0, pos1.width * 0.75, pos1.height])
            pos2 = ax2.get_position()
            ax2.set_position([pos2.x0, pos2.y0, pos2.width * 0.75, pos2.height])

            handles, labels = ax1.get_legend_handles_labels()
            for i in range(4,8):
                handles[i].set_linestyle('--')

            ax1.legend(bbox_to_anchor=(1, 1))
            
       
    # RETURN UHI VALUES
    return UHI_obs, UHI_filled
    
    
    
def Test_multiple_series_of_gaps(df, columns_urban, column_rural, P_begingap, histogram, settings, repetitions, plot=True):
    """
    Calculates the mean UHI before and after the gap-filling of a series of gaps.
    The construction of the series of gaps and calculation of UHI is repeated multiple times. In the end the mean UHI is calculated.
    The series of gaps is made based on a chance and a distribution of gaplengths.
    The UHI is calculated based on the original observations/ filled values at the location of the gaps.
    

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with datetime index and hourly temperature observations of urban station(s) and one rural station.
    columns_urban : list of strings
        List with the names of the columns with the urban stations.
    column_rural : string
        Name of the column with the rural station.
    P_begingap : float
        Chance that a random datetime is the beginning of a gap.
    histogram : list of two elements
        Information about the probability distribution of the gaplengths.
        The first element gives the probability for each bin.
        The second element gives the edges of the bins. Bins include the left edge.
    settings : dictionary
        Values for the parameters of the gap-filling algorithm.
    repetitions : integer
        Number of repetitions of the procedure.
    plot : boolean, optional
        If plot==True, a plot is made for each urban station separately with the UHI of the observed values and the filled values for each season separately. 
        The default is True.

    Returns
    -------
    UHI_obs : list of dataframes
        UHI values for the observations.
        List of 4 pandas dataframes, one for each season in the order ['winter', 'spring', 'summer', 'autumn'].
        The dataframe has as index the hours (0-23) and the urban stations as columns, and contains the UHI-values.
    UHI_filled : list of dataframes
        UHI values for the filled values.
        List of 4 pandas dataframes, one for each season in the order ['winter', 'spring', 'summer', 'autumn'].
        The dataframe has as index the hours (0-23) and the urban stations as columns, and contains the UHI-values.

    """
    
    idx = pd.IndexSlice
    
    # Create dataset to store results
    
    index_season = ['winter', 'spring', 'summer', 'autumn']
    index_stations = columns_urban
    index_repetitions = np.arange(repetitions)
    
    indices = [index_season, index_stations, index_repetitions]
    
    multiindex = pd.MultiIndex.from_product(indices, names=["season", "station", "repetition"])
    
    UHI_obs_all = pd.DataFrame(index=np.arange(24), columns = multiindex)
    UHI_filled_all = pd.DataFrame(index=np.arange(24), columns = multiindex)
    
    
    # Perform for each repetitions
    for i in np.arange(repetitions):
        print('Start of repetition number '+ str(i))
        UHI_obs, UHI_filled = Test_one_series_of_gaps(df, columns_urban, column_rural, P_begingap, histogram, settings, plot=False)
        # print(UHI_obs)
        for number, season in enumerate(index_season):
            for station in columns_urban:
                UHI_obs_all.loc[:, idx[season, station, i]]=UHI_obs[number].loc[:,station]
                UHI_filled_all.loc[:, idx[season, station, i]]=UHI_filled[number].loc[:,station+'_FILLED']
    
    # print(UHI_obs_all)
    # print(UHI_filled_all)
    
    # Calculate mean and standard error
    UHI_obs_calculations = UHI_obs_all.T.groupby(level=[0,1]).agg(['mean', 'std'])
    UHI_obs_calculations.loc[:,idx[:,'std']]=UHI_obs_calculations.loc[:,idx[:,'std']] / np.sqrt(repetitions)
    
    UHI_filled_calculations = UHI_filled_all.T.groupby(level=[0,1]).agg(['mean', 'std'])
    UHI_filled_calculations.loc[:,idx[:,'std']]=UHI_filled_calculations.loc[:,idx[:,'std']] / np.sqrt(repetitions)
    
    # print(UHI_obs_calculations)
    # print(UHI_filled_calculations)
    
    return UHI_obs_calculations, UHI_filled_calculations
    
    
        
    
    
    

    
    
    
    
