"""

This file contains functions to perform the gap-filling algorithm on a dataset with a series of gaps

"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


def GF_debmodelMeanbias_algorithm(df, name_gapped, name_model, LP, time_variation = 0, positioning = 'both', plot = False):
    """
    Performs gap-filling bij debiasing model data by calculating a mean bias (which depends on the hour of the day) for the model data of a certain learning period.
    The learning period is given as one of the arguments, and is not determined by this function.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with the gapped data and one column with model data.
        The gapped data can only contain one gap!
    name_gapped : string
        Name of the column with the gapped data.
    name_model : string
        Name of the column with the model data.
    LP: list of dates
        List with two elements. The first/second element is the datetime of the beginning/end of the learning period.
    time_variation : integer, optional
        Variation in hour when calculating the mean bias for a certain hour X. In practice this is done by recalculating the mean bias in a rolling window.
        Example:
        For the mean bias of hour X, the values which belong to a datetime with hour within the time window [X-time_variation, X+time_variation] will contribute.
        If time_variation > 11, there won't be a distinction between the different hours of the gap and each hour has the same mean bias.
        The default is 0.
    positioning : string
        Positioning of the learning period. Possibilities:
            -'both': the LP is placed symmetrical around the gap
            -'separate': the LP is divided in two parts, one part before the gap and one part after the gap
        The default is 'both'.
    plot : boolean, optional
        If plot==True, a plot is made of the mean bias in function of the hour of the day.
        The default is False.

    Returns
    -------
    df : pandas dataframe
        The original dataframe, but with an extra column names 'P_debmodelMeanbias' with the gap filled data.

    """

    df = df.copy()  # Take a copy or original df in main will also be altered

    # 1. Determine begin and end of gap
    datesgap = df.index[pd.isnull(df[name_gapped])]
    begingap = datesgap[0]
    endgap = datesgap[-1]


    # 2. Determine LP
    beginLP = LP[0]
    endLP = LP[1]
    
    datesLP_L = pd.date_range(start=beginLP, end=begingap, freq='H', name='DateTime', inclusive="left")
    datesLP_R = pd.date_range(start=endgap, end=endLP, freq='H', name='DateTime', inclusive="right")
    
    if positioning == 'both':
       LP = [datesLP_L.union(datesLP_R)]
    elif positioning == 'separate':
       LP = [datesLP_L, datesLP_R]
    else:
       print('ERROR: positioning must be both or seperate. Using both as default.')
       LP = [datesLP_L.union(datesLP_R)]
    
    
    # 3. Debias ERA5
    
    # Initialise the dataframe to store debiased values
    debias = pd.DataFrame(index=datesgap)
    debias.loc[:, 'ERA5'] = df.loc[datesgap, name_model]
    
    # Debias for each part of the learning period separately
    for i, datesLP in enumerate(LP):
        # Calculate difference for each timestamp of the learning period
        bias= pd.DataFrame(index=datesLP)
        bias.loc[:,'Bias']=df.loc[datesLP, name_model]-df.loc[datesLP, name_gapped]
        bias.loc[:, 'Hour']=bias.index.map(lambda x: x.hour)

        # Calculate the mean bias for each hour
        biasHour = bias.groupby('Hour').mean()

        # Calculate mean bias in time window
        if time_variation > 11:
            biasHour.loc[:, 'Bias']=biasHour['Bias'].mean()
        elif time_variation!=0:
            # Determine size of time window
            rollingW=1+2*time_variation
            # Extend overview of mean bias per hour because we have periodic conditions
            biasHourpast=biasHour.copy()
            biasHourpast.set_index(biasHourpast.index.to_series() - 24, inplace=True)
            biasHourfuture=biasHour.copy()
            biasHourfuture.set_index(biasHourfuture.index.to_series() + 24, inplace=True)
            biasextend=pd.concat([biasHourpast, biasHour, biasHourfuture])
            # Determine mean using rolling window en drop the extensions
            biasHour = biasextend.rolling(rollingW, min_periods=rollingW, center=True).mean().drop(biasHourpast.index.union(biasHourfuture.index))
            
        # Make a plot of the mean bias per hour
        if plot==True:
            plt.plot(biasHour.index, biasHour.Bias,'o')
            plt.xlabel('Hour')
            plt.ylabel('Bias')
            plt.title('Mean bias of learning period number ' +str(i))
            plt.show()

        # Debias ERA5 with the calculated mean bias
        debias.loc[:, 'debmodel_' + str(i)] = debias.apply(lambda x: (x.ERA5 - biasHour.loc[x.name.hour, 'Bias']), axis='columns')


    # 4. Determine final debiased value
    
    if positioning == 'both':
        debias.rename(columns={'debmodel_0': 'debmodel'}, inplace=True)
    elif positioning == 'separate':
        # Calculate weights for left and right
        if len(datesgap) == 1:
            wL = [0.5]
            wR = [0.5]
        else:
            wR = np.linspace(0, 1, num=len(datesgap), endpoint=True)
            wL = 1 - wR
        weights = pd.DataFrame(list(zip(wL, wR)), index=datesgap, columns=['wL', 'wR'])
        # Multiply each column with their weights and add the two columns up
        debias.loc[:, 'debmodel_0'] = debias.loc[:, 'debmodel_0'] * weights.wL
        debias.loc[:, 'debmodel_1'] = debias.loc[:, 'debmodel_1'] * weights.wR
        debias.loc[:, 'debmodel'] = debias.loc[:, 'debmodel_0'] + debias.loc[:, 'debmodel_1']
        

    # 5. Fill in debiased ERA5 to fill gap
    
    df.loc[:, 'P_debmodelMeanbias'] = df.loc[:, name_gapped]
    df.loc[datesgap, 'P_debmodelMeanbias'] = debias.debmodel
    

    return df




def GF_algorithm(df, name_gapped, name_model, settings, print_infogaps=True):
    """
    Performs gap-filling on a timeseries with a series of gaps.
    !!! The algorithm assumes hourly data !!!
    First all the gaps (and their gaplength) are determined. Based on the gaplenght, the gaps are filled with a certain gap-filling technique.
    1. The small gaps are filled by performing linear interpolation.
    2. Larger gaps are filled by performing the debiasing method by calculating a mean bias.
        - A distiction is made between semi-large gaps, for which the 'both' positioning is selected,
          and large gaps, for which the 'separate' positioning is selected.
        - The learning period (LP) of the debiasing method consists of available timestamps before and after the gap till the previous/next gap.
          If possible the learning period is placed symmetrically, but if not possible a shift or shrinkage of the learning period is possible.
          Positioning of LP:
              1. Try to place LP symmetrical around gap
              2. If this is not possible: shift the learning period to the right/left without adjusting size of LP
              3. If this is not possible: shrink the LP (but never below treshold of minimum length of LP)
              4. If this is not possible: allow to use previous filled gaps by debiasing method. Still try to place the LP as symmetrical as possible around gap.
        
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with DateTime as index and at least one column with the gapped data and one column with model data.
    name_gapped : string
        Name of the column with the gapped data.
    name_model : string
        Name of the column with the model data.
    settings : dictionary
        Dictionary with the settings for the gap-filling algorithm. Possiblities:
             - "threshold_LI": for gaps smaller than treshold_LI (in amount of hours), LI will be applied
               "threshold_bs": for gaps with threshold_bs <= length, separate positioning is used
               "seasonal_span": seasonal span for the debiasing method
               "time_variation": time variation for the debiasing method
               "threshold_minLP": minimum amount of days needed for the learning period for the debiasing method
               "threshold_minLPoneside": minimum amount of days needed on one side of the gap to perform the debiasig method with separate positioning
    print_infogaps : boolean, optional
        If this is true, a list of the gaps with startdate and length is printed. The default is True.

    Returns
    -------
    df : pandas dataframe.
        The original dataframe, but with an extra column named name_gapped+'_FILLED' with the gap filled data.

    """

    
    # 0. Select the timeseries which will be filled
    df=df.copy()
    obs=df[[name_gapped]] 


    # 1. Set values of some parameters
    seasonal_span = settings['seasonal_span']
    time_variation = settings['time_variation']
    threshold_LI = settings['threshold_LI']
    threshold_minLP = settings['threshold_minLP']        
    threshold_minLPoneside = settings['threshold_minLPoneside']
    threshold_bs = settings['threshold_bs']


    # 2. Determine the gaps (location + gaplength + GF-technique)

    # Determine location and length
    df_gapped = df[[name_gapped]].isnull().astype(int)
    df_gapped['mask']=(df_gapped != df_gapped.shift()).cumsum() # Give each series of values of series of NaN's a number
    df_gapped['DateTime']=df_gapped.index
    gaps = df_gapped.groupby('mask').agg({'DateTime': 'first', name_gapped: 'sum'})
    infogaps = gaps.loc[gaps[name_gapped] != 0,:].reset_index().drop('mask', axis=1)
    infogaps.rename(columns={name_gapped: 'Length'}, inplace=True)    
    
    # Determine the technique for every gap
    infogaps.loc[:, 'Technique'] = 'MB_b'
    infogaps.loc[infogaps.Length<threshold_LI, 'Technique'] = 'LI'
    infogaps.loc[infogaps.Length>=threshold_bs, 'Technique'] = 'MB_s'

    # Print the info about the gaps
    if print_infogaps:
        print(infogaps)


    # 3. For small gaps: fill with linear interpolation
    
    # Create dataframe to store gap-filling + dataframe with interpolation values
    obs_afterLI = obs.copy()
    obs_LI = obs.interpolate()
    
    # For each small gap: store the linear interpolation values
    infogaps_LI = infogaps.loc[infogaps.Technique == 'LI', :]
    for i in infogaps_LI.index:
        DT_range = pd.date_range(infogaps_LI.loc[i, 'DateTime'], periods = infogaps_LI.loc[i, 'Length'], freq="H").tolist()
        obs_afterLI.loc[DT_range,:]=obs_LI.loc[DT_range,:]
    df.loc[:,'obs_afterLI'] = obs_afterLI
    

    # 4. For all other gaps: fill in with meanbias technique
    
    # Create dataframe to store gap-filling
    df.loc[:,'obs_afterMB']=df.loc[:,'obs_afterLI'].copy()
    
    # For each larger gap: perform the meanbias gap-filling technique
    infogaps_MB = infogaps.loc[infogaps.Technique.isin(['MB_b', 'MB_s']),:].reset_index(drop=True)
    for i in infogaps_MB.index:
        # Determine begin and end of gap
        begingap = infogaps_MB.loc[i, 'DateTime']
        endgap = begingap + dt.timedelta(hours=int(infogaps_MB.loc[i, 'Length'] - 1))

        # Determine amount of time till the previous and next gap
        # !!! gaps filled with LI will from now on not be seen as gaps, but as normal data. !!!
        # (This means that the result of the LI can be used as part of a LP.)
        if i == 0:
            diff_previous = begingap - obs.index[0]
        else:
            diff_previous = diff_next
        if i == infogaps_MB.index[-1]:
            diff_next = obs.index[-1] - endgap
        else:
            beginnextgap = infogaps_MB.loc[infogaps_MB.index[i+1], 'DateTime']
            diff_next = beginnextgap - endgap - dt.timedelta(hours=1)

        # Determine begin and end of learning period
        UsePreviousValues = False
        if (diff_previous >= dt.timedelta(days=seasonal_span/2) and diff_next >= dt.timedelta(days=seasonal_span/2)):
            # both sides have no problem
            beginLP = begingap - dt.timedelta(days=seasonal_span/2)
            endLP = endgap + dt.timedelta(days=seasonal_span/2)
        elif (diff_previous < dt.timedelta(days=seasonal_span/2) and diff_next >= dt.timedelta(days=seasonal_span)-diff_previous):
            # left side has problem, but fixable by applying a shift
            beginLP = begingap - diff_previous
            endLP = endgap + (dt.timedelta(days=seasonal_span)-diff_previous)
        elif (diff_previous >= dt.timedelta(days=seasonal_span)-diff_next and diff_next < dt.timedelta(days=seasonal_span/2)):
            # right side has problem, but fixable by applying a shift
            beginLP = begingap - (dt.timedelta(days=seasonal_span)- diff_next)
            endLP = endgap + diff_next
        elif diff_previous+diff_next>=dt.timedelta(days=threshold_minLP):
            # problem is not fixable by applying a shift, but can be fixed by applying a shrinkage + shift
            print("WARNING: For gap with DateTime " + str(infogaps_MB.loc[i, 'DateTime']) + ": LP is only " + str(diff_previous + diff_next) + " long.")
            beginLP = begingap - diff_previous
            endLP = endgap + diff_next
        elif diff_previous+diff_next<dt.timedelta(days=threshold_minLP) and begingap-obs.index[0]+diff_next>=dt.timedelta(days=threshold_minLP):
            # prbolem is not fixabel by applying a shift and/or schrinkage
            print("WARNING: For gap with DateTime " + str(infogaps_MB.loc[i, 'DateTime']) + ": earlier filled data will be used.")
            UsePreviousValues = True
            beginLP= begingap - min((dt.timedelta(days=seasonal_span) - diff_next), begingap-obs.index[0])
            endLP= endgap + diff_next
        elif diff_previous+diff_next<dt.timedelta(days=threshold_minLP) and begingap-obs.index[0]+diff_next<dt.timedelta(days=threshold_minLP):
            print("WARNING: First gap in dataset can not be filled (not enough LP)")
            # TO DO: A solution needs to be programmed
        else:
            print("WARNING: We have another situation")

        # Select the data needed for the gap filling technique
        DT_LPandgap = pd.date_range(start=beginLP, end=endLP, freq='H', name='DateTime', inclusive="both")
        if UsePreviousValues == False:
            df_onegap = df.loc[DT_LPandgap, ['obs_afterLI', name_model]]
            df_onegap.rename(columns= {'obs_afterLI': 'obs_gapped'}, inplace=True)
        else:
            df_onegap = df.loc[DT_LPandgap, ['obs_afterMB', name_model]]
            df_onegap.rename(columns={'obs_afterMB': 'obs_gapped'}, inplace=True)

        # Perform gap filling with meanbias
        if infogaps_MB.Technique.iloc[i] == 'MB_b':
            result = GF_debmodelMeanbias_algorithm(df_onegap, 'obs_gapped', name_model, [beginLP, endLP], time_variation, positioning='both')
        elif infogaps_MB.Technique.iloc[i] == 'MB_s':
            if (diff_previous>=dt.timedelta(days=threshold_minLPoneside) and diff_next>=dt.timedelta(days=threshold_minLPoneside)):
                result = GF_debmodelMeanbias_algorithm(df_onegap, 'obs_gapped', name_model, [beginLP, endLP], time_variation, positioning='separate')
            else: # In this case the part of the LP before/after the gap is not big enough to perform a statistically correct separate positioning
                result = GF_debmodelMeanbias_algorithm(df_onegap, 'obs_gapped', name_model, [beginLP, endLP], time_variation,
                                             positioning='both')
                print("WARNING: For gap with DateTime " + str(infogaps_MB.loc[i, 'DateTime']) + ": 'both' is applied instead of 'separate'.")
       
        # Store results in dataframe
        DT_gap = pd.date_range(start=begingap, end=endgap, freq='H', name='DateTime', inclusive="both")
        df.loc[DT_gap, 'obs_afterMB'] = result.loc[DT_gap, 'P_debmodelMeanbias']
        

    # 5. Finalise the dataframe
    df.rename(columns={'obs_afterMB':name_gapped+'_FILLED'}, inplace=True)
    df.drop(columns=['obs_afterLI'], inplace=True)
    # print(df)

    return df