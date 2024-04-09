# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:17 2024

@author: ambjacob
"""
import pandas as pd
from GF_evaluation import *

def Test_parameters(df, name_fulldata, name_model, sv=list(range(15,55,10)), tv=0, positioning='both', error = 'MSE', gaplengths=list(range(5,20,1)), weights=False, repetitions=1000, check=250, plot=True):
    """ Performs the gap filling technique GF_debmodelMeanbias with different values for a certain parameter. The
        technique is performed for multiple gaps and for multiple gap lengths.
        df is the dataframe and has DateTime as index and at least a column with the original data and a column with the
        model data.
        name_fulldata is the name of the column with the complete and original data.
        name_model is the name of the column with the model data.
        sv, tv and positioning can be an integer/string or a list. If an integer/string, this value
        is taken for this parameter. If a list is given, the value of this parameter is varied over the values in the
        list. If two parameters are given as lists, a heatmap is shown where both parameters are varied (not implemented yet).
        error is the kind of error which is plotted
        gaplengths is a list and consists of the gaplengths for which the technique will be performed. (for example:
        list(range(5, 21,1)) ).
        weights is a list and consists of the weights used to calculate the mean error. When making a heatmap, a mean
        will be taken over the errors of all the gaplengths. This will be a weighted mean with the weights given in this
        list. This list must have the same length as gaplengths, and the sum of the weights must be 1.
        repetitions is the amount of gaps that are made for each gaplength.
        check determines when something is printed on the screen (to track the progress of the code)
        If plot==True, the error is plotted in function of the value of the parameter.
        Example: Examine_parameters(Melle, 'Melle_synoptic', 'Melle_ERA5', 35, [0,2,4,6,8], 'left', 'MSE', list(range(4,200,40), 2000, 250, True)"""
    
                                    
    # Check which parameter is given as list (and will be varied)
    list_parameters = [sv, tv, positioning]
    list_islist = [isinstance(i, list) for i in list_parameters]
    amount_varpar = sum(list_islist)

    if amount_varpar==3:
        print("ERROR: Examine_parameters can only vary one or two parameters at the same time.")

    elif amount_varpar==0:
        print("ERROR: At least one of the parameters needs to be a list.")

    elif amount_varpar==1:
        # Prepare dataframe to store error
        rangeofvalues = [par for count, par in enumerate(list_parameters) if list_islist[count]][0]
        df_allerrors = pd.DataFrame(index=gaplengths, columns=rangeofvalues)
        df_allstd = pd.DataFrame(index=gaplengths, columns=rangeofvalues)

        for value in rangeofvalues:
            print('Running for value of parameter: ' + str(value))
            # For every value of the parameter: perform meanbias technique for different gaplengths and multiple gaps
            if list_islist[0]==True:
                # Determine the number of days the gap must stay from the ends of the data set
                if positioning in ('left', 'right'):
                    cutoff = value
                elif positioning in ('both', 'separate'):
                    cutoff = value / 2
                # Perform meanbias technique
                df_result, df_std = Test_techniques_differentgaplengths(df, name_fulldata, name_model, {
                    "debmodelMeanbias": [value, tv, positioning, False]}, int(np.ceil(cutoff)), error, gaplengths, repetitions,
                                                                        check, plot=False)
            elif list_islist[1]==True:
                # Determine the number of days the gap must stay from the ends of the data set
                if positioning in ('left', 'right'):
                    cutoff = sv
                elif positioning in ('both', 'separate'):
                    cutoff = sv / 2
                # Perform meanbias technique
                df_result, df_std = Test_techniques_differentgaplengths(df, name_fulldata, name_model, {
                    "debmodelMeanbias": [sv, value, positioning, False]}, int(np.ceil(cutoff)), error, gaplengths,
                                                                        repetitions, check, plot=False)
            elif list_islist[2]==True:
                # Determine the number of days the gap must stay from the ends of the data set
                if value in ('left', 'right'):
                    cutoff = sv
                elif value in ('both', 'separate'):
                    cutoff = sv / 2
                # Perform meanbias technique
                df_result, df_std = Test_techniques_differentgaplengths(df, name_fulldata, name_model, {
                    "debmodelMeanbias": [sv, tv, value, False]}, int(np.ceil(cutoff)), error, gaplengths, repetitions,
                                                                        check, plot=False)

            # Store the results in dataframe
            df_allerrors.loc[:, value] = df_result['debmodelMeanbias']
            df_allstd.loc[:, value] = df_std['debmodelMeanbias']
            

        # If asked, plot the error in function of gaplength for every value of the parameter
        if plot == True:
            df_allerrors.plot(marker='o', yerr=df_allstd, capsize=10)
            plt.xlabel('Gaplength (hour)')
            plt.ylabel(error)
            if error == 'MBE':
                plt.axhline(color='k')
            plt.show()

    elif amount_varpar==2:
        # Prepare dataframe to store error
        rangeofvalues = [par for count, par in enumerate(list_parameters) if list_islist[count]]
        df_allerrors = pd.DataFrame(index=rangeofvalues[0], columns=rangeofvalues[1])
        df_allstd = pd.DataFrame(index=rangeofvalues[0], columns=rangeofvalues[1])
        for value0 in rangeofvalues[0]:
            print('Running for value of first parameter: ' + str(value0))
            for value1 in rangeofvalues[1]:
                print('Running for value of second parameter: ' + str(value1))
                if (list_islist[0] and list_islist[1]) ==True:
                    df_result, df_std = Test_techniques_differentgaplengths(df, name_fulldata, name_model, {
                        "debmodelMeanbias": [value0, value1, positioning, False]}, int(np.ceil(value0/2)), error, gaplengths, repetitions,
                                                                            check, plot=False)
                elif (list_islist[0] and list_islist[2]) ==True:
                    df_result, df_std = Test_techniques_differentgaplengths(df, name_fulldata, name_model, {
                        "debmodelMeanbias": [value0, tv, value1, False]}, int(np.ceil(value0/2)), error, gaplengths, repetitions,
                                                                            check, plot=False)
                elif (list_islist[1] and list_islist[2]) ==True:
                    df_result, df_std = Test_techniques_differentgaplengths(df, name_fulldata, name_model, {
                        "debmodelMeanbias": [sv, value0, value1, False]}, int(np.ceil(sv/2)), error, gaplengths, repetitions,
                                                                            check, plot=False)

                # Take mean over all gaplengths:
                sum_weights=weights.sum()
                if round(sum_weights, 3)!=1:
                    print('ERROR: sum of weights must be 1, sum of weights is ' + str(sum_weights))
                df_result.loc[:, 'weights'] = weights
                df_result.loc[:, 'weighted_results'] = df_result.debmodelMeanbias * df_result.weights
                weightedmean= df_result.weighted_results.sum()

                # Store the results in dataframe
                df_allerrors.loc[value0, value1] = weightedmean



        # If asked, make a heatmap
        if plot == True:
            # Take copy of dataframe (no clue why, but it won't work if I don't do this: error about not able to convert to float)
            df_copy = pd.DataFrame(df_allerrors.values.tolist(), index=df_allerrors.index, columns=df_allerrors.columns)
            print(df_copy)
            # Plot errors in heatmap
            ax = sns.heatmap(df_copy, cbar_kws={'label': error}, cmap="Blues")
            nameofparameters = [par for count, par in enumerate(['sv (days)', 'tv (hour)', 'positioning']) if list_islist[count]]
            plt.xlabel(nameofparameters[1])
            plt.ylabel(nameofparameters[0])
            ax.invert_yaxis()
            plt.show()


    return df_allerrors, df_allstd

