
import pandas as pd
import os

def read_csv(path, dtindex=True):
    """
    Reads in a csv-file to pandas dataframe.

    Parameters
    ----------
    path : string
        Path to the file (folder + name of file + .csv).
        File has to have datetime column.
    dtindex : boolean, optional
        Indication if the index of the dataframe should be the datetimes or not. 
        The default is True. 
        In general dtindex is only put to False when handling raw data, where duplicated timestamps are possible.

    Returns
    -------
    csvfile : pandas dataframe
        Pandas dataframe with the data.

    """
    
    csvfile = pd.read_csv(path, sep=',', parse_dates=['DateTime'])
    if dtindex:
        csvfile.set_index(csvfile.columns[0], inplace=True)
    print('COMMENT: The file ' + os.path.basename(path) + ' is sucessfully read in.')
    
    return csvfile
