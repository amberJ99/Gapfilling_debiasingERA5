"""
This file contains the code to make the figures of the paper.
"""

#%% SET-UP

# Import all needed packages and files
import sys
from Visualize_gaps import*

# Define path
sys.path.append(r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5")

#%% Figure 1: gaps of MOCCA

main_path = r"C:\Users\ambjacob\Documents\Python_projecten\Gapfilling_debiasingERA5"

df_allstations = Put_stations_together(main_path)
Plot_position_gaps(df_allstations, main_path)
# Plot_both_distributions(df_allstations, main_path)

#%% Figure 2: 