-------------------------
Gapfilling_debiasingERA5
-------------------------

This code supports the publication of a paper related to gap-filling of urban temperature time series, by debiasing ERA5 reanalysis data.

Publication:
	Amber Jacobs, Sara Top, Thomas Vergauwen, Steven Caluwaerts, ...
	Filling gaps in urban temperature observations by debiasing ERA5 reanalysis data
	Urban Climate

----------------------------------------------------------------------------------------------------------------------------------

Important remark: functions only work properly for hourly temperature time series!!!

----------------------------------------------------------------------------------------------------------------------------------

The different python files consist functions to:
	- perform GF, with separate techniques (only for datasets with single gap) or with the algorithm (for datasets with series of gaps)
	- evaluate the GF by making artificial gaps in a complete dataset
	- visualize the results

The main folder also needs to consists of following folders:
	- Data: in this folder you need to place the datasets
	- Figures: in this folder the figures will be saved as png-file
	- Results: in this folder a csv-file will be written with the results of the evaluation procedures

The main folder consists of following python files:
	- (Evaluation_GFalgorithm)
		Functions to perform the evaluation of the GF algorithm
	- (Evaluation_GFparameters)
		Functions to perform the evaluation of the selection parameters
	- (Evaluation_GFtechniques)
		Functions to perform the evaluation of the GF techniques
	- Execution_Evaluation_GFalgorithm
		Code to perform the evaluation of the GF algorithm, by looking at the orginal UHI and the estimated UHI
	- Execution_Evaluation_GFparameters
		Code to perform the evaluation of the different selection parameters of the MB debiasing technique in function of gaplength
	- Execution_Evaluation_GFtechniques
		Code to perform the evaluation of the GF techniques next to each other in function of gaplength, for a constant set of selection parameters
	- Execution_make_figures
		Code to make the figures of the results
	- (GFalgorithm)
		Functions to perform the GF algorithm
	- (GFtechniques)
		Functions to perform the GF techniques
	- (Read_file)
		Function to read csv-file
	- (Visualize_gaps)
		Functions to visualize the location and distribution of gaps

Files with Execution_* are files that you need to run to perform the evaluations and to make the figures. These files will use functions defined in the other python-files.
Files with () are files with the definition of functions. 


