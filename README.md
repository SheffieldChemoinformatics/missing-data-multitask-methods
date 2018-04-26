# Introduction

This repository contains the data sets, the code and the results that were reported on the article "Effect of missing data on multitask prediction methods".
This material is made available to aid reproduction of our work.

# Contents

## Data

This folder contains the three data sets used for the analysis reported: PKIS, HTSFP5, and HTSFP10.
This data was collected from ChEMBL and PubChem.
Label and descriptor data are provided as Gzipped CSV files.
The format of the files is as follows:
- the first column corresponds to the compound ID
- then a set of columns describing the outputs or targets
- finally a set of columns representing the fingerprint of the compounds, labelled D_0 to D_N, where N is the fingerprint length 
Additionally, SMILES are provided in a separate file linked to the compound ID.

## Code

Contains the files used to perform the DNN and the Macau computations.
The different files are:
- helper_functions.py: is a file with different functions used in the other files
- RandomForest.py: generates Random Forest models both for PKIS and HTSFP sub sets
- DNN.py: generates DNNs using Tensorflow and was used both for PKIS and HTSFP sub sets
- MacauRegression.py: generates Macau models and provides regression results (used on PKIS data)
- MacauClassification.py: generates Macau simulating a classification procedure (used on HTSFP subsets)
- plotFigures.py: generates the figures that were included in the article
- general.mplstyle: style file for matplotlib to format the plots

Both the DNN and the Macau programs require only one argument when called.
This argument is the path to the arguments file.
The arguments file is a JSON file that contains the parameters needed to run the models.
Required parameters are:
- training file: a file URL
- testing file: can be a file URL or a number. If it is number, it determines the seed that will split the data set given in training file intro train and test sets.
- output_file: a file URL
- prediction_mode: wether classification or regression is performed
For a list with most parameters see DNN.py and MacauRegression.py, at the top of the file is a DEFAULT_MAP dictionary that contains all parameters and their default value, as well as comments explaining their meaning.
An exemplary JSON file for DNN and Macau is available in the code folder.
If the value of a parameter is a list, and the option to read parameters from a file is not selected, the program will generate random values based on the list at each iteration.

These programs require the following libraries:
- numpy
- pandas
- sklearn
- scipy
- tensorflow (for DNN.py)
- macau (for Macau*.py)
- matplotlib (for plotFigures.py)

## Results

In this folder are the result files that were outputted by the programs and used in our analysis.
This folder is organized first by dataset and then by technique.
Inside most folder are 36 files:
- 10 correspond to the label removal model ran on each of the sets of hyperparameters
- 10 correspond to the compound removal model, again one per set of hyperparameters
- 16 correspond to the seed variation test, where the first number is the seed of the train/test split and the second number is the seed of the label removal process
Folder for Random Forest only contain 10 files, those of the compound removal model
DNN and Macau folders for PKIS contain 10 additional files, those of the assay removal model

In each output file the following structure is used.
Each line represents a model (DNN or Macau).
The first columns represent all parameters of each program.
Next column is the time to train and evaluate the model.
Then the results of the model are provided per target.
For each target, measure values for the training and test set are given.
Regression measures provided: R2(coefficient of determination), r2(square of the correlation coefficient), RMSD, MAE.
Classification measures provided: precision, recall, fscore (F1-score), mcc.








