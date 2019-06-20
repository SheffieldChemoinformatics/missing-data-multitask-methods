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

### Installation

After following the instructions on https://github.com/jaak-s/macau to install the `macau` package, the code can be
installed directly [GitHub](https://github.com/SheffieldChemoinformatics/missing-data-multitask-methods) with:

```bash
$ pip install git+https://github.com/SheffieldChemoinformatics/missing-data-multitask-methods.git
```

It can also be installed in development mode with:

```bash
$ git clone https://github.com/SheffieldChemoinformatics/missing-data-multitask-methods.git
$ cd missing-data-multitask-methods
$ pip install -e .
```


### Usage

The library contains the files used to perform the DNN and the Macau computations.

Both the DNN and the Macau programs require only one argument when called.
This argument is the path to the arguments file.
The arguments file is a JSON file that contains the parameters needed to run the models.
Required parameters are:
- training file: a file URL
- testing file: can be a file URL or a number. If it is number, it determines the seed that will split the data set given in training file intro train and test sets.
- output_file: a file URL
- prediction_mode: whether classification or regression is performed
For a list with most parameters see DNN.py and MacauRegression.py, at the top of the file is a DEFAULT_MAP dictionary that contains all parameters and their default value, as well as comments explaining their meaning.
An exemplary JSON file for DNN and Macau is available in the code folder.
If the value of a parameter is a list, and the option to read parameters from a file is not selected, the program will generate random values based on the list at each iteration.

#### Random Forest

Generate random forest models both for PKIS and HTSFP subsets.

Example usage:

```bash
$ python -m missing_data_multitask_methods.RandomForest examples/RF_example.json
```

#### Deep Neural Networks

Generate DNNs using Tensorflow and was used both for PKIS and HTSFP subsets.

Example usage:

```bash
$ python -m missing_data_multitask_methods.DNN examples/DNN_examples.args
```

#### Macau Regression

- MacauRegression.py: generates Macau models and provides regression results (used on PKIS data)

#### Macau Classification

- MacauClassification.py: generates Macau simulating a classification procedure (used on HTSFP subsets)

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
