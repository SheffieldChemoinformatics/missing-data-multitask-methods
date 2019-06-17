import gzip
import os
import sys
import time
import itertools
import json
import math
from collections import defaultdict

import inspect
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.externals import joblib
from scipy.stats import pearsonr

from .helper_functions import *
  
##SET DEFAULTS

DEFAULT_MAP={
'num_iterations':1,
'n_estimators':400,                     #number of trees in the random forest
'criterion':'gini',                     #criterion to use when splitting the sample
'max_depth':None,                       #maximum depth of the tree
'min_samples_split':2,                  #number of samples required to split a node
'min_samples_leaf':1,                   #number of samples required to generate a leaf node
'min_weight_fraction_leaf':0.0,         #minimum weighted fraction required to generate a leaf node
'max_features':'auto',                  #number of features to consider when splitting a node
'min_impurity_split':1e-07,             #impurity threshold to split a node
'bootstrap':True,                       #wether samples taken to construct a tree use bootstrap
'oob_score':False,                      #wether to compute out-of-bag errors
'n_jobs':1,                             #number of CPU cores to use during training
'random_state':1234,                    #seed for the random forest method
'class_weight':None,                    #weights given to different classes
'sample_weight':None,                   #weights given to individual data points
'seed':1234,                            #general seed 
'model_file':None,                      #If present models are saved there after training 
'model_suffix':1,                       #Integer to keep track of model iteration number  
'restore_forest':False,                 #whether to check for a model file to use to load a model rather than train it
'shuffle_labels':False,                 #whether to shuffle descriptor data with respect to activity labels 
'shuffle_seed':1234,                    #seed to be used during shuffling
'read_parameters':False,                #if true, hyperparameter will be taken from param_file rather than generated randomly
'predictions_file':None,                #if present, save predicted values 
'remove_labels': False,                 #wether to remove activity labels from training data
'remove_type': 'cells',                 #wether individual cells ('cells'), whole molecules ('compounds') or whole assays ('assays') are removed
'percent_to_delete': 0,                 #percentage of data labels to delete 
'remove_seed': 1234,                    #seed that controls which activity labels are removed 
'remove_labels_file': None,             #if present, which compounds were removed will be saved 
'impute_labels': False,                 #if true, empty values will be filled with imputed values 
'imputation_method':'mean',             #method to use to fill empty values: 'mean', 'simmean', and 'normal'
'impute_seed':1234,                     #seed to be used if using normal method during imputation
'allow_restart':False                   #wether to start from scratch or continue if the output file exists 
} 

def check_arguments(arg_file):
    """ Load an argument file and add missing parameters based on default values
    """
    with open(arg_file) as fin:
        args=json.load(fin)
        
    for param in DEFAULT_MAP.keys():
        if param not in args:
            args[param]=DEFAULT_MAP[param]


    return args

# NUM_ITERATIONS=1
# N_ESTIMATORS=400
# CRITERION='gini'
# MAX_DEPTH=None
# MIN_SAMPLES_SPLIT=2
# MIN_SAMPLES_LEAF=1
# MIN_WEIGHT_FRACTION_LEAF=0.0
# MAX_FEATURES='auto'
# MAX_LEAF_NODES=None
# MIN_IMPURITY_SPLIT=1e-07
# BOOTSTRAP=True
# OOB_SCORE=False
# N_JOBS=1
# RANDOM_STATE=1234
# CLASS_WEIGHT=None
# SAMPLE_WEIGHT=None
# RESTORE_FOREST=False
# MODEL_FILE=None
# DATA_TRANSFORM=None

   
# def check_arguments(arg_file):

    # with open(arg_file) as fin:
        # args=json.load(fin)
    # if 'num_iterations' not in args:
        # args['num_iterations']=NUM_ITERATIONS
    # if 'n_estimators' not in args:
        # args['n_estimators']=N_ESTIMATORS
    # if 'criterion' not in args:
        # args['criterion']=CRITERION
    # if 'max_depth' not in args:
        # args['max_depth']=MAX_DEPTH
    # if 'min_samples_split' not in args:
        # args['min_samples_split']=MIN_SAMPLES_SPLIT
    # if 'min_samples_leaf' not in args:
        # args['min_samples_leaf']=MIN_SAMPLES_LEAF
    # if 'min_weight_fraction_leaf' not in args:
        # args['min_weight_fraction_leaf']=MIN_WEIGHT_FRACTION_LEAF
    # if 'max_features' not in args:
        # args['max_features']=MAX_FEATURES
    # if 'max_leaf_nodes' not in args:
        # args['max_leaf_nodes']=MAX_LEAF_NODES
    # if 'min_impurity_split' not in args:
        # args['min_impurity_split']=MIN_IMPURITY_SPLIT
    # if 'bootstrap' not in args:
        # args['bootstrap']=BOOTSTRAP
    # if 'oob_score' not in args:
        # args['oob_score']=OOB_SCORE
    # if 'n_jobs' not in args:
        # args['n_jobs']=N_JOBS
    # if 'random_state' not in args:
        # args['random_state']=RANDOM_STATE
    # if 'class_weight' not in args:
        # args['class_weight']=CLASS_WEIGHT
    # if 'sample_weight' not in args:
        # args['sample_weight']=SAMPLE_WEIGHT
    # if 'restore_forest' not in args:
        # args['restore_forest']=RESTORE_FOREST
    # if 'model_file' not in args:
        # args['model_file']=MODEL_FILE
    # if 'data_transform' not in args:
        # args['data_transform']=DATA_TRANSFORM
    # return args
    
def train_forest(train_data, params, model_suffix):

    train_descriptors, train_labels=train_data
    
    if params['prediction_mode']=='classification':
        kwargs=generate_kwargs_dictionary(RandomForestClassifier, params)
        rf = RandomForestClassifier(**kwargs)
        
        rf.fit(train_descriptors, train_labels, sample_weight=params['sample_weight'])
    elif params['prediction_mode']=='regression':
        kwargs=generate_kwargs_dictionary(RandomForestRegressor, params)
        rf = RandomForestRegressor(**kwargs)
        rf.fit(train_descriptors, train_labels, sample_weight=params['sample_weight'])
            
    
    if params['model_file'] is not None:
        joblib.dump(rf, params['model_file']+'_{0}.gz'.format(model_suffix)) 
    return rf
    
def restore_forest(model_file, model_suffix):
    
    rf = joblib.load(model_file+'_{0}.gz'.format(model_suffix))
    return rf
    
def predict_forest(data, rf, prediction_mode):
    
    descriptors, labels=data
    if prediction_mode=='classification':
        predicted = rf.predict_proba(descriptors)
        # print(len(predicted))
        # for values in predicted:
            # print(len(values))
        predicted = np.array(predicted).T[1,:,:]
        # print(predicted[0,0,:], predicted[1,0,:])
    elif prediction_mode=='regression':
        predicted = rf.predict(descriptors)
    
    results=obtain_results(labels, predicted, prediction_mode)
    return results
        
def set_up(arg_file, verbose=False):
    """ Based on a argument file, read the input data and modify it if necessary
    """
    arg_file=sys.argv[1]
    args=check_arguments(arg_file)
    training_file=args['training_file']
    testing_file=args['testing_file']
    output_file=args['output_file']


    if os.path.exists(testing_file):
        if verbose:
            print('Reading training data')
        start=time.time()
        infil=os.path.join(training_file)
        train_data=read_train_test_data(infil)

        if verbose:
            print('Reading training data done in {0} s'.format(time.time()-start))

            print('Reading test data')
        start=time.time()
        infil=os.path.join(testing_file)
        test_data=read_train_test_data(infil)
        if verbose:
            print('Reading test data done in {0} s'.format(time.time()-start))
        
        datasets=read_dataset_names(infil)
    else:
        if verbose:
            print('Reading data')
        start=time.time()
        data=read_train_test_data(training_file)
        if verbose:
            print('Reading data done in {0} s'.format(time.time()-start))
        
        train_data, test_data = split_data(data, seed=int(testing_file))
        
        datasets=read_dataset_names(training_file)
        
    if args['shuffle_labels']:
        train_data=shuffle_data(train_data, seed=args['shuffle_seed'])
        
    
    
    if args['prediction_mode']=='regression':
        result_names=['r2', 'R2', 'MAE', 'RMSD']
    elif args['prediction_mode']=='classification':
        result_names=['precision', 'recall', 'fscore', 'mcc', 'auc']
    elif args['prediction_mode']=='classification_multiclass':
        result_names=['accuracy', 'errors']
        
    return (args, train_data, test_data, result_names, output_file, datasets)


    


if __name__ == "__main__":
    #Parameter setting
    arg_file=sys.argv[1]
    args, original_train_data, original_test_data, result_names, output_file, datasets = set_up(arg_file)
    param_prng=np.random.RandomState(args['seed'])
    
    num_iterations=args['num_iterations']
    del args['num_iterations']
    
    # print('Starting the DNN prediction')
    if args['allow_restart'] and os.path.exists(output_file):
        num_lines = sum(1 for line in open(output_file) if line.rstrip())-1
        start_line=num_lines
        fout=open(output_file, 'a', 1)
        for j in range(0, start_line):
            create_random_parameter(args, param_prng)
        
    else:
        fout=open(output_file, 'w', 1)
        start_line=0
        out=sorted([arg for arg in sorted(args.keys()) if '_file' not in arg])
        out.append('Time')
        out.extend([dataset+'_'+name+'_'+tt for dataset in datasets for name in result_names for tt in ['Train', 'Test']])
        fout.write('\t'.join(out)+'\n')
        
            
    if args['remove_labels']:
        remove_labels_dict={}
        
    if args['impute_labels']:
        similarity_matrix=1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(original_train_data[0], 'jaccard'))
        
    for j in range(start_line, num_iterations):
        
        args['model_suffix']=j
    
        if args['read_parameters']:
            params = create_parameters_from_file(args['arg_file'], args, j)
        else:
            params=create_random_parameter(args, param_prng)
        

        start=time.time()
        train_data, test_data = transform(original_train_data, original_test_data, params)
        if (np.isnan(train_data[1]).any()):
            print(params)
        # print(np.isnan(train_data[1]).any(), train_data[1].shape)
        if params['remove_labels']:
            train_data, to_delete=remove_activity_labels(train_data, params['percent_to_delete'], params['remove_seed'], params['remove_type'])
            to_keep=[i for i in range(train_data[0].shape[0]) if i not in to_delete] 
            train_data = (train_data[0][to_keep], train_data[1][to_keep])
            remove_labels_dict[str(j)]=to_delete
            
        # print(params['percent_to_delete'], np.isnan(train_data[1]).any(), train_data[1].shape)
            
        if params['impute_labels']:
            kwargs=generate_kwargs_dictionary(impute_missing_values, params)
            train_data=impute_missing_values(train_data, similarity_matrix=similarity_matrix, **kwargs)
                   
        if args['restore_forest']:
            rf = restore_forest(params['model_file'], j)
        else:
            rf = train_forest(train_data, params, j)
            
        # print(rf.oob_score_)
        training_results = predict_forest(train_data, rf, params['prediction_mode'])
        test_results = predict_forest(test_data, rf, params['prediction_mode'])
        elapsed=time.time()-start
        out=[params[param] for param in sorted(args.keys()) if '_file' not in param]
        out.append(elapsed)
        for i in range(len(datasets)):
            for name in result_names:
                out.extend([training_results[name][i], test_results[name][i]])
        fout.write('\t'.join(map(str, out))+'\n')
        
    fout.close()
    if args['remove_labels_file'] is not None:
        np.savez_compressed(args['remove_labels_file'], **remove_labels_dict)            
            



