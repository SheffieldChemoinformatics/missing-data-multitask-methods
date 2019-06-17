import os
import sys
import time
import json

import numpy as np
import pandas as pd
import scipy
import macau

from .helper_functions import *

##SET DEFAULTS

DEFAULT_MAP={
'lambda_beta':5.0,                      #Regularization parameter related to the side information
'num_latent': 32,                       #Number of latent dimensions
'precision': "adaptive",                #Precision of observations of the matrix. See more info at Macau's documentation
'burnin': 400,                          #Number of iterations to drop during training
'nsamples': 1600,                       #Number of iterations to perform
'univariate':False,                     #If true, use a faster sampler
'tol':1e-6,                             #Error tolerance during training
'sn_max':10.0,                          #Maximum signal-to-noise ratio to use in adaptive precision mode
'model_suffix':1,                       #Integer to keep track of model iteration number
'data_transform':None,                  #Transform input descriptors: 'log', 'binary', and 'none'
'shuffle_labels':False,                 #whether to shuffle descriptor data with respect to activity labels 
'shuffle_seed':1234,                    #seed to be used during shuffling
'read_parameters':False,                #if true, hyperparameter will be taken from param_file rather than generated randomly
'predictions_file':None,                #if present, save predicted values 
'remove_labels': False,                 #wether to remove activity labels from training data
'percent_to_delete': 0,                 #percentage of data labels to delete 
'remove_seed': 1234,                    #seed that controls which activity labels are removed 
'remove_type': 'cells',                 #wether individual cells ('cells'), whole molecules ('compounds') or whole assays ('assays') are removed
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
        
        descriptors, labels = data
        
        split=0.75
        seed=int(testing_file)
        num_compounds=descriptors.shape[0]
        num_train=int(num_compounds*split)
        prng = np.random.RandomState(seed)
        idx = np.arange(num_compounds)
        prng.shuffle(idx)
        train_idx=idx[:num_train]
        test_idx=idx[num_train:]
        
        
        
        train_labels=pd.DataFrame(labels.copy())
        train_labels=train_labels+1
        train_labels.iloc[test_idx,:]=np.nan
        
        
        test_labels=pd.DataFrame(labels.copy())
        test_labels=test_labels+1
        test_labels.iloc[train_idx,:]=np.nan
        
        
        datasets=read_dataset_names(training_file)
        
    if args['shuffle_labels']:
        train_data=shuffle_data(train_data, seed=args['shuffle_seed'])
        
    
    
    if args['prediction_mode']=='regression':
        result_names=['r2', 'R2', 'MAE', 'RMSD']
    elif args['prediction_mode']=='classification':
        result_names=['precision', 'recall', 'fscore', 'mcc', 'auc']
    elif args['prediction_mode']=='classification_multiclass':
        result_names=['accuracy', 'errors']
        
    return (args, train_labels, test_labels, descriptors, result_names, output_file, datasets, train_idx, test_idx)
   
def transform_descriptors(descriptors, args):
    """ Transform descriptor data
    """
    if args['data_transform'] not in [None, 'log', 'binary', 'none', 'None']:
        print('Data transformation not understood: {0}. Exiting'.format(args['data_transform']))
        sys.exit(1)
    if args['data_transform'] is None:
        pass
    elif args['data_transform'].lower()=='log':
        descriptors=np.log(descriptors+1)
        
    elif args['data_transform'].lower()=='binary':
        descriptors=np.where(descriptors>=1, 1, 0)
        
        
    return descriptors
    
def remove_activity_labels_random_cells(labels, percent_to_delete, remove_seed=1234):    
    """ Removes data by setting individual cells in the data matrix as nan
    """
    
    prng = np.random.RandomState(remove_seed)
    idx_x, idx_y = np.where(np.isfinite(labels))
    num_real_labels=len(idx_x)
    to_delete=prng.binomial(1, 1-percent_to_delete, num_real_labels)

    
    new_labels=labels.copy().values
    new_labels[idx_x[to_delete==0],idx_y[to_delete==0]]=np.nan
    

    new_labels=pd.DataFrame(new_labels)

    
    return new_labels, to_delete

def remove_activity_labels_random_compounds(labels, percent_to_delete, remove_seed=1234):
    """ Removes data by setting whole rows in the data matrix as nan
    """
    prng = np.random.RandomState(remove_seed)
    
    real_cpds=labels.dropna(axis=0, how='all').index.values
    num_real_cpds=labels.dropna(axis=0, how='all').shape[0]
    num_real_todelete=int(percent_to_delete*num_real_cpds)

    
    to_delete=prng.choice(real_cpds, num_real_todelete, replace=False)

    
    new_labels=labels.copy().values
    new_labels[to_delete]=np.nan
    

    new_labels=pd.DataFrame(new_labels)

    
    return new_labels, to_delete
    
def remove_activity_labels_random_assays(labels, percent_to_delete, remove_seed=1234):

    prng = np.random.RandomState(remove_seed)
    
    num_assays=labels.shape[1]
    num_assays_todelete=int(percent_to_delete*num_assays)
    
    to_delete=prng.choice(num_assays, num_assays_todelete, replace=False)

    
    new_labels=labels.copy().values
    new_labels[:,to_delete]=np.nan

    new_labels=pd.DataFrame(new_labels)

    
    return new_labels, to_delete
    
def remove_labels(labels, percent_to_delete, remove_type='cells', remove_seed=1234):
    if remove_type=='cells':
        return remove_activity_labels_random_cells(labels, percent_to_delete, remove_seed)
    elif remove_type=='compounds':
        return remove_activity_labels_random_compounds(labels, percent_to_delete, remove_seed)
    elif remove_type=='assays':
        return remove_activity_labels_random_assays(labels, percent_to_delete, remove_seed)
    else:
        print('Remove type must be one of: cells, compounds, assays. Exiting.')
        sys.exit(1)
    
def run_model(train_labels, test_labels, descriptors, train_idx, test_idx, params):
    """ Runs the Macau calculation and returns the statistical results
    """    
    macau_param_list=['lambda_beta', 'num_latent', 'precision', 'burnin','nsamples','univariate','tol','sn_max']
    kwargs={key:params[key] for key in macau_param_list}
    
    # print(train_labels)
    
    descriptors=scipy.sparse.csr_matrix(descriptors)
    train_labels_sparse=scipy.sparse.csr_matrix(train_labels)

    test_labels_sparse=scipy.sparse.csr_matrix(test_labels)

    
    result = macau.macau(Y = train_labels_sparse, Ytest = test_labels_sparse, side = [descriptors, None], verbose=False, **kwargs)

                     
    y_predicted=pd.pivot_table(result.prediction, values=['y_pred'], columns=['col'], index=['row']).values-1
    y_true=pd.pivot_table(result.prediction, values=['y'], columns=['col'], index=['row']).values-1
    

    
    # y_true_2=test_labels.copy()
    # y_true_2_idxs=y_true_2.dropna(axis=0, how='all').index.values   
    # y_true_2=y_true_2.dropna(axis=0, how='all').values

    
    # y_predicted_2=np.zeros(test_labels.shape)
    # y_predicted_2[:,:]=np.nan
    
    # y_predicted_2[result.prediction['row'].values, result.prediction['col'].values]=result.prediction['y_pred'].values
    # y_predicted_2=pd.DataFrame(y_predicted_2)
    # y_predicted_2=y_predicted_2[y_predicted_2.index.isin(y_true_2_idxs)].values
    

    if params['predictions_file'] is not None:
        np.savez_compressed("{0}_{1}".format(params['predictions_file'], params['model_suffix']), y_predicted)
    
    
    results=obtain_results(y_true, y_predicted, params['prediction_mode'])
    return results
    
if __name__ == "__main__":
    #Parameter setting
    arg_file=sys.argv[1]
    args, original_train_labels, original_test_labels, original_descriptors, result_names, output_file, datasets, train_idx, test_idx = set_up(arg_file)
    param_prng=np.random.RandomState(args['seed'])
    
    num_iterations=args['num_iterations']
    del args['num_iterations']
    

    if args['allow_restart'] and os.path.exists(output_file):
        num_lines = sum(1 for line in open(output_file) if line.rstrip())-1
        start_line=num_lines
        fout=open(output_file, 'a', 1)
        for j in range(0, start_line):
            create_random_parameter(args, param_prng)
        
    else:
        fout=open(output_file, 'w', 1)
        start_line=0
        out=sorted([arg for arg in args.keys() if '_file' not in arg])
        out.append('Time')
        out.extend([dataset+'_'+name+'_'+tt for dataset in datasets for name in result_names for tt in ['Train', 'Test']])
        fout.write('\t'.join(out)+'\n')
        
    del args['allow_restart']
        
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
            descriptors = transform_descriptors(original_descriptors, params)
            train_labels=original_train_labels.copy()
            test_labels=original_test_labels.copy()
            

            if params['remove_labels']:
                train_labels, to_delete=remove_labels(train_labels, params['percent_to_delete'], params['remove_type'], params['remove_seed'])
                remove_labels_dict[str(j)]=to_delete

                
            if params['impute_labels']:

                kwargs=generate_kwargs_dictionary(impute_missing_values, params)
                (_, train_labels)=impute_missing_values([None,train_labels], similarity_matrix=similarity_matrix, **kwargs)
           
            training_results = run_model(train_labels, train_labels, descriptors, train_idx, train_idx, params)
            test_results = run_model(train_labels, test_labels, descriptors, train_idx, test_idx, params)
            elapsed=time.time()-start
            out=[params[param] for param in sorted(args.keys()) if '_file' not in param]
            out.append(elapsed)

            
            try:
                result_values=[]
                for i in range(len(datasets)):
                    for name in result_names:
                        result_values.extend([training_results[name][i], test_results[name][i]])

            except IndexError:
                #This is done when some assays are completely empty and we get no predictions from them
                result_values=[]
                for i in range(len(datasets)):
                    for name in result_names:

                        result_values.extend(['NaN', 'NaN'])
            except:
                raise
            out.extend(result_values)
            fout.write('\t'.join(map(str, out))+'\n')
            
            if args['remove_labels_file'] is not None:
                np.savez_compressed(args['remove_labels_file'], **remove_labels_dict)
            
            