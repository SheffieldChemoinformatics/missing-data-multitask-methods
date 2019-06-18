import gzip
import sys
import math
import inspect

import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import pearsonr

def split_data(data, seed=1234, split=0.75):
    """ Generate a training and test split
    """

    descriptors, labels = data
    num_compounds=descriptors.shape[0]
    num_train=int(num_compounds*split)
    prng = np.random.RandomState(seed)
    idx = np.arange(num_compounds)
    prng.shuffle(idx)
    train_idx=idx[:num_train]
    test_idx=idx[num_train:]

    
    train_descriptors=descriptors[train_idx,:]
    train_labels=labels[train_idx,:]
    test_descriptors=descriptors[test_idx,:]
    test_labels=labels[test_idx,:]

    train_data=(train_descriptors, train_labels)
    test_data = (test_descriptors, test_labels)
    return (train_data, test_data)
    
def impute_missing_values(data, imputation_method='mean', similarity_matrix=None, impute_seed=1234):
    """ Given a sparse data matrix, fill empty values throught three possible methods
    """
    descriptors, labels = data
    labels = pd.DataFrame(labels)
    if imputation_method=='mean':
        labels=labels.fillna(labels.mean())
    if imputation_method=='simmean':

        denom=np.repeat(np.sum(similarity_matrix, axis=1), labels.shape[1]).reshape(labels.shape)
        simmean=pd.DataFrame(np.dot(similarity_matrix, np.nan_to_num(labels) )/denom)
        labels=labels.fillna(simmean)
    if imputation_method=='normal':
        prng = np.random.RandomState(impute_seed)
        mean_list=np.nanmean(labels, axis=0)
        std_list=np.nanstd(labels, axis=0)
        new_labels=pd.DataFrame(prng.normal(mean_list, std_list, size=labels.shape))
        labels=labels.fillna(new_labels)
    return (descriptors, labels.values)  

def remove_activity_labels_random_cells(data, percent_to_delete, remove_seed=1234):    
    """ Removes data by setting individual cells in the data matrix as nan
    """
    descriptors, labels = data
    
    prng = np.random.RandomState(remove_seed)
    to_delete=prng.binomial(1, 1-percent_to_delete, labels.shape)
    new_labels=labels.copy()
    new_labels[to_delete==0]=np.nan
    
    new_data = (descriptors, new_labels)
    
    return new_data, to_delete

def remove_activity_labels_random_compounds(data, percent_to_delete, remove_seed=1234):
    """ Removes data by setting whole rows in the data matrix as nan
    """
    descriptors, labels = data
    
    
    prng = np.random.RandomState(remove_seed)
    num_compounds=labels.shape[0]
    num_to_delete=int(round(num_compounds*percent_to_delete))
    to_delete=prng.choice(num_compounds, size=num_to_delete, replace=False)
    new_labels=labels.copy()
    new_labels[to_delete]=np.nan
    new_data = (descriptors, new_labels)

    
    return new_data, to_delete
    
def remove_activity_labels_random_assays(data, percent_to_delete, remove_seed=1234):
    """ Removes data by setting whole rows in the data matrix as nan
    """
    descriptors, labels = data
    
    
    prng = np.random.RandomState(remove_seed)
    num_assays=labels.shape[1]
    num_to_delete=int(round(num_assays*percent_to_delete))
    to_delete=prng.choice(num_assays, size=num_to_delete, replace=False)
    new_labels=labels.copy()
    new_labels[:,to_delete]=np.nan
    new_data = (descriptors, new_labels)

    
    return new_data, to_delete
    
    
def remove_activity_labels(data, percent_to_delete, remove_seed=1234, remove_type='cells'):
    """ Removes a certain percentage of data based on the removal model chosen
    """
    
    if remove_type=='cells':
        return remove_activity_labels_random_cells(data, percent_to_delete, remove_seed)
    elif remove_type=='compounds':
        return remove_activity_labels_random_compounds(data, percent_to_delete, remove_seed)
    elif remove_type=='assays':
        return remove_activity_labels_random_assays(data, percent_to_delete, remove_seed)
    else:
        print('Remove type must be one of: cells, compounds, assays. Exiting.')
        sys.exit(1)
    
 
    
def shuffle_data(data, seed=1234):
    """ Shuffles the rows in the descriptors 
    """
    descriptors, labels = data
    prng=np.random.RandomState(seed)
    prng.shuffle(descriptors)
    return (descriptors, labels)

def read_train_test_data(infil, sep=',', zipped=True):
    """ Read the file and return the descriptors and activity labels
    """
    desc=[]
    labels=[]
    j=1
    if zipped:
        fin=gzip.open(infil, 'rt')
    else:
        fin=open(infil)
    header = next(fin).strip().split(sep)
    for i,name in enumerate(header):
        if name[:2]=="D_":
            break
    for line in fin:
        
        dat=line.strip().split(sep)
        desc.append(dat[i:])
        labels.append(dat[1:i])
        j+=1
    fin.close()
    
    descriptors=np.array(desc, dtype=np.float32)
    labels=np.array(labels, dtype=np.float32 )
    return (descriptors, labels)
    
def read_dataset_names(infil, sep=','):
    """ Read the header of the file and return the names of the activity labels
    """
    with gzip.open(infil, 'rt') as fin:
        header = next(fin).strip().split(sep)
        for i,name in enumerate(header):
            if name[:2]=="D_":
                break
    
    datasets=header[1:i]
    return datasets
    
def obtain_results(y_true, y_predicted, prediction_mode):
    """ Given two matrices of true and predicted values return a set of statistics
    """
    
    results={}
    idx_func = lambda i: np.logical_not(np.isnan(y_true[:,i]))
    if prediction_mode=='regression':
        if np.isnan(y_predicted).all():
            print('Nan predicted values.')
            results['r2']=['NaN'  for i in range(y_true.shape[1]) ]
            results['R2']=['NaN' for i in range(y_true.shape[1])] 
            results['MAE']=['NaN' for i in range(y_true.shape[1])] 
            results['RMSD']=['NaN' for i in range(y_true.shape[1])] 
        else:
            
            results['r2']=[pearsonr(y_predicted[idx_func(i),i], y_true[idx_func(i),i])[0]**2  for i in range(y_predicted.shape[1]) ]
            results['R2']=[metrics.r2_score(y_true[idx_func(i),i], y_predicted[idx_func(i),i]) for i in range(y_predicted.shape[1])] 
            results['MAE']=[metrics.mean_absolute_error(y_true[idx_func(i),i], y_predicted[idx_func(i),i]) for i in range(y_predicted.shape[1])] 
            results['RMSD']=[math.sqrt(metrics.mean_squared_error(y_true[idx_func(i),i], y_predicted[idx_func(i),i])) for i in range(y_predicted.shape[1])] 
        

    elif prediction_mode=='classification':
        y_true[np.where(y_true<=0.5)]=0
        y_true[np.where(y_true>0.5)]=1
        
        if np.isnan(y_predicted).any():
            print('Nan predicted values. Changing them to 0')
            y_predicted = np.nan_to_num(y_predicted)
            
        
        results['auc']=[metrics.roc_auc_score(y_true[idx_func(i),i], y_predicted[idx_func(i),i]) if np.nansum(y_true[:,i])>0 else 'NaN' for i in range(y_predicted.shape[1])] 
        
        y_predicted[np.where(y_predicted<=0.5)]=0
        y_predicted[np.where(y_predicted>0.5)]=1
        
        
        results['precision']=[metrics.precision_score(y_true[idx_func(i),i], y_predicted[idx_func(i),i]) for i in range(y_predicted.shape[1])] 
        results['recall']=[metrics.recall_score(y_true[idx_func(i),i], y_predicted[idx_func(i),i]) for i in range(y_predicted.shape[1])]                                                  
        results['fscore']=[metrics.f1_score(y_true[idx_func(i),i], y_predicted[idx_func(i),i]) for i in range(y_predicted.shape[1])] 
        results['mcc']=[metrics.matthews_corrcoef(y_true[idx_func(i),i], y_predicted[idx_func(i),i]) for i in range(y_predicted.shape[1])] 
        
    elif prediction_mode=='classification_multiclass':
        results['accuracy']=metrics.accuracy_score(y_true, y_pred)
        results['errors']=y_true.shape[0]-metrics.accuracy_score(y_true, y_pred, normalize=False)
        

    
        
    return results
    

        
def create_parameters_from_file(param_file, args, iter_num):
    """ Read a result file and obtain a specific set of parameters based on it.
        The line number to use is given by the iter_num parameter.
    """
    params={}
    dat = pd.read_csv(param_file, sep='\t')
     
    for key in args:
        if key in list(dat):
            val=dat[key].values.tolist()[iter_num]
            params[key]=val
            
        else:
            params[key]=args[key]
    

    return params        
    
def create_random_parameter(args, prng=None):
    """ Given the argument dictionary, return a dictionary with the same keys
        but where the values are random if there were two or more elements in their
        arguments dictionary (using either the uniform distribution between two
        numbers or random choice among a list)
    """
    params={}
    if prng is None:
        
        
        prng = np.random.RandomState()
    
    for key in sorted(args.keys()):
        if type(args[key]) in [list, tuple]:
            if len(args[key])==2 and type(args[key][0])==int  and type(args[key][1])==int:
                tmp=int(prng.uniform(args[key][0], args[key][1], 1)[0])
                params[key]=tmp
            elif len(args[key])==2 and np.isreal(args[key][0])  and np.isreal(args[key][1]) and not type(args[key][0])==bool:
                tmp=prng.uniform(args[key][0], args[key][1], 1)[0]
                params[key]=tmp
            else:
                tmp=prng.choice(args[key])
                params[key]=tmp
        else:
            params[key]=args[key]
            
    return params
        
    
def transform(train_data, test_data, args):
    """ Transform descriptor data
    """
    train_descriptors, train_labels=train_data
    test_descriptors, test_labels=test_data

    data_transform = args.get('data_transform')

    if data_transform not in [None, 'log', 'binary', 'none', 'None']:
        print('Data transformation not understood: {0}. Exiting'.format(args['data_transform']))
        sys.exit(1)

    if data_transform is None:
        pass
    elif data_transform.lower() == 'log':
        train_descriptors=np.log(train_descriptors+1)
        test_descriptors=np.log(test_descriptors+1)
    elif data_transform.lower() == 'binary':
        train_descriptors=np.where(train_descriptors>=1, 1, 0)
        test_descriptors=np.where(test_descriptors>=1, 1, 0)
    
    train_data = (train_descriptors, train_labels)
    test_data = (test_descriptors, test_labels)
    return (train_data, test_data)
    
def generate_kwargs_dictionary(func, params):
    """ Given a function with keyword parameters and a dictionary return a subset of the 
        dictionary that only contains keys that match keyword arguments
    """
    kwargs={}
    for key in params:
        if key in inspect.getfullargspec(func)[0]:
            kwargs[key]=params[key]
            
    return kwargs


def calculate_weighting(train_data):
    """ Generate a vector of target weights based on their inverse frequency and a vector of compound weights based on the 
        targets weights to which they are active
    """
    indices=[] 
    for i in range(train_data[1].shape[1]):
        indices.append(list())

    mol_dict=[]
    for i,row in enumerate(train_data[1]):
        tmp=[]

        for j, boolean in enumerate(np.isfinite(train_data[1][i,])):
            if boolean:
                indices[j].append(i)
                tmp.append(j)


        mol_dict.append(tmp)

    cum_sum=sum(len(indices[j]) for j in range(len(indices)))
    target_probs=[len(indices[j])/cum_sum for j in range(len(indices))]
    inverse_probs=[max(target_probs)/t for t in target_probs]
    norm_inverse=[inv/sum(inverse_probs) for inv in inverse_probs]

    comp_prob=[]
    for i in range(len(mol_dict)):
        probs=[norm_inverse[j] for j in mol_dict[i]]
        if len(probs)!=0:
            comp_prob.append(np.max(probs))
        else:
            comp_prob.append(0)
    sum_comp_prob=sum(comp_prob)
    norm_comp_prob=[c/sum_comp_prob for c in comp_prob]
    
    return (norm_inverse, norm_comp_prob)            



