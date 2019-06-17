import os
import sys
import time
import json
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

from .helper_functions import *
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #This should remove info and warnings
        
##SET DEFAULTS

DEFAULT_MAP={
'num_iterations':1,
'num_neurons':1000,                     #number of neurons in each hidden layer
'num_layers':4,                         #number of hidden layers 
'dropout':0.1,                          #percentage of neurons in hidden layer that are removed in each training step
'batch_size':50,                        #number of training instances to use in each training step 
'num_steps':5000,                       #number of training steps to be performed 
'compound_weighting':True,              #whether to bias the selection of compounds per mini-batch based on target frequency
'target_weighting':True,                #wether to weight cost function based on target frequency 
'multitask_strategy':'individual',      #wether to use one optimizer for all outputs ('all') or one per output ('individual')
'activation':'relu',                    #activation function: 'relu' and 'sigmoid' are allowed values
'concurrent_perm':1,                    #one overengineered parameter that ended up not affecting anything
'data_transform':None,                  #Transform input descriptors: 'log', 'binary', and 'none'
'learning_rate':0.05,                   #Learning rate of DNN optimizer function 
'optimizer':'Adagrad',                  #Optimizer function to use: see tensorflow documentation for options
'seed':1234,                            #General seed given to tensorflow
'model_file':None,                      #If present DNN models are saved there after training 
'model_suffix':1,                       #Integer to keep track of model iteration number                       
'restore_network':False,                #whther to check for a model file to use to load a model rather than train it
'shuffle_labels':False,                 #whether to shuffle descriptor data with respect to activity labels 
'shuffle_seed':1234,                    #seed to be used during shuffling
'read_parameters':False,                #if true, hyperparameter will be taken from param_file rather than generated randomly
'predictions_file':None,                #if present, save predicted values 
'remove_labels': False,                 #wether to remove activity labels from training data
'percent_to_delete': 0,                 #percentage of data labels to delete 
'remove_seed': 1234,                    #seed that controls which activity labels are removed 
'remove_labels_file': None,             #if present, which compounds were removed will be saved 
'remove_type': 'cells',                 #wether individual cells ('cells'), whole molecules ('compounds') or whole assays ('assays') are removed
'impute_labels': False,                 #if true, empty values will be filled with imputed values 
'imputation_method':'mean',             #method to use to fill empty values: 'mean', 'simmean', and 'normal'
'impute_seed':1234,                     #seed to be used if using normal method during imputation 
'allow_restart':False                   #wether to start from scratch or continue if the output file exists
} 


def variable_summaries(var):
  """ Attach a lot of summaries to a Tensor (for TensorBoard visualization).
      Taken from https://www.tensorflow.org/get_started/summaries_and_tensorboard
  """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    

def generate_network(train_data, test_data, prediction_mode, num_neurons=DEFAULT_MAP['num_neurons'], num_layers=DEFAULT_MAP['num_layers'], dropout=DEFAULT_MAP['dropout'], 
                     multitask_strategy=DEFAULT_MAP['multitask_strategy'], activation=DEFAULT_MAP['activation'], learning_rate=DEFAULT_MAP['learning_rate'], optimizer=DEFAULT_MAP['optimizer'], seed=DEFAULT_MAP['seed']):
    """ Generate a neural network based on parameters and set up the optimization format
    """
    

    learning_rate=float(learning_rate)
    
    train_descriptors, train_labels=train_data
    test_descriptors, test_labels=test_data

    num_compounds=train_descriptors.shape[0]
    num_tasks=train_labels.shape[1] if len(train_labels.shape)>1 else 1
    
    
    if num_tasks>1 and train_labels.shape[1]!=test_labels.shape[1]:
        print('Training and test data have different number of tasks. Exiting')
        sys.exit(1)
        

    
    #Initialize the computation graph
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    
    
    neuron_per_layer=[num_neurons]*num_layers
    
    #Transform inidividual parameter to per layer parameters
    if activation=="relu":
        activation=[tf.nn.relu]*len(neuron_per_layer)
    elif activation=="sigmoid":
        activation=[tf.nn.sigmoid]*len(neuron_per_layer)
    else:
        print("Didn't recognize activation function {0}. Using ReLu instead.")
        activation=[tf.nn.relu]*len(neuron_per_layer)
    
    if type(activation) in [list,tuple]:
        if len(activation)!=len(neuron_per_layer):
            print("Not enough elements in activation list for each layer. Exiting")
            sys.exit(1)
    
    if type(dropout) in [list,tuple]:
        dropout_per_layer=dropout
        if len(dropout)!=len(neuron_per_layer):
            print("Not enough elements in dropout list for each layer. Exiting")
            sys.exit(1)
    else:
        dropout_per_layer=[dropout]*len(neuron_per_layer)                                       
    
    
    #Generate the network
    features=tf.placeholder(tf.float32, shape=(None, train_descriptors.shape[1]), name='features')
    hidden_layer = tf.contrib.layers.fully_connected(features, neuron_per_layer[0], activation_fn=activation[0])
    dropout_hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=(1.0 - dropout_per_layer[0]))
    variable_summaries(dropout_hidden_layer)
    layer_list=[dropout_hidden_layer]
    for i in range(1, len(neuron_per_layer)):
        hidden_layer = tf.contrib.layers.fully_connected(layer_list[i-1],  neuron_per_layer[i],
                                                  activation_fn=activation[i])
        dropout_hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=(1.0 - dropout_per_layer[i]))
        variable_summaries(dropout_hidden_layer)
        layer_list.append(dropout_hidden_layer)
    
    
    #Set up the output and the learning
        
    loss_fn={'regression': tf.losses.mean_squared_error,
             'classification':tf.losses.sigmoid_cross_entropy,
             'classification_multiclass': tf.losses.softmax_cross_entropy }
             
    output_trans={'regression': tf.identity,
                 'classification': tf.sigmoid,
                 'classification_multiclass': tf.nn.softmax }
    
    
    if num_tasks==1:
        targets=tf.placeholder(tf.float32, shape=(None, 1), name='targets')
        output_layer = tf.contrib.layers.fully_connected(layer_list[-1], 1, activation_fn=None)
        predictions=output_trans[prediction_mode](output_layer, name='predictions')

        loss=loss_fn[prediction_mode](targets, output_layer)
        train_op = tf.contrib.layers.optimize_loss(
              loss=loss,
              global_step=tf.contrib.framework.get_global_step(),
              learning_rate=learning_rate,
              optimizer=optimizer)
        variable_summaries(loss)
        summaries = tf.summary.merge_all()
        placeholders=[features, targets]
        return placeholders, predictions, train_op, summaries
    elif multitask_strategy=="all":
        targets=tf.placeholder(tf.float32, shape=(None, num_tasks), name='target')
        output_layer = tf.contrib.layers.fully_connected(layer_list[-1], num_tasks, activation_fn=None)
        predictions = output_trans[prediction_mode](output_layer, name='predictions')

        loss_weight=tf.placeholder(tf.float32, shape=(None, num_tasks))
        
        
        loss = loss_fn[prediction_mode](targets, output_layer, weights=loss_weight)
        train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=learning_rate,
          optimizer=optimizer)
        variable_summaries(loss)
        summaries = tf.summary.merge_all()
        placeholders=[features, targets, loss_weight]
        return placeholders, predictions, train_op, summaries
    elif multitask_strategy=="individual":
        train_op_list=[]
        predictions_list=[]
        targets=tf.placeholder(tf.float32, shape=(None, ), name='targets')
        for i in range(num_tasks):

            output_layer = tf.contrib.layers.fully_connected(layer_list[-1], 1, activation_fn=None)
            output_layer = tf.reshape(output_layer, [-1])
            predictions=output_trans[prediction_mode](output_layer, name='predictions_{0}'.format(i))
            predictions_list.append(predictions)
            
            loss=loss_fn[prediction_mode](targets, output_layer)
            train_op = tf.contrib.layers.optimize_loss(
              loss=loss,
              global_step=tf.contrib.framework.get_global_step(),
              learning_rate=learning_rate,
              optimizer=optimizer)
            train_op_list.append(train_op)
        variable_summaries(loss)
        summaries = tf.summary.merge_all()
        placeholders=[features, targets]
        return placeholders, predictions_list, train_op_list, summaries
            
def network_training(train_data, placeholders, train_op, summaries, prediction_mode, 
                       batch_size=DEFAULT_MAP['batch_size'], num_steps=DEFAULT_MAP['num_steps'], compound_weighting=DEFAULT_MAP['compound_weighting'], target_weighting=DEFAULT_MAP['target_weighting'],
                       multitask_strategy=DEFAULT_MAP['multitask_strategy'], concurrent_perm=DEFAULT_MAP['concurrent_perm'],
                       seed=DEFAULT_MAP['seed'], model_file=DEFAULT_MAP['model_file'], model_suffix=DEFAULT_MAP['model_suffix']):
    """ Given a network, start a session and perform the training
    """
    
    train_descriptors, train_labels=train_data
    

    num_compounds=train_descriptors.shape[0]
    num_tasks=train_labels.shape[1] if len(train_labels.shape)>1 else 1
    
    #Calculating target and compound probability
    if train_data[1].shape[1]>1 and (compound_weighting or target_weighting):
        norm_inverse, norm_comp_prob = calculate_weighting(train_data)
    else:
        norm_comp_prob=None
        norm_inverse=None
    
    if compound_weighting:
        compound_weighting=norm_comp_prob
    else:
        compound_weighting=[1/num_compounds]*num_compounds
    if target_weighting:
        target_weighting=norm_inverse
    else:
        target_weighting=[1/num_tasks]*num_tasks
    
    saver = tf.train.Saver(tf.trainable_variables())
    

    prng = np.random.RandomState(seed)
             
    #Start the computation
    sess=tf.Session()

    
        
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    
    num_perms=math.floor(num_steps/concurrent_perm)
    
    
    
    #Perform the training
    for i in range(num_perms):
        
        perm = prng.choice(train_descriptors.shape[0], batch_size*concurrent_perm,p=compound_weighting, replace=True)
        permed_descriptors=train_descriptors[perm]
        permed_labels=train_labels[perm][:,]
        
               
        for j in range(0, permed_descriptors.shape[0], batch_size):
            batch_xs = permed_descriptors[j:j+batch_size]
            batch_ys = permed_labels[j:j+batch_size]
            
            if num_tasks==1:
                idx=np.logical_not(np.isnan(batch_ys[:,0]))
                batch_xs_task=batch_xs[idx]
                batch_ys_task=batch_ys[idx, :]
                features, targets = placeholders
                sess.run(train_op, feed_dict={features:batch_xs_task, targets:batch_ys_task})
                
            elif multitask_strategy=="all":
                features, targets, loss_weight = placeholders
                weights=(1-np.isnan(batch_ys))*target_weighting
                batch_ys=np.nan_to_num(batch_ys)
                
                sess.run(train_op, feed_dict={features:batch_xs, targets:batch_ys, loss_weight:weights})

            elif multitask_strategy=="individual":
                features, targets = placeholders
                for k in range(num_tasks):
                    idx=np.logical_not(np.isnan(batch_ys[:,k]))
                    batch_xs_task=batch_xs[idx]
                    batch_ys_task=batch_ys[idx, k]
                    sess.run(train_op[j], feed_dict={features:batch_xs_task, targets:batch_ys_task})
                


    if model_file is not None:
        save_path=saver.save(sess, model_file+'_{0}'.format(model_suffix))
        
    return sess
    
def load_network_state(model_file):
    """ Load the state of a neural network from the model file
    """
    # print('Restoring model')
    
    saver = tf.train.Saver(tf.trainable_variables())

    sess=tf.Session()
    saver.restore(sess, model_file)
    return sess
            
def network_prediction(data, placeholders, predictions, sess, prediction_mode, 
                       batch_size=DEFAULT_MAP['batch_size'],  multitask_strategy=DEFAULT_MAP['multitask_strategy'],
                       write_prediction=False, model_suffix=DEFAULT_MAP['model_suffix'], predictions_file=DEFAULT_MAP['predictions_file']):
    """ Given a neural network, predict the values of a set of data
    """
    descriptors, labels = data
    num_compounds=descriptors.shape[0]
    num_tasks=labels.shape[1] if len(labels.shape)>1 else 1
    
       
    if num_tasks==1:
        features, targets = placeholders
        y_predicted = []
        for i in range(0,num_compounds,batch_size):
            y_predicted.extend(predictions.eval(feed_dict={features:descriptors[i:i+batch_size], 
                                    targets:labels[i:i+batch_size]},session=sess) )
        y_predicted=np.array(y_predicted)
        
        results=obtain_results(labels, y_predicted, prediction_mode)

    elif multitask_strategy=="all":
        features, targets, _ = placeholders
        y_predicted = []
        for i in range(0,num_compounds,batch_size):
            y_predicted.extend(predictions.eval(feed_dict={features:descriptors[i:i+batch_size], 
                                    targets:labels[i:i+batch_size]},session=sess) )
        y_predicted=np.array(y_predicted)
        
        results=obtain_results(labels, y_predicted, prediction_mode)

    elif multitask_strategy=="individual":
        
        features, targets = placeholders
        y_predicted=[]
        
        for j in range(num_tasks):
            
            tmp = []
            for i in range(0, num_compounds, batch_size):
                
                tmp.extend(sess.run(predictions[j], feed_dict={features:descriptors[i:i+batch_size], targets:labels[i:i+batch_size,j]}))

            y_predicted.append(tmp)

            
        
        y_predicted=np.transpose(np.array(y_predicted))
        
        results=obtain_results(labels, y_predicted, prediction_mode)
        
        
    if predictions_file is not None and write_prediction:
        np.savez_compressed("{0}_{1}".format(predictions_file, model_suffix), y_predicted)
    
    return results

   
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
        train_data, test_data = transform(original_train_data, original_test_data, params)
        
        if params['remove_labels']:
            
            train_data, to_delete=remove_activity_labels(train_data, params['percent_to_delete'], params['remove_seed'], params['remove_type'])
            remove_labels_dict[str(j)]=to_delete
            
        if params['impute_labels']:
            kwargs=generate_kwargs_dictionary(impute_missing_values, params)
            train_data=impute_missing_values(train_data, similarity_matrix=similarity_matrix, **kwargs)
        
        try:
            kwargs=generate_kwargs_dictionary(generate_network, params)
            placeholders, predictions, train_op, summaries = generate_network(train_data, test_data, **kwargs)
            if params['restore_network']:
                # session = load_network_state(params['model_file']+'_{0}'.format(j), params['seed'])
                session = load_network_state(params['model_file']+'_{0}'.format(j))
            else:
                kwargs=generate_kwargs_dictionary(network_training, params)
                session = network_training(train_data, placeholders, train_op, summaries, **kwargs)
                
            kwargs=generate_kwargs_dictionary(network_prediction, params)    
            training_results = network_prediction(train_data, placeholders, predictions, session, **kwargs)
            test_results = network_prediction(test_data, placeholders, predictions, session, write_prediction=True,  **kwargs)
            elapsed=time.time()-start
            out=[params[param] for param in sorted(params.keys()) if '_file' not in param]
            out.append(elapsed)
            for i in range(len(datasets)):
                for name in result_names:
                    out.extend([training_results[name][i], test_results[name][i]])
            fout.write('\t'.join(map(str, out))+'\n')
        except KeyboardInterrupt as err:
            raise 
            
        except ValueError as err:
            raise
            
                
    fout.close()
    if args['remove_labels_file'] is not None:
        np.savez_compressed(args['remove_labels_file'], **remove_labels_dict)
            



