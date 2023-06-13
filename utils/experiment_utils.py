import model
import model.base_model
from data import dataset, webkb_datasets

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.metrics import MeanTensor
import networkx as nx
import numpy as np

from collections import defaultdict
from queue import Queue
from functools import partial
import json

def get_constructors(dataset_name):
    '''
    Returns the functions to load the dataset and the model for
    the specified dataset name.
    '''
    if dataset_name.lower() == 'cora':
        get_dataset = dataset.create_cora_dataset
        create_model = create_cora_model
        n_data_splits = 1
        
    elif dataset_name.lower() in ['cora-10', 'pubmed-10', 'citeseer-10']:
        n_data_splits = 10
        dataset_name = dataset_name.rstrip('-10').lower()
        get_dataset = partial(
            dataset.create_citation_10fold_splits, 
            name=dataset_name, 
            data_dir='data', 
            splits_dir='splits'
        )
        create_model = create_cora_model

    elif dataset_name.lower() in webkb_datasets.NAMES:
        n_data_splits = 10
        get_dataset = partial(
            webkb_datasets.create_webkb_10fold_splits, 
            name=dataset_name.lower(), 
            data_dir='data', 
            splits_dir='splits'
        )
        create_model = create_cora_model
        
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented.')
    
    return create_model, get_dataset, n_data_splits

def build_modules(state_model_config=None, attention_model_config=None,
                  sampling_layer_config=None, **kwargs):
    # create all non-optional layers
    att_model = model.build_module_from_config(model.attention_modules, attention_model_config)
    sampling_layer = model.build_module_from_config(model.sampling_modules, sampling_layer_config)
    walker_state_model = model.build_module_from_config(model.state_modules, state_model_config)
    return att_model, sampling_layer, walker_state_model, None
    

def create_cora_model(n_node_features, walkers_per_node, max_walkers_per_node, output_activation, use_state=True, n_classes=None, scale_lr=0.01,
    stepwise_readout=True,  identify_walkers_per_node=True, pooling_level='node', edge_input=False, input_dropout=0., input_l2=0.01, **module_config):
    modules = build_modules(**module_config)
    attention_module, sampling_module, state_module = modules

    Cell = model.state_cells.get(module_config.get('cell_type', 'acgnn'))
    
    input_transform = tf.keras.Sequential([
        tf.keras.layers.Dropout(input_dropout),
        tf.keras.layers.Dense(n_node_features, use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.L2(input_l2)),
    ])
    
    #tf.keras.layers.Dense(n_node_features, use_bias=False, 
    #    kernel_regularizer=tf.keras.regularizers.L2())
    #input_transform = tf.keras.layers.Dense(n_node_features, use_bias=False)
    # create model output layer
    embedding_layer = tf.keras.layers.Dense(state_module.units, activation='relu', use_bias=False)
    #embedding_layer = tf.keras.layers.Dense(n_classes, use_bias=False)
    # create classification decoder head
    output_transform = tf.keras.layers.Dense(n_classes, use_bias=False, activation=output_activation)
    #output_transform = tf.keras.layers.Softmax(axis=-1)
    input_edge_transform, edge_model, edge_dim = None, None, None
    
    if edge_input:
        input_edge_transform = tf.keras.layers.Dense(n_node_features)
        edge_model = tf.keras.layers.Dense(n_node_features)
        edge_dim = n_node_features

    # create the ACGNN cell
    gnn_cell = Cell(attention_module,
                               sampling_module,
                               state_model=state_module,
                               edge_model=edge_model,
                               edge_dim=edge_dim,
                               walkers_per_node=walkers_per_node,
                               max_walkers_per_node=max_walkers_per_node,
                               identify_walkers_per_node=identify_walkers_per_node)

    cora_model = model.base_model.DSGNN(gnn_cell,
                        walker_state_size=state_module.units,
                        input_transform=input_transform,
                        input_edge_transform=input_edge_transform,
                        embedding_layer=embedding_layer,
                        output_transform=output_transform,
                        stepwise_readout=stepwise_readout,
                        pooling_level=pooling_level,
                        scale_lr=scale_lr,
                        use_state=True)

    return cora_model

def combine_training_results(result_non_sampling, results_sampling=None):
    if results_sampling is None:
        results_sampling = {}
        
    result = dict(zip(map(
        lambda s: 'sampling_model/' + s, 
        results_sampling.keys()), 
        results_sampling.values()))
    result.update(dict(zip(map(
        lambda s: 'non_sampling_model/' + s, 
        result_non_sampling.keys()), 
        result_non_sampling.values())))

    if 'val_accuracy' in result_non_sampling:
        result['val_accuracy'] = result_non_sampling['val_accuracy']

    return result

class SparseAUC(tf.keras.metrics.AUC):

    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

    def update_state(self, y_true, y_pred):
        hot_labels = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=self.num_labels)
        return super().update_state(hot_labels, y_pred)
    
    
def check_config(config, fix=False):
    '''
    Helper function to enable sweeps where selected variables are not necessarily valid configurations.
    '''
    
        
    if config['model']['walkers_per_node'] == 'degree':
        config['model']['walkers_per_node'] = 6

    if config['model']['state_model_config']['name'] == 'res2d':
        ks = config['model']['state_model_config'].get('kernel_size', config['model']['walkers_per_node'])
        
        if ks != config['model']['walkers_per_node'] or 'kernel_size' not in config['model']['state_model_config']:
            if fix:
                print('[FIXING CONFIG] Replace Res2D kernel-size with number of walkers per node.')
                config['model']['state_model_config']['kernel_size'] = config['model']['walkers_per_node']
            else:
                raise AttributeError('Kernel size of a Res2D layer must match walkers_per_node.')
    else:
        # if state model is not res2D, kernel size has to be dropped
        config['model']['state_model_config'].pop('kernel_size', None)
        #config['model']['identify_walkers_per_node'] = False
            
    if config['model']['state_model_config']['units'] != config['model']['n_node_features']:
        if fix:
            print('[FIXING CONFIG] Replace State Model Units with Node encoding units.')
            config['model']['state_model_config']['units'] = config['model']['n_node_features']
        else:
            raise AttributeError('State Model Units must be similar to Node encoding units (n_node_features).')
        
    if 'metrics' in config['train']:
        metrics = [ k for k, args in config['train']['metrics'] ]
        if 'sparse_auc' in metrics:
            i = metrics.index('sparse_auc')
            num_labels = config['train']['metrics'][i][1]['num_labels']
            if config['model']['n_classes'] != num_labels:
                if fix:
                    print('[FIXING CONFIG] Replace sparse auc num_labels attribute with the n_classes '
                        'specified in the model config')
                    config['train']['metrics'][i][1]['num_labels'] = config['model']['n_classes']
                else:
                    raise AttributeError('Sparse AUC num_labels should be similar to number of classes.')
    
    
    if config['model']['state_model_config']['name'] == 'res2d':
        if config['model']['walkers_per_node_dist'] != 'uniform':
            dist = config['model']['walkers_per_node_dist']
            if fix:
                print(f'[FIXING CONFIG] Replace walkers per node distribution {dist} '
                       'with "uniform" distribution')
                config['model']['walkers_per_node_dist'] = 'uniform'
            else:
                raise AttributeError('When Res2D is used as the state model the walker distribution has to be "uniform". '
                                     f'Distribution {dist} was passed instead.')
        
        config['model']['max_walkers_per_node'] = config['model']['walkers_per_node']
        
    if config['model']['walkers_per_node_dist'] == 'uniform':
        config['model']['max_walkers_per_node'] = config['model']['walkers_per_node']
    else:
        config['model']['identify_walkers_per_node'] = False

    return config

def read_json(file):
    with open(file) as f:
        return json.load(f)
    
def write_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
    
def serialize_json_item(x):
    if tf.nest.is_nested(x):
        unpacked_x = list(map(lambda _x: _x.numpy().tolist(), x))
        return dict(enumerate(unpacked_x))
    x = np.array(x)
    # if it is a scalar, return it as a python scalar
    if np.isscalar(x):
        return x.item()
    # if x is not a scalar, convert it to a python list
    return x.tolist()

class CheckpointCallback:
    
    def __init__(self, target_metric='accuracy', strategy='maximize', num_splits=1, keep_n=3):
        self.target_metric = target_metric
        self.strategy = strategy
        self.num_splits = num_splits
        self.keep_n = keep_n
        
        if strategy == 'maximize':
            self.compare = lambda x, y: np.greater(x, y).any()
        if strategy == 'minimize':
            self.compare = lambda x, y: not np.greater(x, y).any()
            
        # store performance of individual splits
        self.split_performance = defaultdict(lambda: Queue())
        # keep track of which split reported back
        self.__all__ = np.array([0] * num_splits)
        self.num_splits = num_splits
        self.best_by_strategy = []
        self.best_ckpts_by_strategy = []
        
    def aggregate(self):
        mean = MeanTensor()
        max_metrics = {}
        ckpts = []
        
        # compute the average performance per step/epoch
        for s in range(self.num_splits):
            split_metrics, ckpt = self.split_performance[s].get()
            mean.update_state(split_metrics)
            ckpts.append(ckpt)
            
        self.__all__ -= 1
        mean_over_splits = mean.result()
        
        if self.compare(mean_over_splits, self.best_by_strategy):
            self.best_by_strategy.append(mean_over_splits)
            self.best_ckpts_by_strategy.append(ckpts)
        elif len(self.best_by_strategy) < self.keep_n:
            self.best_by_strategy.append(mean_over_splits)
            self.best_ckpts_by_strategy.append(ckpts)
            
        out = sorted(zip(self.best_by_strategy, self.best_ckpts_by_strategy))[-self.keep_n:]
        perf_out, ckpt_out = list(zip(*out))
        
        return perf_out, ckpt_out
        
    def callback_fn(self, split):
        
        def callback(metrics, ckpt):
            # update the max metric dict and create log dict
            _log = {}
            _buffer_metrics = {}
            suffix = ''
            agg = None
            
            # put metric and checkpoint in buffer
            self.split_performance[split].put((
                metrics[self.target_metric],
                ckpt
            ))
            # increment number of callbacks from this split
            self.__all__[split] += 1
            # if all callbacks for one training step/epoch are collected
            if (self.__all__ > 0).all():
                agg = self.aggregate()
            return agg
            
        return callback

def merge_args(config, args):
    print(args)
    config['data']['selfloops'] = args.selfloops
    config['model']['add_temporal_features'] = args.temporal_encoding
    
    if args.walkers_per_node:
        config['model']['walkers_per_node_dist'] = 'uniform'
        config['model']['max_walkers_per_node'] = args.walkers_per_node
        config['model']['walkers_per_node'] = args.walkers_per_node
        if args.state_model == 'res2d':
            config['model']['kernel_size'] = args.walkers_per_node
    
    config['model']['attention_model_config']['activation'] = args.attention_activation
    config['model']['attention_model_config']['dropout'] = args.attention_dropout
    config['model']['attention_model_config']['units'] = args.attention_units
    config['model']['attention_model_config']['name'] = args.attention
    config['model']['input_l2'] = args.input_weight_l2_reg
    config['model']['identify_walkers_per_node'] = args.walker_embds
    config['model']['input_dropout'] = args.input_dropout
    config['model']['n_node_features'] = args.units
    config['model']['cell_type'] = 'dsgnn' + args.type
    config['model']['scale_lr'] = args.scale_lr
    config['model']['state_model_config']['units'] = args.units
    config['model']['state_model_config']['activation'] = args.state_model_activation
    config['model']['state_model_config']['dropout'] = args.state_model_dropout
    config['model']['state_model_config']['name'] = args.state_model
    
    config['train']['n_steps'] = args.n_steps
    config['train']['train_separately'] = not args.train_parallel
    config['train']['optimizer_config']['learning_rate'] = args.lr
    
    return config