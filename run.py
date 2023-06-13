import tensorflow as tf
from pprint import pprint
import argparse
import os
import json
import wandb
import warnings
import numpy as np
from data import dataset
import utils.wandb_utils as wandb_utils
import utils.experiment_utils as experiment_utils
import configs
from data import webkb_datasets
from datetime import datetime

WANDB_PROJECT_NAME = 'DynamicSamplingGNN'

METRICS = {
    'auc' : tf.keras.metrics.AUC,
    'sparse_auc' : experiment_utils.SparseAUC,
    'mae' : tf.keras.metrics.MeanAbsoluteError,
    'binary_accuracy' : tf.keras.metrics.BinaryAccuracy,
    'sparse_accuracy' : tf.keras.metrics.SparseCategoricalAccuracy,
}

def create_cli_parser():
    parser = argparse.ArgumentParser(add_help=True,
        description='Train ACGNN models.')

    # ------------------------- Training setup args -------------------------
    parser.add_argument('--dataset', '-d', type=str, nargs='?', default='cora',
                    help='Name of the dataset to use. One of "cornell", "texas", '
                    '"cora-10", "pubmed-10", "citeseer-10", "film", "wisconsin", '
                    '"squirrel", "chameleon".')
    parser.add_argument('--debug', action='store_true',
                    help='To make debugging easier, disables wandb logs and '
                    'tensorflow graph execution for this run.')
    # Model Settings
    parser.add_argument('--selfloops', action='store_true', default=False,
                        help='Flag whether to use selfloops, defaults to False.')
    parser.add_argument('--temporal-encoding', action='store_true', default=False,
                        help='Flag whether to use the temporal encoding, defaults to False.')
    # WandB settings
    parser.add_argument('--name', type=str, help='Name of the wandb experiment.')
    parser.add_argument('--group', type=str, help='Name of the wandb group.')
    parser.add_argument('--sweep_context', action='store_true', default=False,
                        help='Flag indicating if this run is part of a sweep, defaults to False.')
    # Experiment settings
    parser.add_argument('--seed', type=int, help='Specifies a seed, use random seed if not passed.')
    parser.add_argument('--validation_outputs', action='store_true', default=False)
    parser.add_argument('--walkers-per-node', type=int, default=None, help='Number of walkers per node.'
                        'Only required if `uniform` distribution is used.')
    parser.add_argument('--attention-activation', type=str, default='leaky_relu')
    parser.add_argument('--attention-dropout', type=float, default=0.0)
    parser.add_argument('--attention-units', type=int, default=64)
    parser.add_argument('--attention', type=str, default='gat',
                        help='Attention model to use. One of `gat`, `dot`, `simple_dot` or `zero`.')
    parser.add_argument('--walker-embds', action='store_true', default=False,
                        help='Whether to initialize the walker state with a learnable embedding.'
                        'Only applies if `uniform` initial walker distribution is used.')
    parser.add_argument('--input-dropout', type=float, default=0.0)
    parser.add_argument('--input_weight_l2_reg', type=float, default=0.01)
    parser.add_argument('--type', type=str, default='')
    parser.add_argument('--units', type=int, default=64,
                        help='Size of the latent vectors of the state model.')
    parser.add_argument('--state-model-activation', type=str, default='relu')
    parser.add_argument('--state-model-dropout', type=float, default=0.0)
    parser.add_argument('--state-model', type=str, default='res')
    parser.add_argument('--n-steps', type=int, default=3)
    parser.add_argument('--train-parallel', action='store_true', default=False)
    parser.add_argument('--scale-lr', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser

def train(config, dataset_fn, create_model_fn, ckpt_path=None, n_data_splits=1, validation_outputs=False):
    '''
    Trains a model created by calling create_model_fn on the dataset created by dataset_fn.
    '''
    seed = config['seed']
    config = wandb_utils.from_wandb_config(config)#['parameters']
    config = experiment_utils.check_config(config, fix=True)
    
    if ckpt_path is not None:
        os.path.isdir(ckpt_path) or os.makedirs(ckpt_path)
        config_file = os.path.join(ckpt_path, 'config.json')

        with open(config_file, 'w') as config_file:
            json.dump(config, config_file)

    dataset_config = config['data']
    model_config = config['model']
    train_config = config['train']

    pprint(dataset_config)
    pprint(model_config)
    pprint(train_config)

    wandb_callback = wandb_utils.WandbMetricsCallback(n_data_splits)
    ckpt_callback = experiment_utils.CheckpointCallback(n_data_splits)

    print('#' * 80)
    print('Training with seed:  ', seed)
    print('Model walkers_per_node: ', model_config['walkers_per_node'])
    print('Model walker_state_size:', model_config['state_model_config']['units'])
    print('#' * 80)
    
    optimizer_config = train_config.pop('optimizer_config')
    metrics_fn = train_config.pop('metrics')
    seeds = [seed]
    n_seeds = 1
    np.random.seed(seed)
    tf.random.set_seed(seed)

    dataset_gen = dataset_fn(**dataset_config)
    early_stop = train_config.pop('early_stop', None)
    perform_stopping = False
    
    if early_stop is not None:
        perform_stopping, stopping_metric, threshold = early_stop
    
    if n_data_splits == 1:
        dataset_gen = [dataset_gen]
        n_seeds = train_config.pop('n_seeds', 1)
        if n_seeds > 1:
            seeds = np.random.RandomState(seed).randint(2**32 - 1, size=n_seeds)
            dataset_gen = dataset_gen * n_seeds
            wandb_callback = wandb_utils.WandbMetricsCallback(n_seeds)
            ckpt_callback = experiment_utils.CheckpointCallback(n_seeds)
    
    # Iterate over multiple dataset splits. Used to implement the
    # cross-validation scheme.
    for split_idx, data_gen in enumerate(dataset_gen):
        artifact_dir = f'results/split_{split_idx}'
        
        if ckpt_path is not None:
            ckpt_path_split = os.path.join(ckpt_path, f'split_{split_idx}')
        else:
            ckpt_path_split = None
        
        if n_seeds > 1: # only if n_data_splits == 1
            np.random.seed(seeds[split_idx])
            tf.random.set_seed(seeds[split_idx])
            artifact_dir = f'results/seed_{seeds[split_idx]}_{split_idx}'
        
        print('Start training data split', split_idx)
        model = create_model_fn(**model_config)

        optimizer = tf.keras.optimizers.get(optimizer_config['name'])
        optimizer = optimizer.from_config(optimizer_config)

        metrics = { m : (METRICS[m], args) for m, args in metrics_fn }
        
        model.validation_outputs = validation_outputs
        result = model.train(
            data_gen['train'], 
            optimizer=optimizer,
            validation_data_gen=data_gen.get('valid'),
            metrics=metrics,
            ckpt_path=ckpt_path_split,
            metrics_callback=wandb_callback.callback_fn(split_idx),
            checkpoint_callback=ckpt_callback.callback_fn(split_idx),
            **train_config)
        
        if perform_stopping:
            metric_value = wandb_callback.max_metric[split_idx][stopping_metric]
            if metric_value < threshold:
                print('[INFO] Terminating due to poor performance.')
                print(f'Metric {stopping_metric} of first run was {metric_value} '
                      f'wich is lower than the threshold of {threshold}.')
                break
            
    return 0, model

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args, unknown_args = create_cli_parser().parse_known_args()
    args.dataset = args.dataset.lower()

    if args.debug:
        os.environ['WANDB_MODE'] = 'offline'
        tf.config.run_functions_eagerly(True)

    if args.dataset == 'cora':
        config_stack = (
            configs.cora_dataset_config, 
            configs.cora_model_config, 
            configs.cora_train_config
        )
        
    elif args.dataset.lower() in ['cora-10', 'pubmed-10', 'citeseer-10']:
        dataset_name = args.dataset.rstrip('-10').lower()
        model_config = configs.citation_model_config
        model_config['n_classes'] = dataset.CITATION_CLASSES[dataset_name]
        config_stack = (
            configs.citation_dataset_config, 
            model_config, 
            configs.citation_train_config
        )
        
    elif args.dataset.lower() in webkb_datasets.NAMES:
        model_config = configs.webkb_model_config # configs.cora_model_config
        model_config['n_classes'] = webkb_datasets.N_CLASSES
        config_stack = (
            configs.webkb_dataset_config, 
            model_config, 
            configs.webkb_train_config
        )

    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented.')

    create_model, get_dataset, n_data_splits = experiment_utils.get_constructors(args.dataset)

    # workaround to get multi-level configs working in wandb
    wandb_config = wandb_utils.merge_configs(('data', 'model', 'train'), config_stack)
    wandb_config = experiment_utils.merge_args(wandb_config, args)
    wandb_config = wandb_utils.to_wandb_config(wandb_config)
    wandb_config.update(vars(args))
    pprint(wandb_config)
    
    name = args.name
    name = f'{name}' if name is not None else ''
    #experiment_name = '%s %s' % (name, str(uuid.uuid1()))
    experiment_name = '%s %s' % (name, str(datetime.now().timestamp()))

    if not args.sweep_context:
        logger_config = {
            'project' : WANDB_PROJECT_NAME,
            'group'   : args.group,
            'name'    : experiment_name,
            'config'  : wandb_config,
            #'id'      : model.id
        }

        if args.seed is None:
            np.random.seed(42)
            seeds = np.random.randint(0, np.iinfo(np.int32).max, size=10)
        else:
            seeds = [args.seed]

        for seed in seeds:
            wandb.init(**logger_config)
            config = wandb.config
            config.update({ 'seed' : seed }, allow_val_change=True)
            ckpt_path = os.path.join('ckpts', args.dataset, experiment_name)
            exit_code, _ = train(config, get_dataset, create_model, ckpt_path=ckpt_path, 
                                 n_data_splits=n_data_splits, validation_outputs=args.validation_outputs)
            wandb.finish(exit_code=exit_code)
    else:
        wandb.init(config=wandb_config)
        config = wandb.config
        print('\nSweep Iteration Setup:\n')
        pprint(config)
        
        exit_code, _ = train(config, get_dataset, create_model, 
                                 n_data_splits=n_data_splits)
        wandb.finish(exit_code=exit_code)
        tf.keras.backend.clear_session()