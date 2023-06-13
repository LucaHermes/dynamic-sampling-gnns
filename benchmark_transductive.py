import os
# use cpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only print errors
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from functools import partial
import argparse
from pathlib import Path
from utils.experiment_utils import serialize_json_item, read_json, write_json, get_constructors
from tqdm.autonotebook import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np

VALIDATION_METRICS = {
    'sparse_accuracy' : (tf.keras.metrics.SparseCategoricalAccuracy, {})
}

BENCHMARKING_METRICS = {
    'cora' : 'validation/sparse_accuracy',
    'texas' : 'validation/sparse_accuracy',
}

def create_cli_parser():
    parser = argparse.ArgumentParser(add_help=True,
        description='Evaluate DSGNN models.')

    # ------------------------- Training setup args -------------------------
    parser.add_argument('--dataset', '-d', type=str, nargs='?', default='cora',
                    help='Name of the dataset to use. One of "cornell", "texas", '
                    '"cora-10", "pubmed-10", "citeseer-10", "film", "wisconsin", '
                    '"squirrel", "chameleon".')
    parser.add_argument('--name', '-n', type=str, required=False,
                    help='Name of the run. Deault is None and the last run using the dataset '
                          'passed in --dataset is used.'
    )
    parser.add_argument('--epoch', '-e', type=int, required=False,
                    help='Epoch number of the model. If None, best epoch is used.'
    )
    return parser

def extract_performance(_split, experiment_path, metric):
    split = str(_split[1])
    split_nr = int(split.split('_')[-1])
    split_path = experiment_path / split
    split_performance, epochs = get_epoch_metric_from_split(split_path, metric)
    return split_performance, epochs

def get_best_n_epochs(experiment_path, metric, n=1):
    # find all metrics
    experiment_splits = find_all_splits(experiment_path)
    
    with Pool(processes=min(os.cpu_count(), len(experiment_splits))) as pool:
        jobs = pool.imap_unordered(partial(extract_performance, 
                                           experiment_path=experiment_path, 
                                           metric=metric), 
                                   experiment_splits)
        epochs_performance = [ (r, es) for r, es in jobs ]
    
    epochs_performance, epochs = list(zip(*epochs_performance))
    epoch = epochs[-1]
    # filter possible incomplete epochs
    max_epoch_len = max(map(len, epochs_performance))
    epochs_performance = [ e for e in epochs_performance if len(e) == max_epoch_len ]

    epochs_performance = np.stack(epochs_performance)
    avg_over_epochs = epochs_performance.mean(0)
    std_over_epochs = epochs_performance.std(0)
    epoch_idx_best_n = avg_over_epochs.argsort()[::-1][:n]
    return (
        [ epoch[n] for n in epoch_idx_best_n ][0], 
        avg_over_epochs[epoch_idx_best_n][0], 
        std_over_epochs[epoch_idx_best_n][0]
    )

def get_epoch_metric_from_split(split_dir, metric_name):
    # sort ascending by epoch number
    split_dir = sorted(split_dir.glob('*metrics.json'), key=lambda p: int(p.name.split('-')[1]))
    epochs = list(map(lambda s: int(s.name.split('-')[-2]), split_dir))
    return [ read_json(f)[metric_name] for f in split_dir ], epochs

def find_all_splits(experiment_path):
    experiment_splits = list(experiment_path.glob('*split_*'))
    splits = [ (int(s.name.split('_')[-1]), s.name) for s in experiment_splits ]
    return sorted(splits)

def get_experiment(dataset=None, name=None, artifact_path='ckpts'):
    if name is None:
        name = ''
    else:
        name = name + ' '
    
    # if no name was passed, look for the latest experiment
    latest_key = lambda p: float(p.name.split()[-1])
    return sorted(list((Path(artifact_path) / dataset).glob(f'{name}*')), key=latest_key)[-1]

def get_split_epoch_ckpts(experiment_path, epoch, split=None):
    pattern = f'{split}/epoch-{epoch}'
    return next(experiment_path.glob(pattern))

def set_validation_metrics(model, metric_dict):
    model.validation_metrics = {}
    for metric, (metric_fn, args) in metric_dict.items():
        model.validation_metrics[metric] = metric_fn(**args)
        
def reset_validation_metrics(model):
    [ m.reset_state() for m in model.validation_metrics.values() ]
    
def create_benchmark_outputs(exp_path, model, dataset, splits, n_steps):
    model.validation_outputs = True
    
    for datasplit, (_, folder_name) in tqdm(zip(dataset, splits), total=len(splits), desc='Validating all epochs splits ...'):
        datasplit = datasplit['valid']
        epoch_paths = list((exp_path / folder_name).glob('*/ckpt-1.index'))
        
        for epoch_ckpt in tqdm(epoch_paths, desc='Validating epoch ...', leave=False):
            reset_validation_metrics(model)
            ckpt_dir = epoch_ckpt.parent
            
            if Path(str(ckpt_dir) + '-metrics.json').is_file():
                continue
            
            model.load(ckpt_dir, tf.keras.optimizers.Adam())
            model.validation_outputs_path = ckpt_dir.parent
            split_result = model.evaluate(datasplit, n_steps)
    

def benchmark(create_model_fn, create_dataset_fn, dataset=None, epoch=None, name=None, metric='validation/sparse_accuracy', 
              artifact_path='ckpts', results_out='results'):
    print('Loading Artifacts ...')
    
    if dataset == 'molhiv':
        metric='validation/auc'
        val_metric_fns = {
            'binary_accuracy' : (tf.keras.metrics.BinaryAccuracy, {}),
            'auc' : (tf.keras.metrics.AUC, {})
        }
    else:
        metric='validation/sparse_accuracy'
        val_metric_fns = {
            'sparse_accuracy' : (tf.keras.metrics.SparseCategoricalAccuracy, {})
        }
    
    exp_path = get_experiment(dataset=dataset, name=name, artifact_path=artifact_path)
    splits = find_all_splits(exp_path)
    exp_config = read_json(exp_path / 'config.json')
    model = create_model_fn(**exp_config['model'])
    set_validation_metrics(model, val_metric_fns)
    dataset = create_dataset_fn(**exp_config['data'])
    if isinstance(dataset, dict):
        dataset = [dataset] * len(splits)
    create_benchmark_outputs(exp_path, model, dataset, splits, exp_config['train']['n_steps'])
    dataset = create_dataset_fn(**exp_config['data'])
    if isinstance(dataset, dict): # only a single dataset - evaluate seeds
        dataset = [dataset] * len(splits)
    print('Found Experiment here:', str(exp_path))
    best_val_epoch, avg_metric, std_metric = get_best_n_epochs(exp_path, metric)
    print(f'Best performing epoch on metric {metric}:')
    print(f' * Epoch: {best_val_epoch}')
    print(f' * Average {metric} of epoch: {best_val_epoch}')
    print(f' * This {metric} over all splits: {avg_metric.round(2)} (+/- {std_metric.round(2)})')
    output_path = Path(results_out) / exp_path.name / f'epoch_{best_val_epoch}'
    output_path.mkdir(parents=True, exist_ok=True)
    model.validation_outputs = False
    

    # compute results for each split
    for datasplit, (_, folder_name) in tqdm(zip(dataset, splits), total=len(splits), desc='Benchmarking splits ...'):
        print(dataset)
        print(datasplit)
        datasplit = datasplit['test']
        ckpt_dir = get_split_epoch_ckpts(exp_path, best_val_epoch, split=folder_name)
        model.load(ckpt_dir, tf.keras.optimizers.Adam())
        for i in range(10):
            reset_validation_metrics(model)
            split_result = model.evaluate(datasplit, exp_config['train']['n_steps'])
            x, y = next(iter(datasplit()))
            _, _, _, walker_output_states = model((None, x['node_features'], x['edge_list'], x['node_degrees'], x.get('edge_features', None)), 
                n_steps=exp_config['train']['n_steps'],
                hard_sampling=False,
                soft_select=False,
                graph_mask=y['graph_mask'],
                output_all_states=True,
                
            )
            # save trajectories
            np.save(output_path / f'{i}_{folder_name}-trajectories.npy', np.stack(walker_output_states))
            json_serialized_values = map(serialize_json_item, split_result.values())
            split_result_dict = dict(zip(
                split_result.keys(), 
                json_serialized_values
            ))
            # save quantitative results
            write_json(output_path / f'{i}_{folder_name}-metrics.json', split_result_dict)
    
    metrics_files = list(output_path.glob('*metrics.json'))
    print(np.mean([ read_json(f)[metric] for f in metrics_files ]))
    print('Done!')
    print(f'Output artifacts saved to {output_path}.')

if __name__ == '__main__':
    args, unknown_args = create_cli_parser().parse_known_args()
    args.dataset = args.dataset.lower()
    create_model, get_dataset, n_data_splits = get_constructors(args.dataset)
    benchmark(create_model, get_dataset, **vars(args))