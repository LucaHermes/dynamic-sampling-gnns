import wandb
import numpy as np
from collections import defaultdict
from queue import Queue
from tensorflow.keras.metrics import MeanTensor

def merge_configs(keys, configs):
    return { key : config for key, config in zip(keys, configs) }

def from_wandb_config(d):
    unflattened = {}
    
    for k, v in d.items():        
        if isinstance(v, dict):
            v = from_wandb_config(v)
        key_split = k.split('..')
        parent = key_split[0]
        
        if len(key_split[1:]) == 0:
            unflattened[parent] = v
            continue
            
        if parent not in unflattened:
            unflattened[parent] = {}
            
        unflattened[parent]['..'.join(key_split[1:])] = v
        unflattened = from_wandb_config(unflattened)

    return unflattened
        
def to_wandb_config(config, sweep_params=None):
    sweep_config = {}
    
    def pack(d, key=None):
        packed = {}

        for k, v in d.items():

            if key is not None:
                _k = key+'..'+k
            else:
                _k = k

            if isinstance(v, dict):
                subpack = pack(v, key=_k)
                packed.update(subpack)
                continue
                
            if k in ['metric', 'values', 'min', 'max', 'type']:
                if key not in packed:
                    packed[key] = {}
                packed[key][k] = v
                continue

            packed[_k] = v
        return packed
    
    sweep_config = pack(config)

    if sweep_params is not None:
        sweep_config.update(sweep_params)
        
    return sweep_config

def make_edge_probs_plot(step_edge_probs):
    # the n_neightbors means total number of neighbors that walkers 
    # sample from in a sinlge sampling round
    # shape: [n_steps, n_neighbors]
    '''n_steps = len(step_edge_probs)
    n_samples = len(step_edge_probs[0])

    step = np.tile(np.arange(n_steps), [n_samples])
    key = np.repeat([ 'step_%d' % s for s in range(n_steps) ], n_samples)
    data = np.concatenate(step_edge_probs)

    data = list(zip(step, key, data))
    table = wandb.Table(data=data, columns=['step', 'key', 'data'])
    return table'''
    
    hists = []
    keys = []

    for i, probs in enumerate(step_edge_probs):
        hist, bins = np.histogram(probs, bins=20, range=(0, 1))
        hists.append(hist)
        keys.append('step_%d' % i)

    return wandb.plot.line_series(
                xs=bins,
                ys=hists,
                keys=keys,
                title="Edge Probability Distribution",
                xname="Bins")

def make_sampling_entropy_plot(sampling_entropy):
    # shape: [n_steps, n_walkers]
    hists = []
    keys = []

    for i, entropies in enumerate(sampling_entropy):
        hist, bins = np.histogram(entropies, bins=20, range=(0, 1))
        hists.append(hist)
        keys.append('step_%d' % i)

    return wandb.plot.line_series(
                xs=bins,
                ys=hists,
                keys=keys,
                title="Edge Entropy Distribution",
                xname="Bins")
    
class WandbMetricsCallback:
    
    def __init__(self, num_splits=1):
        self.split_metrics = defaultdict(lambda: Queue())
        self.__all__ = np.array([0] * num_splits)
        self.num_splits = num_splits
        self.max_metric = defaultdict(lambda: defaultdict(lambda: -1e10))
        self.mean_max_metric = defaultdict(lambda: -1e10)
        self.step = defaultdict(lambda: 0)
        
    def aggregate(self):
        mean = MeanTensor()
        max_metrics = {}
        for s in range(self.num_splits):
            split_metrics = self.split_metrics[s].get()
            mean.update_state(list(split_metrics.values()))
            
        self.__all__ -= 1
        result = { k : v for k, v in zip(split_metrics.keys(), mean.result()) }
        for k in split_metrics.keys():
            self.mean_max_metric[k] = max(self.mean_max_metric[k], result[k])
            result[k + '-max'] = self.mean_max_metric[k]
        return { 'mean/' + k : v for k, v in result.items() }
    
    def log(self, x, step=None):
        if step is not None:
            x['global_step'] = step
        wandb.log(x)
        
    def callback_fn(self, split):
        
        def callback(metrics):
            # update the max metric dict and create log dict
            _log = {}
            _buffer_metrics = {}
            suffix = ''
            
            # iterate metrics and buffer results
            for k, v in metrics.items():
                try:
                    item = np.array(v).item()
                    if isinstance(item, float) or isinstance(item, int):
                        self.max_metric[split][k] = max(self.max_metric[split][k], v)
                        _buffer_metrics[k] = v
                except:
                    pass
                if self.num_splits > 1:
                    suffix = f'-split_{split}'
                _log[k + suffix] = v
                if k in self.max_metric[split]:
                    _log[k + '-max' + suffix] = self.max_metric[split][k]
            
            # put metric in buffer
            self.split_metrics[split].put(_buffer_metrics)
            # increment number of logs
            self.__all__[split] += 1
            # log individual metric of this split
            self.log(_log, step=self.step[split])
            
            if (self.__all__ > 0).all():
                agg = self.aggregate()
                #agg['epoch'] = epoch
                self.log(agg, step=self.step[split])
                
            self.step[split] += 1
            
        return callback
    
    def __call__(self, metrics):
        self.log(metrics)