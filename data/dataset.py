import networkx as nx
import numpy as np
import tensorflow as tf
import pickle
from collections import defaultdict
import os
import sys
import pandas as pd
from functools import partial
from collections.abc import Iterable
import scipy.sparse as sp
from pathlib import Path
import requests


CITATION_CLASSES = {
    'cora' : 7,
    'pubmed' : 3,
    'citeseer' : 6
}

SPLITS_URL = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
              'master/splits')
DATA_URL = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
             'master/data')

def to_edge_list_graph(edge_index, n_nodes, add_edge_index=False):
    senders, receivers = tf.unstack(edge_index, axis=-1)
    if add_edge_index:
        receivers = tf.stack((receivers, tf.range(len(receivers), dtype=tf.int32)), axis=-1)
    edge_list = tf.ragged.stack_dynamic_partitions(receivers, senders, n_nodes)
    node_degrees = edge_list.row_lengths()
    return edge_list, node_degrees

def to_edge_list(inputs, targets, add_edge_index=True):    
    edge_list, node_degrees = to_edge_list_graph(
        inputs['edge_index'], 
        inputs['num_nodes'],
        add_edge_index=add_edge_index)
    del inputs['edge_index']
    del inputs['num_nodes']
    inputs['edge_list'] = edge_list
    inputs['node_degrees'] = node_degrees
    return inputs, targets

def get_tf_data_specs(data_generator):
    graph, labels = next(data_generator())
    tf_data_spec = tuple()
    
    for d in [graph, labels]:
        tf_spec = {}
        
        for k, v in d.items():
            if isinstance(v, int):
                s = tf.TensorSpec(shape=[], dtype=tf.int32)
            elif isinstance(v, float):
                s = tf.TensorSpec(shape=[], dtype=tf.float32)
            elif isinstance(v, Iterable):
                shape = [None, *np.shape(v)[1:]]
                dtype = v.dtype
                s = tf.TensorSpec(shape=shape, dtype=dtype)
            else:
                raise NotImplementedError('Signature cannot be inferred for key', k, 'and value', v)
            tf_spec[k] = s
            
        tf_data_spec = tf_data_spec + (tf_spec,)
        
    return tf_data_spec

def add_selfloops(inputs, targets, feats=None):
    node_ids = tf.range(inputs['num_nodes'])
    selfloops = tf.stack((node_ids, node_ids), axis=-1)
    # add selfloops to edge-index
    inputs['edge_index'] = tf.concat((inputs['edge_index'], selfloops), axis=0)
    
    # if the given dataset contains edge features, add a `selfloop` feature
    if 'edge_features' in inputs:
        if feats is None:
            feats = tf.zeros_like(inputs['edge_features'][:1])
        selfloop_features = tf.tile(feats, [inputs['num_nodes'], 1])
        inputs['edge_features'] = tf.concat((inputs['edge_features'], selfloop_features), axis=0)
        
    return inputs, targets

def to_tf_dataset(data_gen, selfloops=False, selfloop_feats=None):
    tf_data_spec = get_tf_data_specs(data_gen)
    has_edge_features = 'edge_features' in tf_data_spec[0]
    tfds = tf.data.Dataset.from_generator(data_gen,
        output_signature=tf_data_spec
    )
    if selfloops:
        add_selfloops_fn = partial(add_selfloops, feats=selfloop_feats)
        tfds = tfds.map(add_selfloops_fn, num_parallel_calls=tf.data.AUTOTUNE)
    to_edge_list_fn = partial(to_edge_list, add_edge_index=has_edge_features)
    tfds = tfds.map(to_edge_list_fn,  num_parallel_calls=tf.data.AUTOTUNE)
    tfds = tfds.prefetch(tf.data.AUTOTUNE)
    return tfds

def batch_graph(graph_gen, batch_size):
    '''
    Create batches with batch_size many graphs from a dataset generator.
    '''
    graph_gen = graph_gen()
    current_batch, current_label = next(graph_gen)
    current_label['graph_mask'] = [0] * current_batch['num_nodes']
    
    for i, (graph, label) in enumerate(graph_gen):
        if (i + 1) % batch_size == 0:
            current_label['graph_mask'] = np.array(current_label['graph_mask'])
            yield current_batch, current_label
            current_batch, current_label = graph, label
            current_label['graph_mask'] = [0] * graph['num_nodes']
            continue
        
        current_batch['edge_index'] = np.concatenate((
            current_batch['edge_index'],
            graph.pop('edge_index') + current_batch['num_nodes']
        ))
        current_label['graph_mask'] += [i % batch_size + 1] * graph['num_nodes']
        current_batch['num_nodes'] += graph.pop('num_nodes')
        
        for k, v in graph.items():
            current_batch[k] = np.concatenate((current_batch[k], v))
        
        current_label['labels'] = np.concatenate((current_label['labels'], label['labels']))
        if 'mask' in current_label:
            current_label['mask'] = np.concatenate((current_label['mask'], label['mask']))
    
    # yield the remainder
    current_label['graph_mask'] = np.array(current_label['graph_mask'])
    yield current_batch, current_label


def build_batched_replicated_graph(edge_index, batch_size, n_nodes):
    edge_index = edge_index.astype(np.int32)
    batched_edge_index = [ edge_index + i * n_nodes for i in range(batch_size) ]
    batched_edge_index = np.concatenate(batched_edge_index)
    return batched_edge_index

def build_batched_obg_graph(graph_dicts):
    batched_graph = defaultdict(lambda: [])
    node_offset = 0
    
    for graph in graph_dicts:
        for k, v in graph.items():
            if k == 'edge_index':
                #selfloops = np.repeat(np.arange(graph['num_nodes'])[np.newaxis], 2)
                batched_graph[k].append((v + node_offset).T)
            else:
                batched_graph[k].append(v)
        
        node_offset += graph['num_nodes']
        
    for k, v in batched_graph.items():
        if k == 'num_nodes':
            # batched_graph[k] = np.sum(v)
            continue
        batched_graph[k] = np.concatenate(v)
    
    return dict(batched_graph)

def download_file(url, output_file):
    '''
    Download a file from url and save it to output_file.
    '''
    r = requests.get(url)
    if not r.ok:
        raise requests.HTTPError(f'Url {url} is not available [Code: {r.status_code}]')
    
    our_dir = os.path.dirname(output_file)
    os.path.isdir(our_dir) or os.makedirs(our_dir)
    
    with open(output_file, 'wb') as f:
        f.write(r.content)
        
def load_geomgcn_split(name, split_nr, splits_dir):
    assert split_nr < 10 and split_nr >= 0
    assert isinstance(split_nr, int)
    split_file = f'{name}_split_0.6_0.2_{split_nr}.npz'
    split_out = splits_dir / split_file

    if not split_out.exists():
        print(f'Downloading split {split_nr} of {name} dataset, save to {split_out}.')
        split_url = SPLITS_URL + '/' + split_file
        download_file(split_url, split_out)
    
    return { k : v.astype(bool) for k, v in np.load(str(split_out)).items() }

def create_cora_dataset(batch_size, selfloops=False):
    # load the data: x, y, tx, ty, graph
    names = ['x', 'y', 'tx', 'ty', 'graph', 'allx', 'ally']
    objects = []

    for n in names:
        with open(f"data/cora/ind.cora.{n}", 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))
            
    x, y, tx, ty, graph, allx, ally = tuple(objects)

    test_idx = np.loadtxt('data/cora/ind.cora.test.index', dtype=np.int32)

    e_index = np.concatenate([ np.stack((np.ones(len(v), np.int32) * k, v)) 
                              for k, v in graph.items() ], axis=-1).astype(np.int32).T
    node_features = np.concatenate((allx.todense(), tx.todense()), axis=0)

    n_nodes = len(node_features)

    #edge_list, edge_index, node_degrees = build_batched_graph(e_index, batch_size, n_nodes)
    #edge_list, edge_index, node_degrees = 

    node_features = np.tile(node_features, [batch_size, 1])
    batch_offsets = np.arange(batch_size)[:, np.newaxis] * n_nodes

    test_mask = test_idx.astype(np.int32)[np.newaxis]
    #test_mask = np.tile(test_mask, [batch_size])
    test_mask = np.reshape(test_mask + batch_offsets, -1)
    train_mask = np.arange(len(y)).astype(np.int32)[np.newaxis]
    train_mask = np.reshape(train_mask + batch_offsets, -1)
    #train_mask = np.tile(train_mask, [batch_size])
    val_mask = np.arange(len(y), len(y)+500).astype(np.int32)[np.newaxis]
    val_mask = np.reshape(val_mask + batch_offsets, -1)
    #val_mask = np.tile(val_mask, [batch_size])

    node_labels = np.concatenate((ally, ty), axis=0)
    node_labels = node_labels.argmax(-1).astype(np.int32)
    node_labels = np.tile(node_labels, [batch_size])[:, np.newaxis]

    train_labels = node_labels[train_mask]
    val_labels = node_labels[val_mask]
    
    def dataset_gen(subset):
        if subset == 'train':
            labels = train_labels
            mask = train_mask
        elif subset == 'valid':
            labels = val_labels
            mask = val_mask
        yield {
            'node_features' : node_features, 
            'edge_index' : e_index,
            'num_nodes' : len(node_features),
        }, {
            'mask' : mask, 
            'labels' : labels,
        }
        
    train_gen = partial(dataset_gen, 'train')
    train_gen = partial(batch_graph, train_gen, 1)
    train_gen = partial(to_tf_dataset, train_gen, selfloops=selfloops)
    valid_gen = partial(dataset_gen, 'valid')
    valid_gen = partial(batch_graph, valid_gen, 1)
    valid_gen = partial(to_tf_dataset, valid_gen, selfloops=selfloops)

    return { 
        'train' : train_gen,
        'valid' : valid_gen  
    }
    
def full_load_citation(dataset_str, raw_dir):
    """
    Code adapted from Yan et al. 
    https://github.com/Yujun-Yan/Heterophily_and_oversmoothing/blob/4b555a229b570c6802c7074e80c3928f7c9bbc66/process.py#L33
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        file = os.path.join(raw_dir, "ind.{}.{}".format(dataset_str, names[i]))
        
        if not os.path.isfile(file):
            file_url = DATA_URL + f'/{os.path.basename(file)}'
            download_file(file_url, output_file=file)
            
        with open(file, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))
    
    text_idx_file = os.path.join(f"data/{dataset_str}/ind.{dataset_str}.test.index")
    
    if not os.path.isfile(text_idx_file):
        file_url = DATA_URL + f'/{os.path.basename(text_idx_file)}'
        download_file(file_url, output_file=text_idx_file)
        
    test_idx_reorder = np.loadtxt(text_idx_file, dtype=np.int32)

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_range = np.sort(test_idx_reorder)
    test_idx_range_full = range(test_idx_reorder.min(), test_idx_reorder.max() + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    if len(test_idx_range_full) != len(test_idx_range):
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position, mark them
        # Follow H2GCN code
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        non_valid_samples = set(test_idx_range_full) - set(test_idx_range)
    else:
        non_valid_samples = set()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    non_valid_samples = list(non_valid_samples.union(set(list(np.where(labels.sum(1) == 0)[0]))))
    labels = np.argmax(labels, axis=-1)[:, np.newaxis]

    features = np.array(features.todense())
    sparse_mx = sp.coo_matrix(adj).astype(np.float32)
    edge_index = np.stack((sparse_mx.row, sparse_mx.col), axis=-1).astype(np.int32)

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))
    
    return features, edge_index, labels, non_valid_samples
    
def create_citation_10fold_splits(name, batch_size, selfloops=False, data_dir='data', splits_dir='splits'):
    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir)
    
    features, edge_index, labels, non_valid_samples = full_load_citation(name, f'{data_dir}/{name}')
    
    for i in range(10):
        split = load_geomgcn_split(name, split_nr=i, splits_dir=splits_dir)
        
        for n_i in non_valid_samples:
            if split['train_mask'][n_i]:
                split['train_mask'][n_i] = False
            elif split['val_mask'][n_i]:
                split['val_mask'][n_i] = False
            elif split['test_mask'][n_i]:
                split['test_mask'][n_i] = False
        
        train_mask = np.where(split['train_mask'])[0]
        valid_mask = np.where(split['val_mask'])[0]
        test_mask = np.where(split['test_mask'])[0]
        
        def dataset_gen(subset):
            if subset == 'train':
                split_labels = labels[train_mask]
                mask = train_mask
            elif subset == 'valid':
                split_labels = labels[valid_mask]
                mask = valid_mask
            elif subset == 'test':
                split_labels = labels[test_mask]
                mask = test_mask
            yield {
                'node_features' : features, 
                'edge_index' : edge_index,
                'num_nodes' : len(features),
            }, {
                'mask' : mask, 
                'labels' : split_labels,
            }
        train_gen = partial(dataset_gen, 'train')
        train_gen = partial(batch_graph, train_gen, 1)
        train_gen = partial(to_tf_dataset, train_gen, selfloops=selfloops)
        valid_gen = partial(dataset_gen, 'valid')
        valid_gen = partial(batch_graph, valid_gen, 1)
        valid_gen = partial(to_tf_dataset, valid_gen, selfloops=selfloops)
        test_gen = partial(dataset_gen, 'test')
        test_gen = partial(batch_graph, test_gen, 1)
        test_gen = partial(to_tf_dataset, test_gen, selfloops=selfloops)

        yield { 
            'train' : train_gen,
            'valid' : valid_gen,
            'test' : test_gen  
        }
