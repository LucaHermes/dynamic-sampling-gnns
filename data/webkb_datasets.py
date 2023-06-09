from pathlib import Path
import numpy as np
from functools import partial
from data.dataset import batch_graph, to_tf_dataset, load_geomgcn_split, download_file

N_CLASSES = 5
NAMES = ['cornell', 'texas', 'film', 'wisconsin', 'squirrel', 'chameleon']

WEBKB_FILES = { 
    'edges' : 'out1_graph_edges.txt',
    'node_features' : 'out1_node_feature_label.txt'
}

DATA_URL = (
    'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data'
)
WIKI_DATA_URL = (
    'https://raw.githubusercontent.com/Yujun-Yan/Heterophily_and_oversmoothing/master/new_data'
)

def create_webkb_10fold_splits(name, batch_size, selfloops=False, data_dir='data', splits_dir='splits'):
    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir)
    split_loader = load_webkb_dataset(name, data_dir=data_dir, splits_dir=splits_dir)
    
    for data, split in split_loader:
        train_mask = np.where(split['train_mask'])[0]
        valid_mask = np.where(split['val_mask'])[0]
        test_mask = np.where(split['test_mask'])[0]
        node_features = data['node_features']
        
        def dataset_gen(subset):
            if subset == 'train':
                labels = data['labels'][train_mask]
                mask = train_mask
            elif subset == 'valid':
                labels = data['labels'][valid_mask]
                mask = valid_mask
            elif subset == 'test':
                labels = data['labels'][test_mask]
                mask = test_mask
                
            yield {
                'node_features' : node_features, 
                'edge_index' : data['edge_index'],
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
        test_gen = partial(dataset_gen, 'test')
        test_gen = partial(batch_graph, test_gen, 1)
        test_gen = partial(to_tf_dataset, test_gen, selfloops=selfloops)

        yield { 
            'train' : train_gen,
            'valid' : valid_gen ,
            'test' : test_gen 
        }
        
def process(name, key, file):
    if key == 'edges':
        return { 'edge_index' : np.loadtxt(file, skiprows=1, dtype=np.int32) }
    
    elif key == 'node_features':
        node_features = []
        labels = []
        
        with open(file) as f:
            for line in f.readlines()[1:]:
                n_id, feats, label = line.split('\t')
                feats = np.asarray(feats.split(','), dtype=np.float32)
                
                if name == 'film':
                    # film uses sparse features -> convert to dense
                    feats = np.eye(932)[feats.astype(int)].sum(0)
                    
                node_features.append((int(n_id), feats))
                labels.append((int(n_id), int(label)))
                
        return {
            'node_features' : np.stack([ f for _, f in sorted(node_features) ]),
            'labels' : np.stack([ l for _, l in sorted(labels) ])[:, np.newaxis]
        }
        
def load_webkb_dataset(name, data_dir, splits_dir):
    assert name in NAMES
    
    if name in ['squirrel', 'chameleon']:
        data_url = WIKI_DATA_URL
    else:
        data_url = DATA_URL
    
    # download the edge and node files
    for k, f in WEBKB_FILES.items():
        out_file = data_dir / name / f
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not out_file.exists():
            print(f'Downloading {name} dataset, save to {out_file}.')
            file_url = data_url + f'/{name}/{f}'
            download_file(file_url, output_file=out_file)
    
    dataset = {}
    for k, f in WEBKB_FILES.items():
        dataset.update(process(name, k, str(data_dir / name / f)))
    
    # download the 10 different train-val-test splits
    for i in range(10):
        split = load_geomgcn_split(name, i, splits_dir)
        yield dataset, split
