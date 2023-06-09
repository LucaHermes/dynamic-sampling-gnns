import pandas as pd
from tqdm.autonotebook import tqdm 
from pathlib import Path
import numpy as np
from benchmark_transductive import read_json, get_best_n_epochs, get_experiment, find_all_splits

DATASET_NAMES = ['texas', 'wisconsin', 'film', 'cornell', 'citeseer-10', 'pubmed-10', 'cora-10']
exp_names = ['ACGNNII-Texas', 'DSGNN-Wisconsin', 'DSGNN-Actor', 
             'DSGNN-Cornell', 'DSGNN-Citeseer-10', 'DSGNN-Pubmed-10', 'DSGNN-Cora-10']

def generate_table(exp_names, dataset_names=None, index_item=None):
    if dataset_names  is None:
        dataset_names = DATASET_NAMES
        
    table = {}

    # collect benchmarking results
    for dsn, expn in tqdm(zip(dataset_names, exp_names), desc='Converting Results to Latex table ...'):
        exp_path = get_experiment(dsn, name=expn)
        splits = find_all_splits(exp_path)
        best_val_epoch, avg_metric, std_metric = get_best_n_epochs(exp_path, 'validation/sparse_accuracy')
        output_path = Path('results') / exp_path.name / f'epoch_{best_val_epoch}'
        
        metrics = []
        for _, folder_name in splits:
            for i in range(10):
                metrics_file = output_path / f'{i}_{folder_name}-metrics.json'
                assert metrics_file.is_file()
                metrics.append(read_json(metrics_file)['validation/sparse_accuracy'])
            
        table[dsn] = metrics
    
    # convert to latex table
    df = pd.concat((
        pd.DataFrame(table, columns=dataset_names).mean(0) * 100,
        pd.DataFrame(table, columns=dataset_names).std(0) * 100,
    ), axis=1).round(2).apply(lambda x: f'{x[0]} (± {x[1]})', axis=1)
    df = df.to_frame().T
    if index_item is not None:
        df.index = [index_item]
    return df
    
    
    