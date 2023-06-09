# ================ Config for CORA Dataset ================

cora_model_config = {
    'n_node_features' : 64,
    'walkers_per_node' : 8,
    'walkers_per_node_dist' : 'degree',
    'max_walkers_per_node' : 13,
    'output_activation' : 'softmax',
    'use_state' : True,
    'n_classes' : 7,
    'pooling_level' : 'node',
    'input_dropout' : 0.64,
    'input_l2' : 0.01,
    'identify_walkers_per_node' : False, 

    'add_temporal_features' : True,
    
    'sampling_layer_config' :  { 
        'name' : 'gumbel',
        'temperature' : .6,
    },
    'state_model_config' : {
        'name' : 'res', 
        'units' : 64,
        'dropout' : 0.65,
        'activation' : 'relu',
        'use_batch_norm' : False
    },
    'attention_model_config' : {
        #'name' : 'gat',
        'name' : 'dot',
        'units' : 64,
        'dropout' : 0.18,                           
        'activation' : 'tanh'
    },
    'cell_type' : 'dsgnn',
}
cora_dataset_config = {
    'batch_size' : 1,
    'selfloops' : False
}
cora_train_config = {
    'n_steps' : 2,
    'train_separately' : True,
    'epochs' : 200,
    'loss_fn' : 'sparse_categorical_crossentropy',
    'optimizer_config' : {
        'name': 'Adam',
        'learning_rate': 0.001,
    },
    'metrics' : [
        ('sparse_accuracy', {}),
        ('sparse_auc', { 'num_labels' : 7 })
    ]
}

# =============== Config for WebKB Datasets ===============

webkb_model_config = {
    'n_node_features' : 64,            
    'walkers_per_node' : 3,              
    'walkers_per_node_dist' : 'degree',    
    'max_walkers_per_node' : 14,
    'output_activation' : 'softmax',
    'use_state' : True,
    'n_classes' : 5,
    'stepwise_readout' : True,
    'pooling_level' : 'node',
    'input_dropout' : 0.5,
    'input_l2' : 0.01,
    'identify_walkers_per_node' : False, 

    'add_temporal_features' : False,
    
    'sampling_layer_config' :  { 
        'name' : 'gumbel',
        'temperature' : .6,
        'init_scale' : 1.
    },
    
    'state_model_config' : {
        'name' : 'res', 
        'units' : 64,
        'dropout' : 0.5, 
        'activation' : 'relu',
        'use_batch_norm' : False
    },
    
    'attention_model_config' : {
        'name' : 'gat',      
        'units' : 64,    
        'activation' : 'leaky_relu',  
        'dropout' : 0.5,     
    },
    'cell_type' : 'dsgnn'
}
webkb_dataset_config = {
    'batch_size' : 1,
    'selfloops'  : False
}
webkb_train_config = {
    'n_steps' : 3, # 6,
    'train_separately' : True,
    'log_every' : 100,
    'epochs' : 200,
    'loss_fn' : 'sparse_categorical_crossentropy',
    'optimizer_config' : {
        'name': 'Adam',
        'learning_rate': 0.001,
    },
    'metrics' : [
        ('sparse_accuracy', {}),
        ('sparse_auc', { 'num_labels' : 5 })
    ]
}

# ============== Config for Citation Datasets =============

citation_model_config = {
    'n_node_features' : 64,             
    'walkers_per_node' : 3,                
    'walkers_per_node_dist' : 'degree',   
    'max_walkers_per_node' : 14,
    'output_activation' : 'softmax',
    'use_state' : True,
    'n_classes' : 5,
    'stepwise_readout' : True,
    'pooling_level' : 'node',
    'input_dropout' : 0.5,
    'input_l2' : 0.01,
    'identify_walkers_per_node' : False, 
    'add_temporal_features' : False,  
    
    'sampling_layer_config' :  { 
        'name' : 'gumbel',
        'temperature' : .6,
        'init_scale' : 1.
    },
    
    'state_model_config' : {
        'name' : 'res',                 
        'units' : 64,                   
        'dropout' : 0.5,               
        'activation' : 'relu',       
        'use_batch_norm' : False
    #    'kernel_size' : 10,       
    },
    
    'attention_model_config' : {
        'name' : 'gat',          
        'units' : 64,             
        'activation' : 'leaky_relu',
        'dropout' : 0.5,             
    },
    'cell_type' : 'dsgnn'
}
citation_dataset_config = {
    'batch_size' : 1,
    'selfloops'  : False
}
citation_train_config = {
    'n_steps' : 3, 
    'train_separately' : True, 
    'log_every' : 100,
    'epochs' : 200, 
    'loss_fn' : 'sparse_categorical_crossentropy',
    'optimizer_config' : {
        'name': 'Adam',
        'learning_rate': 0.001, 
    },
    'metrics' : [
        ('sparse_accuracy', {}),
        ('sparse_auc', { 'num_labels' : 5 })
    ]
}