import tensorflow as tf
from functools import partial
import numpy as np
from model import model_utils

class DSGNNCellBase(tf.keras.layers.Layer):
    
    def __init__(self, attention_transform, sampling, input_transform=None, walkers_per_node=None, walkers_per_node_dist='uniform',
        max_walkers_per_node=None, state_model=None, edge_model=None, edge_dim=None, identify_walkers_per_node=True, 
        add_temporal_features=False):
        super(DSGNNCellBase, self).__init__()
        self.attention_transform = attention_transform
        self.sampling = sampling
        self.input_transform = input_transform or (lambda *args: args[0])
        self.walkers_per_node = walkers_per_node
        self.max_walkers_per_node = max_walkers_per_node
        self.walkers_per_node_dist = walkers_per_node_dist
        self.initialize_states = self.init_position_uniformly
        self.aggregate_states = self.aggregate_uniform_states
        self.state_model = state_model
        self.edge_model = edge_model
        self.use_state_model = state_model is not None
        self.use_edge_model = edge_model is not None
        self.identify_walkers_per_node = identify_walkers_per_node
        self.stats = {}
        self.edge_dim = edge_dim
        self.add_temporal_features = add_temporal_features
        self.max_timesteps = 20
        self.secondary_inputs = {}
        if self.edge_dim is not None:
            terminal_edge_embd = tf.random.uniform([1, self.edge_dim], -1., 1.)
            self.terminal_edge_features = tf.Variable(terminal_edge_embd)
        if self.add_temporal_features:
            assert self.max_timesteps is not None
            self.temporal_features = model_utils.positional_encoding(
                self.max_timesteps, self.state_model.units
            )

    def build(self, input_shape):
        _, _, _, _, node_shape = input_shape
        self.node_dim = int(node_shape[-1])
        terminal_node_embd = tf.random.uniform([1, self.node_dim], -1., 1.)
        self.terminal_node_features = tf.Variable(terminal_node_embd)

    def zero_state(self, batch_size):
        if not self.use_state_model:
            return
        if hasattr(self.state_model, 'zero_state'):
            return self.state_model.zero_state(batch_size)
        return tf.zeros([batch_size, self.state_model.units])
    
    def init_positions(self, *args, **kwargs):
        pass
    
    def init_position_uniformly(self, n_nodes=None, indices=None):
        if indices is None:
            indices = tf.range(n_nodes, dtype=tf.int32)
        multiples = tf.concat((
            [self.walkers_per_node], 
            tf.ones((tf.rank(indices) - 1,), dtype=tf.int32)
            ), axis=0)
        return tf.tile(indices, multiples)
    
    def aggregate_uniform_states(self, x, method='sum'):
        x = tf.split(x, self.walkers_per_node, axis=0)
        if method == 'sum':
            return tf.reduce_sum(x, axis=0)
        elif method == 'mean':
            return tf.reduce_mean(x, axis=0)
        elif method == 'max':
            return tf.reduce_max(x, axis=0)
        else:
            raise NotImplemented(f'Aggregation method {method} is not implemented.')
    
    def init_position_by_node_degree(self, n_nodes=None, indices=None, node_degrees=None, node_degree_transform=None):
        if node_degree_transform is not None:
            node_degrees = node_degree_transform(node_degrees)
        if indices is None:
            indices = tf.range(n_nodes, dtype=tf.int32)
        n_nodes = tf.shape(node_degrees)[0]
        return tf.repeat(indices, tf.maximum(node_degrees, 1))
        
    def aggregate_states_by_node_degree(self, x, initial_state, method='sum'):
        if method == 'sum':
            return tf.math.segment_sum(x, initial_state)
        elif method == 'mean':
            return tf.math.segment_mean(x, initial_state)
        elif method == 'max':
            return tf.math.segment_max(x, initial_state)
        else:
            raise NotImplemented(f'Aggregation method {method} is not implemented.')

    def _walker_relative_node_id(self, state):
        '''
        This function turns node ids into relative node ids
        that reflect the first step when this node was visited by a sampler.
        Example: [10, 13, 18, 13, 32, 10] -> [0, 1, 2, 1, 3, 0]
        '''
        was_visited = self.seen_nodes == state[:, tf.newaxis]
        update_visited = tf.concat((self.seen_nodes, state[:, tf.newaxis]), axis=-1)

        if tf.shape(self.seen_nodes)[-1] == 0:
            # first iteration (no nodes seen so far)
            walkers_revisiting = tf.reduce_any(was_visited, axis=-1, keepdims=True)
            update_visited = tf.where(walkers_revisiting, -1, state[:, tf.newaxis])
            # update seen nodes
            self.seen_nodes = tf.concat((self.seen_nodes, update_visited), axis=-1)
            # update next unseen node index
            self.unique_visited_nodes += 1
            return self.unique_visited_nodes - 1

        walkers_revisiting = tf.reduce_any(was_visited, axis=-1, keepdims=True)
        visited_node = tf.concat((was_visited, tf.logical_not(walkers_revisiting)), axis=-1)
        revisited_nodes_idx = tf.cast(tf.where(visited_node)[:, 1:2], tf.int32)
        rel_state = tf.where(walkers_revisiting, revisited_nodes_idx, self.unique_visited_nodes)

        update_visited = tf.where(walkers_revisiting, -1, state[:, tf.newaxis])
        # update seen nodes
        self.seen_nodes = tf.concat((self.seen_nodes, update_visited), axis=-1)
        # update next unseen node index
        self.unique_visited_nodes += tf.cast(tf.logical_not(walkers_revisiting), tf.int32)
        return rel_state
    
    def state_transition_index(self, states, edge_list, node_degrees, edge_features=None):
        '''
        Return a transition index, contains indices of states and neighbors:
        shape : [n_transitions], [n_transitions]
        '''
        n_nodes = tf.shape(node_degrees)[0]
        num_edges = tf.reduce_sum(node_degrees)
        n_edges = tf.gather(node_degrees, states)

        terminals = n_edges == 0
        state_neighbors = tf.gather(edge_list, states)
        receivers = state_neighbors.flat_values

        if tf.reduce_any(terminals):
            # account for the edge case of a node without outgoing edge
            # a virtual node is added for this case
            n_edges = tf.maximum(n_edges, 1)
            terminals_rep = tf.repeat(terminals, n_edges)
            rec_dims = tf.size(receivers[0])
            rec_shape = [tf.shape(terminals_rep)[0], *receivers[0].shape]

            if rec_dims == 1:
                # edge list contains ONLY node indices
                virtual_rec = tf.cast(n_nodes, receivers.dtype)
            else:
                # edge list contains node AND edge indices
                n_nodes = tf.cast(n_nodes, tf.int64)
                num_edges = tf.cast(num_edges, tf.int64)
                virtual_rec = tf.cast(tf.stack([n_nodes, num_edges])[tf.newaxis], receivers.dtype)
            #else:
            #    raise ValueError('An edge_list with more than two dimensions is not valid. '
            #                     'Edgelists may contain receiver node index and edge index only, '
            #                     'i.e. a ragged shape of [n_nodes, <ragged>] or [n_nodes, <ragged>, 2].')

            receivers = tf.tensor_scatter_nd_update(
                tf.ones(rec_shape, dtype=receivers.dtype) * virtual_rec,
                tf.where(tf.logical_not(terminals_rep)),
                receivers
            )

        # get the number of outgoing edges from the current nodes
        walker_idx = tf.range(tf.shape(states)[0])
        senders = tf.repeat(walker_idx, n_edges)
        neighbor_edge_idx = None
        
        # get the neighboring edge indices if edge features are available
        if edge_features is not None:
            receivers, neighbor_edge_idx = receivers[:,0], receivers[:,1]
            neighbor_edge_idx.set_shape((None,))
        receivers.set_shape((None,))

        return senders, receivers, n_edges, neighbor_edge_idx, terminals

    def reindex_terminals(self, next_state_idx, terminals, prior_state_idx):
        if tf.reduce_any(terminals):
            return tf.where(terminals, prior_state_idx, next_state_idx)
        return next_state_idx

    def concat_terminal_features(self, node_features, edge_features=None):
        '''
        In case there are any unconnected nodes, a learned virtual node is added.
        Same for the according edge, in case there are edge features.
        '''
        node_features = tf.concat((node_features, self.terminal_node_features), axis=0)
        if edge_features is not None:
            edge_features = tf.concat((edge_features, self.terminal_edge_features), axis=0)
        return node_features, edge_features

    def remove_terminal_features(self, node_features, edge_features=None):
        node_features = node_features[:-1] # remove the terminal virtual node
        if edge_features is not None:
            edge_features = edge_features[:-1] # remove the terminal virtual edge
        return node_features, edge_features

    def update_walker_state(self, state_features, walker_hidden, training=False):
        # update walker state
        walker_states, walker_hidden = self.state_model(state_features, walker_hidden, training=training)
        return walker_states, walker_hidden

    def message_passing(self, node_features, edge_index, edge_features=None):
        n_nodes = tf.shape(node_features)[0]
        senders, receivers = tf.unstack(edge_index, axis=-1)
        senders = tf.gather(node_features, senders)
        aggregation = tf.math.unsorted_segment_mean(senders, receivers, n_nodes)
        return aggregation

    def edge_transform(self, x, edge_features=None):
        if edge_features is not None:
            x = tf.concat((x, edge_features), axis=-1)
        if not self.use_edge_model:
            return x
        return self.edge_model(x)
    
    def is_first_iteration(self, inputs):
        return not tf.nest.is_nested(inputs[0])
    
    def _on_first_iteration(self, inputs, walker_hidden, training=False):
        '''overwrite this'''
        states, state_features, edge_list, node_degrees, node_features = inputs
        return (states, state_features, edge_list, node_degrees, node_features), state_features
    
    def pre_step_traversal(self, inputs, edge_features):
        '''overwrite this'''
        self.step_aux_loss = 0
        states, state_features, edge_list, node_degrees, node_features = inputs
        return (
            (states, state_features, edge_list, node_degrees, node_features), 
            edge_features
        )
        
    def post_step_traversal(self, inputs, prior_state, walker_states, walker_hidden, terminals, edge_features=None):
        '''overwrite this'''
        next_state_idx, state_features, edge_list, node_degrees, node_features = inputs
        next_state_idx = self.reindex_terminals(next_state_idx, terminals, prior_state)
        node_features, edge_features = self.remove_terminal_features(node_features, edge_features)
        return (next_state_idx, state_features, edge_list, node_degrees, node_features), walker_states, walker_hidden

    
    def on_first_iteration(self, inputs, walker_hidden, training=False):
        '''
        Called upon the first step of a graph traversal.
        '''       
        states, state_features, edge_list, node_degrees, node_features = inputs
        
        if self.walkers_per_node_dist == 'degree':
            self.initialize_states = partial(self.init_position_by_node_degree, node_degrees=node_degrees)
        if self.walkers_per_node_dist == 'double_degree':
            self.initialize_states = partial(self.init_position_by_node_degree, node_degrees=node_degrees * 2)
            
        # initialize the positions of the walkers
        if states is None:
            states = self.initialize_states(tf.shape(node_features)[0])
        
        # initialize the hidden state
        if walker_hidden is None:
            walker_hidden = self.zero_state(tf.shape(states)[0])
            
        # initialize the first state
        if state_features is None:
            state_features = tf.gather(node_features, states)
            #state_features = tf.zeros([tf.shape(states)[0], tf.shape(node_features)[-1]])
        
        if self.walkers_per_node_dist == 'degree' or self.walkers_per_node_dist == 'double_degree':
            self.aggregate_states = partial(self.aggregate_states_by_node_degree, initial_state=states) 
        
        inputs = (states, state_features, edge_list, node_degrees, node_features)
        
        self.stats = {
            'edge_probs' : [],
            'sampling_entropy' : [],
        }
        
        return self._on_first_iteration(inputs, walker_hidden, training=training)
         
    def _pre_step_traversal(self, inputs, edge_features):
        '''
        Pre Processing before each step
        '''
        states, state_features, edge_list, node_degrees, node_features = inputs
        
        # maybe add temporal encoding
        if self.add_temporal_features:
            relative_node_idx = self._walker_relative_node_id(states)
            state_features += tf.gather(self.temporal_features, relative_node_idx)
        
        inputs, edge_features = self.pre_step_traversal(
            (states, state_features, edge_list, node_degrees, node_features), 
            edge_features
        )
        
        return inputs, edge_features
    
    def step(self, inputs, soft_select=True, hard_sampling=True, **kwargs):
        '''
        Main logic to perform a step on the graph.
        '''
        pass
    
    #def post_step_traversal(self, inputs, walker_states, walker_hidden):
    #    '''
    #    Post processing after each step.
    #    '''
    #    pass
        
    def call(self, inputs, edge_features=None, walker_hidden=None, hard_sampling=True, soft_select=True, training=False, **kwargs):
        '''
        Return feature vectors of the nodes given in states:
        Parameters
        -----------
        inputs : tuple (states, state_features, edge_list, node_degrees, node_features)
            expected shapes: 
                * states         - [n_walkers, 1]
                * state_features - [n_walkers, state_dims]
                * edge_list      
                    * with edge features - Ragged([n_nodes, node_neighbors(ragged), 2])
                        When edge features are passed the edge index has to be passed
                    * no edge features   - Ragged([n_nodes, node_neighbors(ragged)])
                        When no edge features are passed the last dimension can be omitted
                * node_degrees   - [n_nodes, 1]
                * node_features  - [n_nodes, node_dims]
        edge_features : tensor with shape: [n_edges, edge_dims]
            Edge feature vectors.
        walker_hidden : tensor with shape: [n_walkers, state_dims]
            Hidden State of the state model.
        hard_sampling : bool
            Whether to return the next input to the state model as the soft
            weighted sum of the neighborhood or a single selected neighbor.
        soft_select : bool
            Whether to perform soft neighbor sampling or hard neighbor selection. 
        '''
        # self.stats = {}
        
        if self.is_first_iteration(inputs):
            inputs, walker_hidden = self.on_first_iteration(inputs, walker_hidden, training=training)
            self.current_step = 0
        else:
            self.current_step += 1
        
        prior_state = inputs[0]#[0]
        pre_step, edge_features = self._pre_step_traversal(inputs, edge_features)
        post_step, walker_states, walker_hidden, terminals = self.step(
            pre_step, 
            edge_features=edge_features,  
            walker_hidden=walker_hidden, 
            soft_select=soft_select,
            hard_sampling=hard_sampling,
            training=training, 
            **kwargs
        )
        post_step, walker_states, walker_hidden = self.post_step_traversal(
            post_step, prior_state, walker_states, walker_hidden, terminals, edge_features=edge_features
        )
        
        next_state_idx, next_state, _, _, node_features = post_step
        #walker_states = (walker_states, self.walker_states_updated)
        
        return next_state_idx, next_state, node_features, walker_states, walker_hidden