import model.base_cell
import tensorflow as tf
import utils.graph_utils as graph_utils
from tensorflow_probability import distributions as tfd

class SimpleDSGNNCell(model.base_cell.DSGNNCellBase):
    
    def step(self, inputs, edge_features=None, walker_hidden=None, soft_select=True, hard_sampling=True, training=False, **kwargs):
        '''
        Main logic to perform a step on the graph.
        '''
        states, state_features, edge_list, node_degrees, node_features = inputs
        n_walkers = tf.shape(states)[0]

        # ==================== Get required indices and prepare for sampling step ==================== 
        # build an edge index from current states. We get walker_ids and the indices of neighboring 
        # nodes just like an edge-index would give sender and receiver node idx.
        walker_ids, neigh_nodes, n_edges, neighbor_edge_idx, terminals = self.state_transition_index(
            states, edge_list, node_degrees, edge_features)
        
        # pass node features through the state model
        walker_states, walker_hidden = self.update_walker_state(
            state_features,
            walker_hidden,
            training=training)
        
        # in case there is a dead-end, we add a special feature `terminal_node_features`
        # because in the current implementation, a walker has to select an edge every step.
        # The terminal_node_feature will be passed to the walker instead of a node, if no edge
        # exists from the current walker position.
        node_features = tf.concat((node_features, self.terminal_node_features), axis=0)
        
        # collect neighbor nodes
        neighbor_feats = tf.gather(node_features, neigh_nodes)
        # get features of the node where the walkers currently are
        state_features = tf.gather(walker_hidden, walker_ids)
        
        # compute logits for edge selection
        edge_logits = self.attention_transform(state_features, neighbor_feats, training=training)

        # sample from edge_logits - a categorical distribution across node neighbors
        next_state, s_hot, probs = self.sampling(edge_logits, walker_ids, n_walkers, soft_select=soft_select)
        next_state_idx = tf.boolean_mask(neigh_nodes, s_hot)

        # mask the neighbor features to select only the 
        next_state = neighbor_feats * next_state[:, tf.newaxis]
        hard_next_state = tf.boolean_mask(neighbor_feats, s_hot)
        soft_next_state = tf.math.segment_sum(next_state, walker_ids)

        probs = graph_utils.segment_softmax(edge_logits, walker_ids, n_walkers)
        self.stats['edge_probs'].append(probs[:1000])
        self.stats['sampling_entropy'].append(
            graph_utils.segment_entropy(probs, n_edges, walker_ids, n_walkers)[:1000]
        )
        
        if hard_sampling:
            next_state = hard_next_state
        else:
            next_state = soft_next_state

        return (next_state_idx, next_state, edge_list, node_degrees, node_features), \
                walker_states, walker_hidden, terminals

class DSGNNCell(model.base_cell.DSGNNCellBase):
    '''
    The implementation of a DSGNN cell used to generate the evaluation results in
    the paper "Graph Learning by Dynamic Sampling".
    '''

    def build(self, input_shape):
        super(DSGNNCell, self).build(input_shape)
        if self.identify_walkers_per_node:
            self.walker_identifiers = [
                tf.Variable(tf.random.uniform([self.state_model.units], -1., 1.), trainable=True) #, name=scope), # mean
                for a in range(self.max_walkers_per_node)
            ]
    
    def _on_first_iteration(self, inputs, walker_hidden, training=False):
        '''
        Called before the first step of a graph traversal. Here, the walker states
        are initialized. Also lists are initialized that track: 
         * The ids of nodes seen by each walker (self.seen_nodes)
         * The edges seen by each walker (specified by sender and receiver node id) (self.seen_edges)
         * The currently traversed edge is set to None, as no edge has been traversed so far.
         * The number of unique nodes seen by each walker. (self.unique_visited_nodes)
        '''
        states, state_features, edge_list, node_degrees, node_features = inputs
        n_nodes = tf.shape(node_features)[0]
        
        # the walker keys are sampled from a categorical distribution 'centered' around zero
        walker_keys = None

        if self.identify_walkers_per_node:
            walker_keys = tf.concat([
                tf.repeat(i[tf.newaxis], n_nodes, axis=0)
                for i in self.walker_identifiers
            ], axis=0)
            walker_hidden = walker_keys
            
        n_walkers = tf.shape(states)[0]
        self.seen_nodes = tf.zeros([n_walkers, 0], dtype=tf.int32)
        self.seen_edges = tf.zeros([n_walkers, 0, 2], dtype=tf.int32)
        self.current_edge = None
        self.unique_visited_nodes = tf.zeros([n_walkers, 1], dtype=tf.int32)
        self.walker_states = tf.zeros_like(state_features)

        node_features = (node_features, None) # add one for terminal node
        
        return (states, state_features, edge_list, node_degrees, node_features), walker_hidden
    
    def pre_step_traversal(self, inputs, edge_features):
        '''
        Pre Processing before each step
        '''
        self.step_aux_loss = 0
        
        states, state_features, edge_list, node_degrees, node_features = inputs
        # concat additional node/edge feature to account for unconnected nodes
        node_features, edge_features = self.concat_terminal_features(node_features, edge_features)
        return (
            (states, state_features, edge_list, node_degrees, node_features), 
            edge_features
        )
    
    def step(self, inputs, edge_features=None, walker_hidden=None, soft_select=True, hard_sampling=True, training=False, logit_bias=None, **kwargs):
        '''
        Main logic to perform a step on the graph.
        '''
        states, state_features, edge_list, node_degrees, node_features = inputs

        # n_nodes is the number of nodes without the terminal node
        n_nodes = tf.shape(node_degrees)[0]
        n_walkers = tf.shape(states)[0]

        # ==================== Get required indices and prepare for sampling step ==================== 
        # build an edge index from current states
        senders, receivers, n_edges, neighbor_edge_idx, terminals = self.state_transition_index(
            states, edge_list, node_degrees, edge_features)
            
        _walker_states, _walker_hidden = self.update_walker_state(
            state_features,
            walker_hidden,
            training=training
        )
        
        # update states only 
        if self.current_edge is not None:
            # [n_walkers, n_visited_edges, 2]
            # [n_walkers, n_visited_edges]
            walker_revisiting_edge = tf.reduce_all(self.seen_edges == self.current_edge, axis=-1)
            # [n_walkers, 1]
            walker_revisiting_edge = tf.reduce_any(walker_revisiting_edge, axis=-1, keepdims=True)
            self.walker_states = tf.where(walker_revisiting_edge, self.walker_states, _walker_states)
            walker_hidden = tf.where(walker_revisiting_edge, walker_hidden, _walker_hidden)
            walker_states = self.walker_states
            walker_states.set_shape((None, _walker_states.shape[-1]))
        else:
            self.walker_states = _walker_states
            walker_states = self.walker_states
            walker_hidden = _walker_hidden

        # Update the node features
        if self.current_edge is not None:
            update_at = tf.cast(tf.logical_not(walker_revisiting_edge), tf.float32)
            node_update = walker_hidden * update_at
            node_update = self.aggregate_states(node_update, method='sum')
            n_updates = self.aggregate_states(update_at, method='sum')
            node_features = node_update / tf.maximum(n_updates, 1)
        else:
            # in first iteration, all walkers can update
            node_features = self.aggregate_states(walker_hidden, method='mean')
            
        node_features = tf.concat((node_features, self.terminal_node_features), axis=0)
        
        if self.secondary_inputs.get('node_neighborhood') is None:
            neighbor_feats = tf.gather(node_features, receivers)
        else:
            neighbor_feats = self.secondary_inputs['node_neighborhood']
            receivers = self.secondary_inputs['receivers']
            senders = self.secondary_inputs['senders']
            n_edges = tf.shape(senders)[0]
            

        if edge_features is not None:
            neighbor_edges = tf.gather(edge_features, neighbor_edge_idx)
            neighbor_feats = self.edge_transform(neighbor_feats, neighbor_edges)

        # get features of the node where the walkers currently are
        state_features = tf.gather(walker_states, senders)
        
        # shape: [n_state_transitions, 1]
        # shape: [batch, n_walkers, n_nodes, 1]
        # compute logits for edge selection
        
        edge_logits = self.attention_transform(state_features, neighbor_feats, training=training)
        edge_logits = tf.squeeze(edge_logits, axis=-1)

        if logit_bias is not None:
            logit_bias = tf.gather(logit_bias, senders)
            edge_logits += logit_bias.flat_values

        # sample from edge_logits - a categorical distribution across node neighbors
        next_state, s_hot, probs = self.sampling(edge_logits, senders, n_walkers, soft_select=soft_select)
        s_hot.set_shape((None, ))
        next_state_idx = tf.boolean_mask(receivers, s_hot)

        next_state = neighbor_feats * next_state[:, tf.newaxis]
        hard_next_state = tf.boolean_mask(neighbor_feats, s_hot)
        soft_next_state = tf.math.segment_sum(next_state, senders)


        probs = graph_utils.segment_softmax(edge_logits, senders, n_walkers)
        self.stats['edge_probs'].append(probs[:1000])
        self.stats['sampling_entropy'].append(
            graph_utils.segment_entropy(probs, n_edges, senders, n_walkers)[:1000]
        )
        
        if hard_sampling:
            next_state = hard_next_state
        else:
            next_state = soft_next_state

        return (next_state_idx, next_state, edge_list, node_degrees, node_features), \
                walker_states, walker_hidden, terminals
    
    def post_step_traversal(self, inputs, prior_state, walker_states, walker_hidden, terminals, edge_features=None):
        '''
        Post processing after each step.
        '''
        next_state_idx, state_features, edge_list, node_degrees, node_features = inputs
        
        next_edge = tf.stack((prior_state, next_state_idx), axis=-1)

        # update seen edges with the current_edge
        if self.current_edge is not None:
            self.seen_edges = tf.concat((self.seen_edges, self.current_edge), axis=-2)

        # put the next traversed edge as current_edge
        self.current_edge = next_edge[:,tf.newaxis]
        
        next_state_idx = self.reindex_terminals(next_state_idx, terminals, prior_state)
        node_features, edge_features = self.remove_terminal_features(node_features, edge_features)
        
    
        return (next_state_idx, state_features, edge_list, node_degrees, node_features), walker_states, walker_hidden
