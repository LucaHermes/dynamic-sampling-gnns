import tensorflow as tf
from functools import lru_cache
import utils.experiment_utils as experiment_utils
from tqdm.auto import tqdm
import os

class DSGNN(tf.keras.layers.Layer):
    
    def __init__(self, cell, walker_state_size, walkers_per_node=1, walkers_reduce_fn=None, use_state=False,
                 input_transform=None, input_edge_transform=None, embedding_layer=None, state_init_fn=None, 
                 output_transform=None, stepwise_readout=True, pooling_level='node'):
        super(DSGNN, self).__init__()
        self.cell = cell
        self.walker_state_size = walker_state_size
        self.use_state = use_state
        self.input_transform = input_transform
        self.input_edge_transform = input_edge_transform
        self.embedding_layer = embedding_layer
        self.output_transform = output_transform
        
        if pooling_level == 'graph':
            self.readout = self.readout_graph_prediction
        elif pooling_level == 'node':
            # simply return node_predictions
            self.readout = lambda x: self.mean_max_walkers_reduce(x[0])
        elif pooling_level is None:
            # simply return node_predictions
            self.readout = lambda x: x[0]
        else:
            raise NotImplementedError(
                f'A pooling method for level `{pooling_level}` is not implemented. '
                'Provide a pooling_level of either `graph` or `node`.'
            )
        if self.embedding_layer is None:
            self.embedding_layer = lambda x : x
        if self.output_transform is None:
            self.output_transform = lambda x : x
        self.stepwise_readout = stepwise_readout
        self.train_epoch = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.validation_outputs = False
        self.validation_outputs_path = None
        
    def save(self, ckpt_path, optimizer=None):
        ckpt = tf.train.Checkpoint(step=self.global_step, optimizer=optimizer, model=self)
        manager = tf.train.CheckpointManager(ckpt, directory=ckpt_path, max_to_keep=None)
        return manager.save()

    def load(self, ckpt_path, optimizer=None):
        
        ckpt = tf.train.Checkpoint(step=self.global_step, optimizer=optimizer, model=self)
        manager = tf.train.CheckpointManager(ckpt, directory=ckpt_path, max_to_keep=None)
        if manager.latest_checkpoint:
            tf.get_logger().info(f'Restoring model from checkpoint {manager.latest_checkpoint}')
            return manager.restore_or_initialize()
        if int(os.environ.get('TF_CPP_MIN_LOG_LEVEL', 0)) < 1:
            tf.get_logger().info('No checkpoint found in directory "%s"' % ckpt_path)

    @property
    @lru_cache(maxsize=1)
    def trainable_variables_without_neighborhood_attention(self):
        return [ v for v in self.trainable_variables if 'neighborhood_attention' not in v.name ]

    @property
    @lru_cache(maxsize=1)
    def trainable_variables_neighborhood_attention(self):
        return [ v for v in self.trainable_variables if 'neighborhood_attention' in v.name ]

    def mean_max_walkers_reduce(self, walker_state):
        # walkers_of_node = tf.stack(tf.split(walker_state, self.walkers_per_node, axis=0))
        return tf.concat((
            self.cell.aggregate_states(walker_state, method='mean'),
            self.cell.aggregate_states(walker_state, method='max'),
            #tf.reduce_mean(walkers_of_node, axis=0),
            #tf.reduce_max(walkers_of_node, axis=0),
        ), axis=-1)

    def pre_propagation(self, states, node_features, edge_features=None, walker_hidden=None, training=False):
        if self.input_transform is not None:
            node_features = self.input_transform(node_features, training=training)

        if edge_features is not None:
            edge_features = self.input_edge_transform(edge_features, training=training)
            
        return states, None, node_features, edge_features, walker_hidden

    def readout_graph_prediction(self, x):
        walker_state, graph_mask = x
        n_graphs = tf.reduce_max(graph_mask) + 1
        return tf.concat((
            tf.math.unsorted_segment_mean(walker_state, graph_mask, n_graphs),
            tf.math.unsorted_segment_max(walker_state, graph_mask, n_graphs)
        ), axis=-1)

    
    def extend_node_to_walker(self, x):
        '''
        Takes a vector with shape [n_nodes, ..] and
        maps it to the walkers [n_walkers, ...] by replicating x.
        '''
        return self.cell.initialize_states(indices=x)
    
    def init_traversal(self, inputs, walker_hidden=None, training=False):
        states, node_features, edge_list, node_degrees, edge_features = inputs
        states, states_features, node_features, edge_features, walker_hidden = self.pre_propagation(
            states, node_features, 
            edge_features=edge_features, 
            walker_hidden=walker_hidden,
            training=training)
        return {
            'inputs' : (states, 
                node_features,
                edge_list,
                node_degrees,
                edge_features),
            'states_features' : states_features,
            'walker_hidden' : walker_hidden,
            'step' : 0,
        }
    
    def call_stepwise(self, inputs, states_features, step, last_step_embedding=None, walker_hidden=None, 
                      secondary_inputs=None, graph_mask=None, post_propagation_processing=True, training=False, **kwargs):
        states, node_features, edge_list, node_degrees, edge_features = inputs
        self.cell.secondary_inputs = secondary_inputs
        
        states, states_features, node_features, walker_state, walker_hidden = self.cell(
            (states, states_features, edge_list, node_degrees, node_features), 
            walker_hidden=walker_hidden, edge_features=edge_features, training=training, **kwargs)
        
        step_prediction = self.embedding_layer(walker_state)
        graph_mask = self.extend_node_to_walker(graph_mask)

        if self.stepwise_readout:
            if last_step_embedding is not None:
                last_step_embedding += step_prediction
            else:
                last_step_embedding = step_prediction
            step_prediction = last_step_embedding / tf.cast(tf.maximum(step, 1), tf.float32)
        
        if post_propagation_processing:
            step_prediction = self.readout((step_prediction, graph_mask))
            step_prediction = self.output_transform(step_prediction)
        
        return {
            'inputs' : (states, 
                node_features,
                edge_list,
                node_degrees,
                edge_features),
            'states_features' : states_features, 
            'last_step_embedding' : last_step_embedding,
            'secondary_inputs' : secondary_inputs,
            'walker_hidden' : walker_hidden,
            'step' : step + 1,
        }
        
    def call_generator(self, inputs, n_steps=3, walker_hidden=None, graph_mask=None, 
             post_propagation_processing=True, training=False, **kwargs):
        # passing inputs as tuple allows to get all input shapes in `build` methods
        states, node_features, edge_list, node_degrees, edge_features = inputs
        # in the first execution, the cell only decides for a neighbor, but only 
        # goes there in the second iteration. Therefore n_steps has to be increased 
        # by one to truly execute n_steps.
        n_steps += 1
        # initialize auxiliary variables
        prediction, step_prediction = 0, 0
        
        # execute input preprocessing before iterating the steps
        states, states_features, node_features, edge_features, walker_hidden = self.pre_propagation(
            states, node_features, 
            edge_features=edge_features, 
            walker_hidden=walker_hidden,
            training=training)

        graph_mask = self.extend_node_to_walker(graph_mask)

        # traverse n_steps on the graph
        for step in range(n_steps):
            # forward pass through the cell
            states, states_features, node_features, walker_state, walker_hidden = self.cell(
                (states, states_features, edge_list, node_degrees, node_features), 
                walker_hidden=walker_hidden, edge_features=edge_features, training=training, **kwargs)
            
            step_prediction = self.embedding_layer(walker_state)

            if self.stepwise_readout:
                prediction += step_prediction
                step_prediction = prediction / tf.cast(tf.maximum(step, 1), tf.float32)
            
            if post_propagation_processing:
                step_prediction = self.readout((step_prediction, graph_mask))
                step_prediction = self.output_transform(step_prediction)
            
            yield states, node_features, step_prediction

    def call(self, inputs, n_steps=3, walker_hidden=None, graph_mask=None, output_all_states=False, 
             post_propagation_processing=True, training=False, **kwargs):
        # passing inputs as tuple allows to get all input shapes in `build` methods
        states, node_features, edge_list, node_degrees, edge_features = inputs
        # in the first execution, the cell only decides for a neighbor, but only 
        # goes there in the second iteration. Therefore n_steps has to be increased 
        # by one to truly execute n_steps.
        n_steps += 1
        # initialize auxiliary variables
        prediction, step_prediction = 0, 0
        self.call_stats = {}
        
        # execute input preprocessing before iterating the steps
        states, states_features, node_features, edge_features, walker_hidden = self.pre_propagation(
            states, node_features, 
            edge_features=edge_features, 
            walker_hidden=walker_hidden,
            training=training)

        if output_all_states:
            walker_output_states = [self.cell.initialize_states(node_features.shape[0])]

        # traverse n_steps on the graph
        for step in range(n_steps):
            # forward pass through the cell
            states, states_features, node_features, walker_state, walker_hidden = self.cell(
                (states, states_features, edge_list, node_degrees, node_features), 
                walker_hidden=walker_hidden, edge_features=edge_features, training=training, **kwargs)
            
            if output_all_states:
                s = states
                if tf.nest.is_nested(states):
                    s, _ = states
                if output_all_states:
                    walker_output_states.append(s)
            
            step_prediction = self.embedding_layer(walker_state)

            if self.stepwise_readout:
                prediction += step_prediction
            else:
                if s == (n_steps - 1):
                    prediction = step_prediction

        if self.stepwise_readout:
            prediction = prediction / n_steps
            
        if post_propagation_processing:
            graph_mask = self.extend_node_to_walker(graph_mask)
            prediction = self.readout((prediction, graph_mask))
            prediction = self.output_transform(prediction)
        
        if output_all_states:
            return states, node_features, prediction, walker_output_states
        return states, node_features, prediction

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, node_features, edge_list, node_degrees, edge_features=None, 
                   graph_mask=None, mask=None, labels=None, n_steps=1, walker_hidden=None, 
                   step_sampling=True, step_non_sampling=True, soft_select=True, hard_sampling=True):
            
        with tf.GradientTape() as tape:
            walker_pos, x, preds = self((
                None,
                node_features,
                edge_list,
                node_degrees,
                edge_features),
                n_steps=n_steps,
                walker_hidden=walker_hidden,
                hard_sampling=hard_sampling,
                soft_select=soft_select,
                graph_mask=graph_mask,
                post_propagation_processing=True,
                training=True)
            
            if mask is not None:
                train_preds = tf.gather(preds, mask)
            else:
                train_preds = preds

            loss = self.loss_fn(labels, train_preds)
            loss = tf.reduce_mean(loss)
            loss += tf.reduce_sum(self.losses)

        var_list = []

        if step_sampling:
            # steps all parts of the model that are 
            # involved in the neighborhood selection
            var_list.extend(self.trainable_variables_neighborhood_attention)
        if step_non_sampling:
            # steps all parts of the model that are not 
            # involved in the neighborhood selection
            var_list.extend(self.trainable_variables_without_neighborhood_attention)

        grad = tape.gradient(loss, var_list)
        grad, grad_norm = tf.clip_by_global_norm(grad, 1.)
        self.optimizer.apply_gradients(zip(grad, var_list))
        
        # compute metrics
        #train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, train_preds)
        #train_accuracy = tf.reduce_mean(train_accuracy)

        results = {
            'loss' : loss,
        #    'train_accuracy' : train_accuracy,
            'gradient_norm' : grad_norm,
            'mean_sampling_entropy' : tf.reduce_mean(self.cell.stats['sampling_entropy']),
        }

        if self.train_metrics is not None:
            m = self.train_metrics[0] if step_sampling else self.train_metrics[1]
            for metric, metric_fn in m.items():
                metric_fn.update_state(labels, train_preds)

        return results

    def train(self, dataset_gen, epochs, loss_fn, n_steps, optimizer, train_separately=False,
              log_every=1, validation_data_gen=None, metrics=None, ckpt_path=None, metrics_callback=None, 
              checkpoint_callback=None):
        
        SAMPLING_LR_FRACTION = 100. # 1. # 100.
        self.loss_fn = tf.keras.losses.get(loss_fn)
        self.optimizer = optimizer
        self.ckpt_path = ckpt_path 

        if metrics is None:
            metrics = {}

        # initialize the metrics
        self.train_metrics = [ {} for _ in range(0, 1 + train_separately) ]
        self.validation_metrics = {}

        for metric, (metric_fn, args) in metrics.items():
            for train_part in range(0, 1 + train_separately):
                self.train_metrics[train_part][metric] = metric_fn(**args)
            self.validation_metrics[metric] = metric_fn(**args)

        if train_separately:
            op_lr = self.optimizer.get_config()['learning_rate']
            # this step has to be done to initialize all variables on 
            # first tf.function call
            # perform this step with zero learning rate
            self.optimizer.lr = 0.
            model_inputs, training_inputs = next(iter(dataset_gen()))
            result = self.train_step(
                **model_inputs,
                **training_inputs,
                n_steps=n_steps,
                step_sampling=True,
                step_non_sampling=True,
                hard_sampling=False,
                soft_select=True)
            # set optimizer back to the original learning rate
            self.optimizer.lr = op_lr

        n_logs = 0
        dataset_size = None

        for e in tqdm(range(epochs), desc=f'Training ACGNN for {epochs} epochs', ncols=80, dynamic_ncols=True):
            # reset the validation metrics before every training epoch
            _ = [ m.reset_state() for m in self.validation_metrics.values() ]
            # reset the training metrics before every training epoch
            _ = [ m.reset_state() for d in self.train_metrics for m in d.values() ]

            for step, (model_inputs, training_inputs) in enumerate(tqdm(dataset_gen(), leave=False, total=dataset_size)):

                if e == 0:
                    if step == 0:
                        dataset_size = 0
                    dataset_size += 1


                if train_separately:
                    # perform training step of the edge sampling model
                    self.optimizer.lr = self.optimizer.lr / SAMPLING_LR_FRACTION
                    results_sampling = {}
                        
                    results_sampling = self.train_step(
                        **model_inputs,
                        **training_inputs,
                        n_steps=n_steps,
                        step_sampling=True,
                        # use soft sampling when optimizing the sampling model
                        hard_sampling=False,
                        step_non_sampling=False,
                        soft_select=True,
                    )
                    results_sampling.pop('edge_probs', None)
                    results_sampling.pop('sampling_entropy', None)

                    self.optimizer.lr = self.optimizer.lr * SAMPLING_LR_FRACTION
                    # perform training step of the model parts not involved in sampling
                    result_non_sampling = self.train_step(
                        **model_inputs,
                        **training_inputs,
                        n_steps=n_steps,
                        step_sampling=False,
                        # use hard sampling when optimizing the non-sampling model
                        hard_sampling=True,
                        step_non_sampling=True,
                        soft_select=True,
                    )

                    sampling_entropy = result_non_sampling.pop('sampling_entropy', None)
                    train_edge_probs = result_non_sampling.pop('edge_probs', None)

                    result = experiment_utils.combine_training_results(
                        result_non_sampling,
                        results_sampling)

                else:
                    result = self.train_step(
                        **model_inputs,
                        **training_inputs,
                        n_steps=n_steps,
                        step_sampling=True,
                        # use hard sampling when optimizing the non-sampling model
                        hard_sampling=False,
                        step_non_sampling=True,
                        soft_select=False,
                    )
                    sampling_entropy = result.pop('sampling_entropy', None)
                    train_edge_probs = result.pop('edge_probs', None)
                    result = experiment_utils.combine_training_results(result)

                result.update({
                    'epoch' : e,
                })

                if step % log_every == 0:
                    n_logs += 1

                if metrics_callback is not None:
                    metrics_callback(result)

                self.global_step.assign_add(1)

            epoch_train_results = result.copy()
            self.train_epoch.assign_add(1)
            
            if validation_data_gen is not None:
                result = self.evaluate(validation_data_gen, n_steps)

            validation_edge_probs = result.pop('edge_probs', None)
            validation_sampling_entropy = result.pop('sampling_entropy', None)
            
            train_metrics_result = [ { name : m.result() for name, m in d.items() } for d in self.train_metrics ]
            train_metrics_result = experiment_utils.combine_training_results(*train_metrics_result)
            result.update(train_metrics_result)
            result['epoch'] = e
            
            if metrics_callback is not None:
                metrics_callback(result)
                
            if self.ckpt_path is not None:
                epoch_ckpt_path = os.path.join(self.ckpt_path, 'epoch-%d' % self.train_epoch)
                self.save(epoch_ckpt_path, optimizer=self.optimizer)
                
    @tf.function(experimental_relax_shapes=True)
    def evaluation_step(self, node_features, edge_list, node_degrees, mask=None,
        edge_features=None, graph_mask=None, labels=None, n_steps=1, walker_hidden=None, soft_select=True, 
        hard_sampling=True):
        
        walker_pos, node_features, preds = self((
            None,
            node_features,
            edge_list,
            node_degrees,
            edge_features),
            n_steps=n_steps,
            walker_hidden=walker_hidden,
            hard_sampling=hard_sampling,
            soft_select=soft_select,
            graph_mask=graph_mask,
            training=False)
        
        if mask is not None:
            preds = tf.gather(preds, mask)

        for metric_fn in self.validation_metrics.values():
            metric_fn.update_state(labels, preds)

        results = {
            'validation/mean_sampling_entropy' : tf.reduce_mean(self.cell.stats['sampling_entropy']),
        }
        if self.validation_outputs:
            results.update({
                'validation/sampling_entropy' : self.cell.stats['sampling_entropy'],
                'validation/edge_probs' : self.cell.stats['edge_probs'],
            })

        return results

    def evaluate(self, dataset_gen, n_steps):
        mean_tensor = tf.keras.metrics.MeanTensor()

        for model_inputs, ground_truth in tqdm(dataset_gen(), desc='Validating...', leave=False):
            results = self.evaluation_step(
                **model_inputs, 
                **ground_truth,
                n_steps=n_steps,
                hard_sampling=False,
                soft_select=False)
            edge_probs = results.pop('validation/edge_probs', None)
            sampling_entropy = results.pop('validation/sampling_entropy', None)
            mean_tensor.update_state(list(results.values()))

        results = dict(zip(results.keys(), mean_tensor.result()))

        for metric, metric_fn in self.validation_metrics.items():
            results['validation/' + metric] = metric_fn.result()
    
        # only use last value, as these are large artifacs
        results['edge_probs'] = edge_probs
        results['sampling_entropy'] = sampling_entropy
        
        if self.validation_outputs:
            epoch_metrics_file = os.path.join(self.validation_outputs_path, 'epoch-%d-metrics.json' % self.train_epoch)
            epoch_ckpt_path = os.path.join(self.validation_outputs_path, 'epoch-%d' % self.train_epoch)
            # save the epoch metrics
            epoch_results = results
            epoch_results['sampling_entropy'] = sampling_entropy
            epoch_results['edge_probs'] = edge_probs
            epoch_results['checkpoint'] = epoch_ckpt_path
            epoch_results = dict(zip(
                epoch_results.keys(), 
                map(experiment_utils.serialize_json_item, epoch_results.values())
            ))
            experiment_utils.write_json(epoch_metrics_file, epoch_results)
            del epoch_results
        
        return results
    