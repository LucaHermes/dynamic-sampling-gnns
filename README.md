# Graph Learning by Dynamic Sampling
Graph neural networks based on message-passing rely on the principle of neighborhood aggregation which has shown to work well for many graph tasks. In other cases these approaches appear insufficient, for example, when graphs are heterophilic. In such cases, it can help to modulate the aggregation method depending on the characteristic of the current neighborhood. Furthermore, when considering higher-order relations, heterophilic settings become even more important.
In this work, we investigate a sparse version of message-passing that allows selective neighbor integration and aims for learning to identify most salient nodes that are then integrated over. In our approach, information on individual nodes is encoded by generating distinct walks. Because these walks follow distinct trajectories, the higher-order neighborhood grows only linearly which mitigates information bottlenecks. Overall, we aim to find the most salient substructures by deploying a learnable sampling strategy. We validate our method on commonly used graph benchmarks and show the effectiveness especially in heterophilic graphs. We finally discuss possible extensions to the framework.

![dynamic_sampling_main_model_figure](https://github.com/LucaHermes/dynamic-sampling-gnns/assets/30961397/64d9f519-6cfe-45c5-9171-6e06d4c5c837)
<sub><br><b>Figure:</b> Overview of the dynamic sampling GNN framework. Left: An individual walker (dark blue) sampling two steps, updating its own state (solid blue arrows) and the node features of the origin node (solid black arrows). Right: Steps performed to sample a single step -- here from node 1 to node 2: Model $p_\theta$ computes edge logits (1) from which the trajectory is sampled to select a single neighbor (2). The walker traverses to the selected neighbor and updates its own state using $s_\phi$ (3). Finally, the walker updates its origin node (4). If multiple walkers (here red and green)---that originated in the same node 1---meet, their states contribute equally to the node update (4). Computations are denoted by solid black arrows, yellow boxes denote parameterized functions and dashed lines denote sampling trajectories.</sub>

# Codebase Overview:

 * `model.gumbel.GumbelSampling`: Differentiable Sampling from categorical distributions. Pass logits as a segmented tensor, together with segment ids. Specify whether you want one-hot samples, or a soft-selection via `soft_select`.
 * `model.attention_models`: Package with different attention functions. Contains the GAT-style and dot product attention.
 * `model.dsgnn_cells.DSGNNCell`: Implements the stepwise logic as described in the paper. The call method performs one step of each walker along the graph edges, integration of node features into the walker state and update of the origin node of the walker.
 * `model.state_models.ResidualBlock`: A simple walker state model implemented as a residual block.
 * `model.base_cell.DSGNNCellBase`: A base cell that provides utility functions for graph traversal and state integration. Can be inherited from to construct custom DSGNN cells.
 * `model.base_model.DSGNN`: A base model which contains the training loop with the two-step training procedure presented in the paper. Also contains evaluation routines, readout functions and other utility functions such as model checkpointing, saving and loading.
 * `benchmark_transductive.py`: A script to run the benchmarks that are shown in the paper. Here, the checkpoints from all epochs are necessary, because they are being compared across the individual 10-fold cross validation results.
 * `run.py`: Use this to run the experiments. Provides a CLI to customize model, dataset and training parameters.
 * `SimpleSamplingTask.ipynb`: A jupyter notebook that shows a simple usecase of the sampling model. The task thats implemented here is a simple navigation task. A graph is generated, a target node is selected and the sampling model is trained to guide the walkers to the target node.
 * `configs.py`: Default configutrations of the model, training and dataset.
