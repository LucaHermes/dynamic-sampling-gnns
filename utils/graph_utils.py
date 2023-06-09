import tensorflow as tf

def segment_softmax(x, segment_ids, num_segments, eps=1e-12):
    '''
    Computes the softmax over segments of x defined by segment_ids.
    Can be used to apply softmax to a neighborhood in a graph (e.g. for GAT).
    Example:
    segments = [1,     0,     0,     1,     1   ]
    data     = [3.,    1.,    1.,    3.,    3.  ]
    output   = [0.33,  0.5,   0.5,   0.33,  0.33]
    '''
    x_max = tf.math.segment_max(x, segment_ids)
    x_max = tf.gather(x_max, segment_ids)
    # in order to allow ever big numbers, use the identify: softmax(x + c) = softmax(x)
    # here c is the segment max (x_max)
    x = tf.exp(x - x_max)
    x_sum = tf.math.segment_sum(x, segment_ids)
    x_sum = tf.gather(x_sum + eps, segment_ids)
    return x / x_sum

def segment_max_hot(x, segment_ids, num_segments, eps_noise=1e-10):
    '''
    Finds the segment-maxima for segments in x.
    Returns a boolean mask with the maxima highlighted.
    '''
    # compute the segment maxima
    noise = tf.zeros([num_segments], dtype=x.dtype)
    s_max = tf.tensor_scatter_nd_max(noise, segment_ids[:,tf.newaxis], x)
    # expand to original size
    s_max = tf.gather(s_max, segment_ids)
    # compare max with non-maxed to create a boolean mask
    s_hot = tf.equal(x, s_max)

    s_hot_int = tf.cast(s_hot, tf.int32)

    # It may happen that there are two identical maxima in one segment. 
    # This is the workaround for that case (not super efficient):
    if tf.reduce_sum(s_hot_int) > num_segments:
        indices = tf.range(1, tf.shape(x)[0] + 1, dtype=tf.int32)
        masked_indices = s_hot_int * indices
        valid_indices = tf.math.unsorted_segment_max(masked_indices, segment_ids, num_segments)
        valid_indices = tf.gather(valid_indices, segment_ids)
        # this is guarwalkereed to only have one maximum per segment which coincides with
        # the segmwent-maxima in x.
        s_hot = tf.equal(indices, valid_indices)

    return s_hot

def symmetric_normalization(edge_index, node_degrees):
    node_degrees = tf.cast(node_degrees, tf.float32)
    normed_degrees = tf.maximum(1., tf.gather(node_degrees, edge_index))**-0.5
    return tf.reduce_prod(normed_degrees, axis=-1)

def segment_categorical_kl_divergence(probs_a, probs_b, segment_ids, num_segments, eps=1e-5):
    kl = probs_a * tf.math.log(probs_a / (probs_b + eps))
    return tf.math.unsorted_segment_sum(kl, segment_ids, num_segments)

def segment_entropy(segment_probs, segment_sizes, segment_ids, num_segments, eps=1e-8):
    log_bases = tf.repeat(segment_sizes, segment_sizes)
    log_bases = tf.cast(log_bases, tf.float32)
    segment_log_probs = tf.math.log(segment_probs + eps) / (tf.math.log(log_bases) + eps)
    segment_log_probs = segment_probs * segment_log_probs
    return -tf.math.segment_sum(segment_log_probs, segment_ids)

def get_sender_receiver(node_features, edge_index):
    '''
    Collects features from sending nodes and corresponding receiving
    nodes and returns both, e.g.
    A -> B
    A -> C
    C -> B
    returns the sender features [A, A, C] 
    and receiver features [B, C, B]
    '''
    senders = tf.gather(node_features, edge_index[:,0])
    receivers = tf.gather(node_features, edge_index[:,1])
    return senders, receivers

def edge_index_from_edge_list(edge_list, node_degrees=None, has_edge_idx=False):
    if node_degrees is None:
        node_degrees = edge_list.row_lengths()

    senders = tf.repeat(tf.range(tf.shape(node_degrees)[0]), node_degrees)
    receivers = edge_list.flat_values
    
    if receivers[0].shape != tf.TensorShape([]):
        receivers = receivers[:,0]
    
    return tf.stack((senders, receivers), axis=-1)