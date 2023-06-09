import tensorflow as tf

class ZeroAttention(tf.keras.layers.Layer):
    '''
    Zero attention returns 0's for all inputs. As these are considered as logits,
    this is a uniform distribution over the neighborhood.
    '''
    
    def __init__(self, *args, **kwargs):
        super(ZeroAttention, self).__init__()

    def call(self, from_vec, to_vec, training=False):
        '''
        Allways returns zero attention.
        '''
        return tf.zeros((tf.shape(from_vec)[0],))

class GATAttention(tf.keras.layers.Layer):
    '''
    This is derived from the attention mechanism that is also used in the Graph 
    Attention Networks (GAT) (Veličković, 2017, https://arxiv.org/abs/1710.10903)
    
    This implements the function as `f(x, y) = a^T (MLP(x) || MLP(y))`,
    where x and y are the inputs, a is a learnable vector and || denotes 
    vector concatenation, the MLPs don't use weight-sharing.
    '''

    def __init__(self, units, activation='leaky_relu', dropout=0):
        super(GATAttention, self).__init__()
        with tf.name_scope('neighborhood_attention') as scope:
            self.dense1 = tf.keras.layers.Dense(units, name=scope, activation=activation)
            self.dense2 = tf.keras.layers.Dense(units, name=scope, use_bias=False)
            self.dense3 = tf.keras.layers.Dense(units, name=scope, activation=activation)
            self.dense4 = tf.keras.layers.Dense(units, name=scope, use_bias=False)
            self.att_layer = tf.keras.layers.Dense(1, use_bias=False, name=scope)
            self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, from_vec, to_vec, training=False):
        '''
        Computes a gat-style non-commutative attention between from_features and to_features.
        '''
        from_vec = self.dropout(from_vec, training=training)
        to_vec = self.dropout(to_vec, training=training)
        from_vec = self.dense2(self.dense1(from_vec))
        to_vec = self.dense4(self.dense3(to_vec))
        a = self.att_layer(tf.concat((from_vec, to_vec), axis=-1))
        return tf.squeeze(a, axis=-1)

class DotAttention(tf.keras.layers.Layer):
    '''
    This is a simpler form of attention than the GATAttention that is a dot product
    computed from tranformed inputs.
    
    This implements the function as `f(x, y) = MLP(x)^T MLP(y)`,
    where x and y are the inputs, the MLPs don't use weight-sharing.
    '''

    def __init__(self, units, activation='leaky_relu', dropout=0):
        super(DotAttention, self).__init__()
        with tf.name_scope('neighborhood_attention') as scope:
            self.dense1 = tf.keras.layers.Dense(units, name=scope, activation=activation)
            self.dense2 = tf.keras.layers.Dense(units, name=scope, use_bias=False)
            self.dense3 = tf.keras.layers.Dense(units, name=scope, activation=activation)
            self.dense4 = tf.keras.layers.Dense(units, name=scope, use_bias=False)
            self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, y, training=False):
        '''
        Computes the dot product between x and y.
        '''
        x = self.dropout(x, training=training)
        y = self.dropout(y, training=training)
        x = self.dense2(self.dense1(x))
        y = self.dense4(self.dense3(y))
        return tf.reduce_sum(x * y, axis=-1)

class Dot(tf.keras.layers.Layer):
    '''
    This is a simpler form of attention than the DotAttention that is just an
    unparameterized dot product of the inputs. This means that the input vectors 
    must have same dimensionality.
    
    This implements the function as `f(x, y) = x^T y`,
    where x and y are the inputs.
    '''

    def call(self, x, y):
        '''
        Computes the dot product between x and y.
        '''
        return tf.reduce_sum(x * y, axis=-1)
