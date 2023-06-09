import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, units, activation='relu', dropout=0, use_batch_norm=False):
        '''
        An RNN with a linear transformation in front.
        '''
        super(ResidualBlock, self).__init__()
        self.units = units
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(units*2, activation=activation)
        if use_batch_norm:
            self.norm = tf.keras.layers.BatchNormalization()
        else:
            self.norm = lambda x, training=False: x
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense2 = tf.keras.layers.Dense(units)

    def call(self, x, hidden_state, training=False):
        '''
        Computes the dot product between x and y.
        '''
        h_in = hidden_state
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.norm(x, training=training)
        x = self.dense2(x)
        x = x + h_in
        return x, x
