import tensorflow as tf
from tensorflow_probability import distributions as tfd
import utils.graph_utils

class GumbelSampling(tf.keras.layers.Layer):

    def __init__(self, init_scale=1., temperature=0.6, learn_scales=False):
        '''
        Sample from a Gumbel-Softmax distribution with reparameterization.
        Parameters
        -----------
        init_scale : float or tensor
            The scale of the Gumbel distribution, higher values will produce more random results.
        temperature : float or tensor
            The temperature applied to the softmax, a high value (>=1) will favor distributions
            with a higher entropy; tends to produce uniform distributions.
            A low value (<=1) will produce more low-entropy distributions with distinct peaks.
            If temperature is a tensor dimensions must match those of logits.
        learn_scales : bool
            Specifies whether scales should be learned.
        '''
        super(GumbelSampling, self).__init__()
        self.learn_scales = learn_scales
        self.temperature = temperature
        if learn_scales:
            self.scales = tf.Variable(tf.ones(1) * init_scale)
        else:
            self.scales = init_scale

    def call(self, logits, logit_groups, n_groups, scales=None, hard=True, soft_select=False):
        '''
        Sample from a Gumbel-Softmax distribution with reparameterization.
        Parameters
        -----------
        logits : tensor
            logits of a categorical distribution from which to sample.
            The last dimension specifies the number of classes.
        scales : float or tensor
            The scale of the Gumbel distribution, higher values will produce more random results.
        hard : bool
            Specifies whether to output a hard or a soft one-hot distribution.
        '''
        scales = scales if scales is not None else self.scales

        s = tfd.Gumbel(tf.zeros_like(logits), tf.ones_like(logits) * scales).sample()
        s = (s + logits)

        if not hard:
            s = utils.graph_utils.segment_softmax(s / self.temperature, logit_groups, n_groups)
            return s, None


        s = utils.graph_utils.segment_softmax(s / self.temperature, logit_groups, n_groups)
        # It happended (weiredly) that two values in one segment were exactly the same, the workaround is
        # defined in utils.graph_utils.segment_max_hot. Still strage that two maxima are that likely.
        # if eps_noise is rather high this doesn't happen and we can set it high after the softmax
        s_hot = utils.graph_utils.segment_max_hot(s, logit_groups, n_groups, eps_noise=1e-4)
        s_hot.set_shape((None, ))
        s_hot_scaled = tf.cast(s_hot, tf.float32) # * s

        if soft_select:
            return s, s_hot, s

        return s + tf.stop_gradient(s_hot_scaled - s), s_hot, s