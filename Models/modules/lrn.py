import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


# tf.keras에 맞게 lrn을 조금 수정했다.

class LRN(Layer):

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, r, c, ch = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = K.square(x) # square the input

        input_sqr = tf.pad(input_sqr, [[0, 0], [0, 0], [0, 0], [half_n, half_n]])
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, :, :, i:i+ch]
        scale = scale ** self.beta
        x = x / scale
        return x


    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))