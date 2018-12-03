import numpy as np
import tensorflow as tf

from keras.layers import Dense, Dropout, LeakyReLU, Lambda
from keras.models import Sequential
from keras import backend as K

class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        self.model.add(Dense(1))

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)
