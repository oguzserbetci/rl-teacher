import numpy as np
import tensorflow as tf

from keras.layers import Dense, Dropout, LeakyReLU, Input
from keras.models import Sequential, Model

class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)


class RepresentationLearningMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        obs_dim = np.prod(obs_shape)
        input_dim = obs_dim + np.prod(act_shape)

        state_action = Input(shape=(input_dim,))

        x = Dropout(0.5)(state_action)
        x = Dense(h_size)(x)
        x = LeakyReLU()(x)

        x = Dropout(0.5)(x)
        x = Dense(h_size)(x)
        x = LeakyReLU()(x)

        # x = Dropout(0.2)(x)
        reward = Dense(1)(x)
        next_state = Dense(obs_dim)(x)

        self.model = Model(inputs=state_action, outputs=[reward, next_state])
        # self.next_state = Model(inputs=state_action, outputs=next_state)

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)  # , self.next_state(x)
