from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers.core import Lambda
from keras import backend as K
from keras.layers import Dense, Activation, Flatten
import tensorflow as tf


def global_average_pooling(x):
    return tf.reduce_mean(x, (1, 2))


def global_average_pooling_shape(input_shape):
    return (input_shape[0], input_shape[3])


def atan_layer(x):
    return tf.mul(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return K.variable(initial)


def steering_net():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, init=normal_init, subsample=(2, 2), name='conv1_1', input_shape=(66, 200, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init=normal_init, subsample=(2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, init=normal_init, subsample=(2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init=normal_init, subsample=(1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init=normal_init, subsample=(1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, init=normal_init, name="dense_0"))
    model.add(Activation('relu'))
    model.add(Dense(100, init=normal_init, name="dense_1"))
    model.add(Activation('relu'))
    model.add(Dense(50, init=normal_init, name="dense_2"))
    model.add(Activation('relu'))
    model.add(Dense(10, init=normal_init, name="dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init=normal_init, name="dense_4"))
    model.add(Lambda(atan_layer, output_shape=atan_layer_shape, name="atan_0"))
    return model


def get_model():
    model = steering_net()
    model.compile(loss='mse', optimizer='Adam')
    return model


def load_model(path):
    model = steering_net()
    model.load_weights(path)
    model.compile(loss='mse', optimizer='Adam')
    return model
