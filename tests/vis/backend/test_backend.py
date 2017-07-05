import pytest
import numpy as np
from vis.backend import modify_model_backprop

from keras.models import Model, Input
from keras.layers import Dense
from keras.initializers import Constant
from keras import backend as K
from keras.activations import get


def _compute_grads(model, input_value):
    grads_fn = K.gradients(model.output, model.input)[0]
    compute_fn = K.function([model.input, K.learning_phase()], [grads_fn])
    return compute_fn([np.array([[input_value]]), 0])[0][0]


def test_guided_grad_modifier():
    # Only test tensorflow implementation for now.
    if K.backend() == 'theano':
        return

    # Create a simple linear sequence x -> linear(w1.x)
    inp = Input(shape=(1, ))
    out = Dense(1, activation='linear', use_bias=False, kernel_initializer=Constant(-1.))(inp)
    model = Model(inp, out)

    # Original model gradient is negative but the modified model should clip it.
    assert _compute_grads(model, 1.) == -1

    # Modified model should clip negative gradients.
    modified_model = modify_model_backprop(model, 'guided')
    assert _compute_grads(modified_model, 1.) == 0

    # Ensure that the original model reference remains unchanged.
    assert model.layers[1].activation == get('linear')
    assert modified_model.layers[1].activation == get('relu')


def test_rectified_grad_modifier():
    # Only test tensorflow implementation for now.
    if K.backend() == 'theano':
        return

    # Create a simple model y = linear(w.x) where w = 1
    inp = Input(shape=(1, ))
    out = Dense(1, activation='linear', use_bias=False, kernel_initializer=Constant(-1.))(inp)
    model = Model(inp, out)

    # Original model gradient is negative but the modified model should clip it.
    assert _compute_grads(model, 1.) == -1

    # Modified model should clip negative gradients.
    modified_model = modify_model_backprop(model, 'rectified')
    assert _compute_grads(modified_model, 1.) == 0

    # Ensure that the original model reference remains unchanged.
    assert model.layers[1].activation == get('linear')
    assert modified_model.layers[1].activation == get('relu')


if __name__ == '__main__':
    test_rectified_grad_modifier()
    # pytest.main([__file__])
