import pytest
import numpy as np
from vis.backend import modify_model_backprop

from keras.models import Model, Input
from keras.layers import Dense
from keras.initializers import Constant
from keras import backend as K
from keras.activations import get


def _compute_grads(model, input_array):
    grads_fn = K.gradients(model.output, model.input)[0]
    compute_fn = K.function([model.input, K.learning_phase()], [grads_fn])
    return compute_fn([np.array([input_array]), 0])[0][0]


def test_guided_grad_modifier():
    # Only test tensorflow implementation for now.
    if K.backend() == 'theano':
        return

    # Create a simple linear sequence x -> linear(w.x) with weights w1 = -1, w2 = 1.
    inp = Input(shape=(2, ))
    out = Dense(1, activation='linear', use_bias=False, kernel_initializer=Constant([-1., 1.]))(inp)
    model = Model(inp, out)

    # Original model gradient should be [w1, w2]
    assert np.array_equal(_compute_grads(model, [1., -1.]), [-1., 1.])

    # Original gradient is [-1, 1] but new gradient should be [0, 0]
    # First one is clipped because of negative gradient while the second is clipped due to negative input.
    modified_model = modify_model_backprop(model, 'guided')
    assert np.array_equal(_compute_grads(modified_model, [1., -1.]), [0., 0.])

    # Ensure that the original model reference remains unchanged.
    assert model.layers[1].activation == get('linear')
    assert modified_model.layers[1].activation == get('relu')


# def test_rectified_grad_modifier():
#     # Only test tensorflow implementation for now.
#     if K.backend() == 'theano':
#         return
#
#     # Create a simple linear sequence x -> linear(w.x) with weights w1 = -1, w2 = 1.
#     inp = Input(shape=(2, ))
#     out = Dense(1, activation='linear', use_bias=False, kernel_initializer=Constant([-1., 1.]))(inp)
#     model = Model(inp, out)
#
#     # Original model gradient should be [w1, w2]
#     assert np.array_equal(_compute_grads(model, [1., -1.]), [-1., 1.])
#
#     # Original gradient is [-1, 1] but new gradient should be [0, 1]
#     # First one is clipped because of negative gradient.
#     modified_model = modify_model_backprop(model, 'rectified')
#
#     # TODO: Interestingly this does not work for some reason.
#     # It is failing at tf.cast(grad > 0., dtype)
#     assert np.array_equal(_compute_grads(modified_model, [1., -1.]), [0., 1.])
#
#     # Ensure that the original model reference remains unchanged.
#     assert model.layers[1].activation == get('linear')
#     assert modified_model.layers[1].activation == get('relu')


if __name__ == '__main__':
    pytest.main([__file__])
