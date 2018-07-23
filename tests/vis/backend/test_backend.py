import pytest
import numpy as np

from vis.backend import modify_model_backprop
from vis.utils.test_utils import skip_backends

import keras
from keras.models import Model, Input, Sequential
from keras.layers import Dense
from keras.initializers import Constant
from keras import backend as K
from keras.activations import get
from keras.layers import advanced_activations, Activation


def _compute_grads(model, input_array):
    grads_fn = K.gradients(model.output, model.input)[0]
    compute_fn = K.function([model.input, K.learning_phase()], [grads_fn])
    return compute_fn([np.array([input_array]), 0])[0][0]


@skip_backends('theano')
def test_guided_grad_modifier():
    # Create a simple 2 dense layer model.
    simple_model = Sequential([
        Dense(2, activation='relu', use_bias=False, kernel_initializer=Constant([[-1., 1.], [-1., 1.]]), input_shape=(2,)),
        Dense(1, activation='linear', use_bias=False, kernel_initializer=Constant([-1., 1.]))
    ])
    simple_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam())

    # Create a simple 2 dense layer model using Activation.
    simple_model_with_activation = Sequential([
        Dense(2, activation='linear', use_bias=False, kernel_initializer=Constant([[-1., 1.], [-1., 1.]]), input_shape=(2,)),
        Activation('relu'),
        Dense(1, activation='linear', use_bias=False, kernel_initializer=Constant([-1., 1.]))
    ])
    simple_model_with_activation.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam())

    for i, model in enumerate([simple_model, simple_model_with_activation]):
        # Create guided backprop model
        modified_model = modify_model_backprop(model, 'guided')

        # Gradients are zeros.
        input_array = [0., 0.]
        assert np.array_equal(_compute_grads(model, input_array), [0., 0.])
        assert np.array_equal(_compute_grads(modified_model, input_array), [0., 0.])

        # Below 3 cases, GuidedBackprop gradients is the same as Original gradients.
        input_array = [1., 0.]
        assert np.array_equal(_compute_grads(model, input_array), [1., 1.])
        assert np.array_equal(_compute_grads(modified_model, input_array), [1., 1.])

        input_array = [0., 1.]
        assert np.array_equal(_compute_grads(model, input_array), [1., 1.])
        assert np.array_equal(_compute_grads(modified_model, input_array), [1., 1.])

        input_array = [1., 1.]
        assert np.array_equal(_compute_grads(model, input_array), [1., 1.])
        assert np.array_equal(_compute_grads(modified_model, input_array), [1., 1.])

        # If inputs contains negative values,
        # GuidedBackprop gradients is not the same as Original gradients.
        input_array = [-1., 0.]
        assert np.array_equal(_compute_grads(model, input_array), [1., 1.])
        assert np.array_equal(_compute_grads(modified_model, input_array), [0., 0.])

        input_array = [0., -1.]
        assert np.array_equal(_compute_grads(model, input_array), [1., 1.])
        assert np.array_equal(_compute_grads(modified_model, input_array), [0., 0.])

        input_array = [-1., -1.]
        assert np.array_equal(_compute_grads(model, input_array), [1., 1.])
        assert np.array_equal(_compute_grads(modified_model, input_array), [0., 0.])

        # Activation is not changed.
        if i == 0:  # modified first model
            modified_model.layers[0].activation == keras.activations.relu
            modified_model.layers[1].activation == keras.activations.linear
        if i == 1:  # modified second model
            modified_model.layers[0].activation == keras.activations.linear
            modified_model.layers[1].activation == keras.activations.relu
            modified_model.layers[2].activation == keras.activations.linear


# Currently, the modify_model_backprop function doesn't support advanced activation.
# Therefore, this test case will temporarily comment out.
#
# @skip_backends('theano')
# def test_advanced_activations():
#     """ Tests that various ways of specifying activations in keras models are handled when replaced with Relu
#     """
#     inp = Input(shape=(2, ))
#     x = Dense(5, activation='elu')(inp)
#     x = advanced_activations.LeakyReLU()(x)
#     x = Activation('elu')(x)
#     model = Model(inp, x)
#
#     # Ensure that layer.activation, Activation and advanced activations are replaced with relu
#     modified_model = modify_model_backprop(model, 'guided')
#     assert modified_model.layers[1].activation == get('relu')
#     assert modified_model.layers[2].activation == get('relu')
#     assert modified_model.layers[3].activation == get('relu')
#
#     # Ensure that original model is unchanged.
#     assert model.layers[1].activation == get('elu')
#     assert isinstance(model.layers[2], advanced_activations.LeakyReLU)
#     assert model.layers[3].activation == get('elu')


# @skip_backends('theano')
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
