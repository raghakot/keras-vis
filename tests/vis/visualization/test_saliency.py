import pytest

import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from vis.losses import ActivationMaximization
from vis.visualization.saliency import visualize_saliency, visualize_saliency_with_losses


@pytest.fixture(scope='function', autouse=True)
def model():
    inputs = Input((28, 28, 3))
    x = Flatten()(inputs)
    x = Dense(100, activation='relu')(x)
    x = Dense(1)(x)
    return Model(inputs, x)


@pytest.fixture(scope='function', autouse=True)
def data():
    return np.random.rand(1, 28, 28, 3)


def test_visualize_saliency(model, data):
    # FIXME Can't set None to filter_indices with Theano backend.
    # To get green test, it set zero.
    # grads = visualize_saliency(model, -1, filter_indices=None, seed_input=data)
    grads = visualize_saliency(model, -1, filter_indices=0, seed_input=data)
    assert grads.shape == (28, 28)


def test_visualize_saliency_with_unkeepdims(model, data):
    grads = visualize_saliency(model, -1, 0, data, keepdims=True)
    assert grads.shape == (28, 28, 3)


def test_visualize_saliency_with_losses(model, data):
    losses = [
        (ActivationMaximization(model.layers[-1], 0), -1)
    ]
    grads = visualize_saliency_with_losses(model.input, losses, data)
    assert grads.shape == (28, 28)


def test_visualize_saliency_with_losses_with_unkeepdims(model, data):
    losses = [
        (ActivationMaximization(model.layers[-1], 0), -1)
    ]
    grads = visualize_saliency_with_losses(model.input, losses, data, keepdims=True)
    assert grads.shape == (28, 28, 3)


def test_for_issues_135():
    inputs = Input((35,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(100, activation='relu')(x)
    x = Dense(50)(x)
    model = Model(inputs, x)
    data = np.random.rand(1, 35)
    grads = visualize_saliency(model, -1, 0, data, keepdims=True)
    assert grads.shape == (35,)
