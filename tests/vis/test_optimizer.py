import keras.backend as K
from keras.layers import Dense, Input, Add
from keras.models import Model, Sequential
from vis.losses import Loss
from vis.optimizer import Optimizer

import pytest


class _DummyLoss(Loss):
    def __init__(self, model):
        self.name = 'dummy-loss'
        self.output = model.output

    def build_loss(self):
        return K.sum(self.output * self.output)


@pytest.fixture(scope="function", autouse=True)
def model_and_losses():
    model = Sequential([Dense(4, activation='linear', input_shape=(2, ))])
    losses = [(_DummyLoss(model), 1)]
    return model, losses


@pytest.fixture(scope="function", autouse=True)
def multi_inputs_model_and_losses():
    input_a = Input((2, ))
    x1 = Dense(4, activation='linear')(input_a)
    input_b = Input((2, ))
    x2 = Dense(4, activation='linear')(input_b)
    output = Add()([x1, x2])
    model = Model(inputs=[input_a, input_b], outputs=output)
    losses = [(_DummyLoss(model), 1)]
    return model, losses


def test_wrt_tensors_is_None(model_and_losses):
    model, losses = model_and_losses
    opt = Optimizer(model.input, losses, wrt_tensors=None)
    opt.minimize(max_iter=1)

    assert opt.wrt_tensors_is_input_tensors
    assert opt.wrt_tensors is not None
    assert opt.wrt_tensors != opt.input_tensors


def test_wrt_tensors_is_input_tensors(model_and_losses):
    model, losses = model_and_losses
    opt = Optimizer(model.input, losses, wrt_tensors=model.input)
    opt.minimize(max_iter=1)

    assert opt.wrt_tensors_is_input_tensors
    assert opt.wrt_tensors is not None
    assert opt.wrt_tensors != opt.input_tensors


def test_wrt_tensors_isnt_input_tensor(model_and_losses):
    model, losses = model_and_losses
    opt = Optimizer(model.input, losses, wrt_tensors=model.output)
    opt.minimize(max_iter=1)

    assert not opt.wrt_tensors_is_input_tensors
    assert opt.wrt_tensors is not None
    assert opt.wrt_tensors != opt.input_tensors


def test_multi_inputs(multi_inputs_model_and_losses):
    model, losses = multi_inputs_model_and_losses
    opt = Optimizer(model.inputs, losses, wrt_tensors=None)
    opt.minimize(max_iter=1)

    assert opt.wrt_tensors_is_input_tensors
    assert opt.wrt_tensors is not None
    assert opt.wrt_tensors != opt.input_tensors


if __name__ == '__main__':
    pytest.main([__file__])
