import pytest

import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
from vis.optimizer import Optimizer
from vis.losses import Loss


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


def test_wrt_tensor_is_None(model_and_losses):
    model, losses = model_and_losses
    opt = Optimizer(model.input, losses, wrt_tensor=None)
    opt.minimize()

    assert opt.wrt_tensor_is_input_tensor
    assert opt.wrt_tensor is not None
    assert opt.wrt_tensor != opt.input_tensor


def test_wrt_tensor_is_input_tensor(model_and_losses):
    model, losses = model_and_losses
    opt = Optimizer(model.input, losses, wrt_tensor=model.input)
    opt.minimize()

    assert opt.wrt_tensor_is_input_tensor
    assert opt.wrt_tensor is not None
    assert opt.wrt_tensor != opt.input_tensor


def test_wrt_tensor_isnt_input_tensor(model_and_losses):
    model, losses = model_and_losses
    opt = Optimizer(model.input, losses, wrt_tensor=model.output)
    opt.minimize()

    assert not opt.wrt_tensor_is_input_tensor
    assert opt.wrt_tensor is not None
    assert opt.wrt_tensor != opt.input_tensor


if __name__ == '__main__':
    pytest.main([__file__])
