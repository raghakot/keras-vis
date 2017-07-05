from __future__ import absolute_import

from . import backend
from .utils import utils


def guided(model):
    """Modifies backprop to only propagate positive gradients for positive activations.

    Args:
        model: The `keras.models.Model` instance whose gradient computation needs to be overridden.

    References:
        Details on guided back propagation can be found in paper: [String For Simplicity: The All Convolutional Net]
        (https://arxiv.org/pdf/1412.6806.pdf)
    """
    return backend.modify_model_backprop(model, 'guided')


def rectified(model):
    """Modifies backprop to only propagate positive gradients.

    Args:
        model: The `keras.models.Model` instance whose gradient computation needs to be overridden.

    References:
        Details can be found in the paper: [Visualizing and Understanding Convolutional Networks]
        (https://arxiv.org/pdf/1311.2901.pdf)
    """
    return backend.modify_model_backprop(model, 'rectified')


# Create aliases
relu = deconv = rectified


def get(identifier):
    return utils.get_identifier(identifier, globals(), __name__)
