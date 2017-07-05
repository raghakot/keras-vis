from __future__ import absolute_import

import numpy as np
from keras import backend as K
from .utils import utils


def negate(grads):
    """Negates the gradients.

    Args:
        grads: A numpy array of grads to use.

    Returns:
        The negated gradients.
    """
    return -grads


def absolute(grads):
    """Computes absolute gradients.

    Args:
        grads: A numpy array of grads to use.

    Returns:
        The absolute gradients.
    """
    return np.abs(grads)


def invert(grads):
    """Inverts the gradients.

    Args:
        grads: A numpy array of grads to use.

    Returns:
        The inverted gradients.
    """
    return 1. / (grads + K.epsilon())


def relu(grads):
    """Clips negative gradient values.

    Args:
        grads: A numpy array of grads to use.

    Returns:
        The rectified gradients.
    """
    grads[grads < 0.] = 0.
    return grads


def small_values(grads):
    """Can be used to highlight small gradient values.

    Args:
        grads: A numpy array of grads to use.

    Returns:
        The modified gradients that highlight small values.
    """
    return absolute(invert(grads))


def get(identifier):
    return utils.get_identifier(identifier, globals(), __name__)
