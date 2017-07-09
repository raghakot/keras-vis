from __future__ import absolute_import

import six
import tensorflow as tf
from keras import backend as K
from . import utils


def across_data_formats(func):
    """Function wrapper to run tests on multiple keras data_format and clean up after TensorFlow tests.

    Args:
        func: test function to clean up after.

    Returns:
        A function wrapping the input function.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        for data_format in {'channels_first', 'channels_last'}:
            K.set_image_data_format(data_format)
            func(*args, **kwargs)
            if K.backend() == 'tensorflow':
                K.clear_session()
                tf.reset_default_graph()
    return wrapper


def skip_backends(backends):
    """Function wrapper to specify which backends should skip the test.

    Args:
        backends: The list of backends to skip.

    Returns:
        A function wrapping the input function.
    """
    backends = set(utils.listify(backends))

    def decorator(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            if K.backend() in backends:
                return
            func(*args, **kwargs)
        return wrapper
    return decorator
