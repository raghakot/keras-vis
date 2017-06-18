from __future__ import absolute_import

import six
import tensorflow as tf
from keras import backend as K


def across_data_formats(func):
    """Function wrapper to run tests on multiple keras data_format and clean up after TensorFlow tests.

    # Arguments
        func: test function to clean up after.

    # Returns
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
