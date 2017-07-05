from __future__ import absolute_import
import numpy as np


def modify_model_backprop(model, backprop_modifier):
    """Creates a copy of model by modifying all activations to use a custom op to modify the backprop behavior.

   Args:
       model:  The `keras.models.Model` instance.
       backprop_modifier: One of `{'guided', 'rectified'}`

   Returns:
       A copy of model with modified activations for backwards pass.
   """
    raise NotImplementedError('Theano version is not supported yet.')


def set_random_seed(seed_value=1337):
    """Sets random seed value for reproducibility.

    Args:
        seed_value: The seed value to use. (Default Value = infamous 1337)
    """
    np.random.seed(seed_value)
