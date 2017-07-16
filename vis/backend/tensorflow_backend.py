from __future__ import absolute_import

import os
import tempfile
import inspect
import numpy as np
import tensorflow as tf

from ..utils import utils
from tensorflow.python.framework import ops
from keras.models import load_model
from keras.layers import advanced_activations, Activation


# Register all classes with `advanced_activations` module
_ADVANCED_ACTIVATIONS = set()
for name, obj in inspect.getmembers(advanced_activations, inspect.isclass):
    if not name.startswith("_") and hasattr(obj, "__module__") and obj.__module__ == advanced_activations.__name__:
        _ADVANCED_ACTIVATIONS.add(obj)
_ADVANCED_ACTIVATIONS = tuple(_ADVANCED_ACTIVATIONS)


def _register_guided_gradient(name):
    if name not in ops._gradient_registry._registry:
        @tf.RegisterGradient(name)
        def _guided_backprop(op, grad):
            dtype = op.outputs[0].dtype
            gate_g = tf.cast(grad > 0., dtype)
            gate_y = tf.cast(op.outputs[0] > 0, dtype)
            return gate_y * gate_g * grad


def _register_rectified_gradient(name):
    if name not in ops._gradient_registry._registry:
        @tf.RegisterGradient(name)
        def _relu_backprop(op, grad):
            dtype = op.outputs[0].dtype
            gate_g = tf.cast(grad > 0., dtype)
            return gate_g * grad

# Map of modifier type to registration function.
_BACKPROP_MODIFIERS = {
    'guided': _register_guided_gradient,
    'rectified': _register_rectified_gradient
}


# Maintain a mapping of original model, backprop_modifier -> modified model as cache.
_MODIFIED_MODEL_CACHE = dict()


def modify_model_backprop(model, backprop_modifier):
    """Creates a copy of model by modifying all activations to use a custom op to modify the backprop behavior.

    Args:
        model:  The `keras.models.Model` instance.
        backprop_modifier: One of `{'guided', 'rectified'}`

    Returns:
        A copy of model with modified activations for backwards pass.
    """
    # The general strategy is as follows:
    # - Clone original model via save/load so that upstream callers don't see unexpected results with their models.
    # - Modify all activations in the model as ReLU.
    # - Save modified model so that it can be loaded with custom context modifying backprop behavior.
    # - Call backend specific function that registers the custom op and loads the model under modified context manager.
    # - Maintain cache to save this expensive process on subsequent calls.
    #
    # The reason for this round about way is because the graph needs to be rebuild when any of its layer builder
    # functions are changed. This is very complicated to do in Keras and makes the implementation very tightly bound
    # with keras internals. By saving and loading models, we dont have to worry about future compatibility.
    #
    # The only exception to this is the way advanced activations are handled which makes use of some keras internal
    # knowledge and might break in the future.

    # 0. Retrieve from cache if previously computed.
    modified_model = _MODIFIED_MODEL_CACHE.get((model, backprop_modifier))
    if modified_model is not None:
        return modified_model

    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        # 1. Clone original model via save and load.
        model.save(model_path)
        modified_model = load_model(model_path)

        # 2. Replace all possible activations with ReLU.
        for i, layer in utils.reverse_enumerate(modified_model.layers):
            if hasattr(layer, 'activation'):
                layer.activation = tf.nn.relu
            if isinstance(layer, _ADVANCED_ACTIVATIONS):
                # NOTE: This code is brittle as it makes use of Keras internal serialization knowledge and might
                # break in the future.
                modified_layer = Activation('relu')
                modified_layer.inbound_nodes = layer.inbound_nodes
                modified_layer.name = layer.name
                modified_model.layers[i] = modified_layer

        # 3. Save model with modifications.
        modified_model.save(model_path)

        # 4. Register modifier and load modified model under custom context.
        modifier_fn = _BACKPROP_MODIFIERS.get(backprop_modifier)
        if modifier_fn is None:
            raise ValueError("'{}' modifier is not supported".format(backprop_modifier))
        modifier_fn(backprop_modifier)

        # 5. Create graph under custom context manager.
        with tf.get_default_graph().gradient_override_map({'Relu': backprop_modifier}):
            #  This should rebuild graph with modifications.
            modified_model = load_model(model_path)

            # Cache to improve subsequent call performance.
            _MODIFIED_MODEL_CACHE[(model, backprop_modifier)] = modified_model
            return modified_model
    finally:
        os.remove(model_path)


def set_random_seed(seed_value=1337):
    """Sets random seed value for reproducibility.

    Args:
        seed_value: The seed value to use. (Default Value = infamous 1337)
    """
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
