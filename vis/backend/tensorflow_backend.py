from __future__ import absolute_import

import os
import tempfile
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from keras.models import load_model


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
    # Retrieve from cache if previously modified.
    modified_model = _MODIFIED_MODEL_CACHE.get((model, backprop_modifier))
    if modified_model is not None:
        return modified_model

    # The general strategy is as follows:
    # - Modify all activations in the model as ReLU.
    # - Save a copy of model to temp file
    # - Call backend specific function that registers the custom op and loads the model under modified context manager.
    #
    # This is done because setting the activation in a Keras layer doesnt actually change the graph. We have to
    # iterate the entire graph and change the layer inbound and outbound nodes with modified tensors. This is doubly
    # complicated in Keras 2.x since multiple inbound and outbound nodes are allowed with the Graph API.
    #
    # This is a reliable and future proof strategy to modify activations in static graph computational frameworks.

    # Replace all layer activations with ReLU.
    # We also don't want to mutate the original model as it will have unexpected consequences on upstream callers.
    # For this reason we will maintain the set of original activations and restore it.
    original_activations = []
    for layer in model.layers[1:]:
        if hasattr(layer, 'activation'):
            original_activations.append(layer.activation)
            layer.activation = tf.nn.relu

    # Save model. This model should save with modified activation names.
    # Upon loading, keras should rebuild the graph with modified activations.
    model_path = '/tmp/' + next(tempfile._get_candidate_names()) + '.h5'
    model.save(model_path)

    # Restore original model to keep upstream callers unaffected.
    idx = 0
    for layer in model.layers[1:]:
        if hasattr(layer, 'activation'):
            layer.activation = original_activations[idx]
            idx += 1

    # Register modifier.
    modifier_fn = _BACKPROP_MODIFIERS.get(backprop_modifier)
    if modifier_fn is None:
        raise ValueError("'{}' modifier is not supported".format(backprop_modifier))
    modifier_fn(backprop_modifier)

    # Create graph under custom context manager.
    try:
        with tf.get_default_graph().gradient_override_map({'Relu': backprop_modifier}):
            modified_model = load_model(model_path)

            # Cache to impove subsequent call performance.
            _MODIFIED_MODEL_CACHE[(model, backprop_modifier)] = modified_model
            return modified_model
    finally:
        # Clean up temp file.
        os.remove(model_path)


def set_random_seed(seed_value=1337):
    """Sets random seed value for reproducibility.

    Args:
        seed_value: The seed value to use. (Default Value = infamous 1337)
    """
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
