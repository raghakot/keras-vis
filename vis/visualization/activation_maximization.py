from __future__ import absolute_import

import numpy as np
from keras import backend as K

from ..losses import ActivationMaximization
from ..optimizer import Optimizer
from ..regularizers import TotalVariation, LPNorm
from ..backprop_modifiers import get
from ..utils import utils


def visualize_activation_with_losses(input_tensor, losses, wrt_tensor=None,
                                     seed_input=None, input_range=(0, 255),
                                     input_indices=0, **optimizer_params):
    """Generates the `input_tensor` that minimizes the weighted `losses`. This function is intended for advanced
    use cases where a custom loss is desired.

    Args:
        input_tensor: An input tensor or list of input tensor.
            The shape of an input tensor is `(samples, channels, image_dims...)` if `image_data_format=
            channels_first`, Or it's `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: Seeds the optimization with a starting image. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        input_indices: A index or a list of index.
            This is the index of visualize target within `wrt_tensor`,
            but when `wrt_tensor` is None, it's `input_tensor`. (Default value = 0)
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer#optimizerminimize). Will default to
            reasonable values when required keys are not found.

    Returns:
        The model input that minimizes the weighted `losses`.
        When `input_indices` is a number, returned a model.input.
        But, when `input_indices` is a list of number, returned a list of model.input.
    """
    # Default optimizer kwargs.
    optimizer_params = utils.add_defaults_to_kwargs({
        'seed_inputs': seed_input,
        'max_iter': 200,
        'verbose': False
    }, **optimizer_params)

    opt = Optimizer(input_tensor, losses, input_range, wrt_tensors=wrt_tensor)
    opt_result = opt.minimize(**optimizer_params)

    images = []
    for i in utils.listify(input_indices):
        if i < len(opt_result):
            img, _, _ = opt_result[i]

            # If range has integer numbers, cast to 'uint8'
            if isinstance(input_range[0], int) and isinstance(input_range[1], int):
                img = np.clip(img, input_range[0], input_range[1]).astype('uint8')

            if K.image_data_format() == 'channels_first':
                img = np.moveaxis(img, 0, -1)

            images.append(img)
        else:
            raise ValueError('# TODO')

    if isinstance(input_indices, list):
        return images
    else:
        return images[input_indices]


def visualize_activation(model, layer_idx, filter_indices=None, wrt_tensor=None,
                         seed_input=None, input_range=(0, 255),
                         backprop_modifier=None, grad_modifier=None,
                         act_max_weight=1, lp_norm_weight=10, tv_weight=10,
                         input_indices=0, **optimizer_params):
    """Generates the model input that maximizes the output of all `filter_indices` in the given `layer_idx`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        seed_input: Seeds the optimization with a starting input. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)
        act_max_weight: The weight param for `ActivationMaximization` loss. Not used if 0 or None. (Default value = 1)
        lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None. (Default value = 10)
        tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None. (Default value = 10)
        input_indices: A index or a list of index.
            This is the index of visualize target within `wrt_tensor`,
            but when `wrt_tensor` is None, it's model.inputs. (Default value = 0)
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer#optimizerminimize). Will default to
            reasonable values when required keys are not found.

    Example:
        If you wanted to visualize the input image that would maximize the output index 22, say on
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer_idx = dense_layer_idx`.

        If `filter_indices = [22, 23]`, then it should generate an input image that shows features of both classes.

    Returns:
        The model input that maximizes the output of `filter_indices` in the given `layer_idx`.
        When `input_indices` is a number, returned a model.input.
        But, when `input_indices` is a list of number, returned a list of model.input.
    """
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight),
        (LPNorm(model.input), lp_norm_weight),
        (TotalVariation(model.input), tv_weight)
    ]

    # Add grad_filter to optimizer_params.
    optimizer_params = utils.add_defaults_to_kwargs({
        'grad_modifier': grad_modifier
    }, **optimizer_params)

    return visualize_activation_with_losses(model.input, losses, wrt_tensor,
                                            seed_input, input_range, input_indices,
                                            **optimizer_params)
