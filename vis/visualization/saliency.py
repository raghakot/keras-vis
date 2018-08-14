from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K

from ..losses import ActivationMaximization
from ..optimizer import Optimizer
from ..backprop_modifiers import get
from ..utils import utils


def _find_penultimate_layer(model, layer_idx, penultimate_layer_idx):
    """Searches for the nearest penultimate `Conv` or `Pooling` layer.

    Args:
        model: The `keras.models.Model` instance.
        layer_idx: The layer index within `model.layers`.
        penultimate_layer_idx: The pre-layer to `layer_idx`. If set to None, the nearest penultimate
            `Conv` or `Pooling` layer is used.

    Returns:
        The penultimate layer.
    """
    if penultimate_layer_idx is None:
        for idx, layer in utils.reverse_enumerate(model.layers[:layer_idx - 1]):
            if isinstance(layer, Wrapper):
                layer = layer.layer
            if isinstance(layer, (_Conv, _Pooling1D, _Pooling2D, _Pooling3D)):
                penultimate_layer_idx = idx
                break

    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Conv` or `Pooling` '
                         'layer for layer_idx: {}'.format(layer_idx))

    # Handle negative indexing otherwise the next check can fail.
    if layer_idx < 0:
        layer_idx = len(model.layers) + layer_idx
    if penultimate_layer_idx > layer_idx:
        raise ValueError('`penultimate_layer_idx` needs to be before `layer_idx`')

    return model.layers[penultimate_layer_idx]


def visualize_saliency_with_losses(input_tensor, losses, seed_input, wrt_tensor=None,
                                   grad_modifier='absolute', input_indices=0):
    """Generates an attention heatmap over the `seed_input` by using positive gradients of `input_tensor`
    with respect to weighted `losses`.

    This function is intended for advanced use cases where a custom loss is desired. For common use cases,
    refer to `visualize_class_saliency` or `visualize_regression_saliency`.

    For a full description of saliency, see the paper:
    [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps]
    (https://arxiv.org/pdf/1312.6034v2.pdf)

    Args:
        input_tensor: An input tensor or list of input tensor.
            The shape of an input tensor is `(samples, channels, image_dims...)` if `image_data_format=
            channels_first`, Or it's `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model inputs for which activation map needs to be visualized.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensors.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). By default `absolute`
            value of gradients are used. To visualize positive or negative gradients, use `relu` and `negate`
            respectively. (Default value = 'absolute')
        input_indices: A index or a list of index.
            This is the index of visualize target within `wrt_tensor`,
            but when `wrt_tensor` is None, it's `input_tensor`. (Default value = 0)
    Returns:
        The normalized gradients of `seed_input` with respect to weighted `losses`.
        When `input_indices` is a number, returned a gradients.
        But, when `input_indices` is a list of number, returned a list of gradients.
    """
    opt = Optimizer(input_tensor, losses, wrt_tensors=wrt_tensor, norm_grads=False)
    opt_result = opt.minimize(seed_inputs=seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    saliency_maps = []
    for i in utils.listify(input_indices):
        if i < len(opt_result):
            _, grads, _ = opt_result[i]
            grads = np.max(grads, axis=channel_idx)
            grads = utils.normalize(grads)[0]
            saliency_maps.append(grads)
        else:
            raise ValueError('# TODO')

    if isinstance(input_indices, list):
        return saliency_maps
    else:
        return saliency_maps[input_indices]


def visualize_saliency(model, layer_idx, filter_indices, seed_input, wrt_tensor=None,
                       backprop_modifier=None, grad_modifier='absolute', input_indices=0):
    """Generates an attention heatmap over the `seed_input` for maximizing `filter_indices`
    output in the given `layer_idx`.

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
        seed_input: The model inputs for which activation map needs to be visualized.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). By default `absolute`
            value of gradients are used. To visualize positive or negative gradients, use `relu` and `negate`
            respectively. (Default value = 'absolute')
        input_indices: A index or a list of index.
            This is the index of visualize target within `wrt_tensor`,
            but when `wrt_tensor` is None, it's model.inputs. (Default value = 0)

    Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
        When `input_indices` is a number, returned a gradients.
        But, when `input_indices` is a list of number, returned a list of gradients.
    """
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    # `ActivationMaximization` loss reduces as outputs get large, hence negative gradients indicate the direction
    # for increasing activations. Multiply with -1 so that positive gradients indicate increase instead.
    losses = [(ActivationMaximization(model.layers[layer_idx], filter_indices), -1)]
    return visualize_saliency_with_losses(model.inputs, losses, seed_input, wrt_tensor,
                                          grad_modifier, input_indices)


def visualize_cam_with_losses(input_tensor, losses, seed_input, penultimate_layer, grad_modifier=None, input_indices=0):
    """Generates a gradient based class activation map (CAM) by using positive gradients of `input_tensor`
    with respect to weighted `losses`.

    For details on grad-CAM, see the paper:
    [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization]
    (https://arxiv.org/pdf/1610.02391v1.pdf).

    Unlike [class activation mapping](https://arxiv.org/pdf/1512.04150v1.pdf), which requires minor changes to
    network architecture in some instances, grad-CAM has a more general applicability.

    Compared to saliency maps, grad-CAM is class discriminative; i.e., the 'cat' explanation exclusively highlights
    cat regions and not the 'dog' region and vice-versa.

    Args:
        input_tensor: An input tensor or list of input tensor.
            The shape of an input tensor is `(samples, channels, image_dims...)` if `image_data_format=
            channels_first`, Or it's `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model inputs for which activation map needs to be visualized.
        penultimate_layer: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            with respect to filter output.
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)
        input_indices: A index or a list of index.
            This is the index that specifies the `sheed_input` to overlay the cam's heatmap.
            (Default value = 0)

    Returns:
        The normalized gradients of `seed_input` with respect to weighted `losses`.
        When `input_indices` is a number, returned a gradients.
        But, when `input_indices` is a list of number, returned a list of gradients.
    """
    penultimate_output = penultimate_layer.output
    opt = Optimizer(input_tensor, losses, wrt_tensors=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)[0]

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grads = grads / (np.max(grads) + K.epsilon())

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output.
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    # Generate heatmap by computing weight * output over feature maps
    output_dims = utils.get_img_shape(penultimate_output)[2:]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights):
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    # ReLU thresholding to exclude pattern mismatch information (negative gradients).
    heatmap = np.maximum(heatmap, 0)

    heatmaps = []
    for i in utils.listify(input_indices):
        if i < len(input_tensor):
            # The penultimate feature map size is definitely smaller than input image.
            input_dims = utils.get_img_shape(input_tensor[i])[2:]

            # Figure out the zoom factor.
            zoom_factor = [i / (j * 1.0) for i, j in iter(zip(input_dims, output_dims))]
            heatmap = zoom(heatmap, zoom_factor)
            heatmap = utils.normalize(heatmap)
            heatmaps.append(heatmap)
        else:
            raise ValueError('# TODO')

    if isinstance(input_indices, list):
        return heatmaps
    else:
        return heatmaps[input_indices]


def visualize_cam(model, layer_idx, filter_indices, seed_input, penultimate_layer_idx=None,
                  backprop_modifier=None, grad_modifier=None, input_indices=0):
    """Generates a gradient based class activation map (grad-CAM) that maximizes the outputs of
    `filter_indices` in `layer_idx`.

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
        seed_input: The input image for which activation map needs to be visualized.
        penultimate_layer_idx: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            wrt filter output. If not provided, it is set to the nearest penultimate `Conv` or `Pooling` layer.
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)
        input_indices: A index or a list of index.
            This is the index that specifies the `sheed_input` to overlay the cam's heatmap.
            (Default value = 0)

     Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image indicating the input regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
        When `input_indices` is a number, returned a heatmap.
        But, when `input_indices` is a list of number, returned a list of heatmap.
    """
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    penultimate_layer = _find_penultimate_layer(model, layer_idx, penultimate_layer_idx)

    # `ActivationMaximization` outputs negative gradient values for increase in activations. Multiply with -1
    # so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    return visualize_cam_with_losses(model.inputs, losses, seed_input, penultimate_layer, grad_modifier, input_indices)
