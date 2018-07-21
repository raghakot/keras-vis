from __future__ import absolute_import

import numpy as np
import matplotlib.cm as cm
from scipy.misc import imresize

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras import backend as K

from ..losses import ActivationMaximization
from ..optimizer import Optimizer
from ..backprop_modifiers import get
from ..utils import utils

import cv2


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

def visualize_cam_with_losses(input_tensor, losses,
                              seed_input, penultimate_layer,
                              grad_modifier=None):
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
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model input for which activation map needs to be visualized.
        penultimate_layer: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            with respect to filter output.
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards minimizing the
        weighted `losses`.
    """
    penultimate_output = penultimate_layer.output
    opt = Optimizer(input_tensor, losses, wrt_tensor=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)

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

    # The penultimate feature map size is definitely smaller than input image.
    input_dims = utils.get_img_shape(input_tensor)[2:]
    #heatmap = imresize(heatmap, input_dims, interp='bicubic', mode='F')

    # Normalize and create heatmap.
    heatmap = utils.normalize(heatmap)

    target_size = input_dims

    mapa = heatmap

    mapa_ex = np.zeros((target_size[0], target_size[1], mapa.shape[2]))
    for i in range(mapa.shape[2]):
        mapa_ex[:, :, i] = cv2.resize(mapa[:, :, i],(target_size[1], target_size[0]), interpolation = cv2.INTER_CUBIC)

    mapa_ex2 = np.zeros(target_size)
    for i in range(len(mapa_ex)):
        mapa_ex2[i, :, :] = cv2.resize(mapa_ex[i, :, :],(target_size[2], target_size[1]), interpolation = cv2.INTER_CUBIC)	
    	
    return mapa_ex2


def visualize_cam(model, layer_idx, filter_indices,
                  seed_input, penultimate_layer_idx=None,
                  backprop_modifier=None, grad_modifier=None):
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

     Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image indicating the input regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
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
    return visualize_cam_with_losses(model.input, losses, seed_input, penultimate_layer, grad_modifier)
