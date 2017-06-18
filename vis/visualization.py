from __future__ import absolute_import

import numpy as np
import matplotlib.cm as cm
import pprint

from scipy.misc import imresize
from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras import backend as K

from .losses import ActivationMaximization
from .optimizer import Optimizer
from .regularizers import TotalVariation, LPNorm
from .modifiers import Jitter
from .utils import utils


_DEFAULT_IMG_MODIFIERS = [
    Jitter()
]


def get_num_filters(layer):
    """
    Returns: Total number of filters within `layer`.
        For `keras.layers.Dense` layer, this is the total number of outputs.
    """
    # Handle layers with no channels.
    if K.ndim(layer.output) == 2:
        return K.int_shape(layer.output)[-1]

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    return K.int_shape(layer.output)[channel_idx]


def visualize_activation(model, layer_idx, filter_indices=None, seed_img=None,
                         act_max_weight=1, lp_norm_weight=10, tv_weight=10,
                         **optimizer_params):
    """Generates stitched input image(s) over all `filter_indices` in the given `layer` that maximize
    the filter output activation.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)` 
            if data_format='channels_first' or `(samples, image_dims..., channels)` if data_format='channels_last'.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)

            An input image is generated for each entry in `filter_indices`. The entry can also be an array.
            For example, `filter_indices = [[1, 2], 3, [4, 5, 6]]` would generate three input images. The first one
            would maximize output of filters 1, 2, 3 jointly. A fun use of this might be to generate a dog-fish
            image by maximizing 'dog' and 'fish' output in final `Dense` layer.

            For `keras.layers.Dense` layers, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_img: Seeds the optimization with a starting image. Initialized with a random value when set to None.
            (Default value = None)
        act_max_weight: The weight param for `ActivationMaximization` loss. Not used if 0 or None. (Default value = 1)
        lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None. (Default value = 10)
        tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None. (Default value = 10)
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer.md##optimizerminimize). Will default to
            reasonable values when required keys are not found.

    Example:
        If you wanted to visualize the input image that would maximize the output index 22, say on
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        If `filter_indices = [22, 23]`, then it should generate an input image that shows features of both classes.

    Returns:
        Stitched image output visualizing input images that maximize the filter output(s). (Default value = 10)
    """
    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    # Default optimizer kwargs.
    optimizer_params_default = {
        'seed_img': seed_img,
        'max_iter': 200,
        'verbose': False,
        'image_modifiers': _DEFAULT_IMG_MODIFIERS
    }
    optimizer_params_default.update(optimizer_params)
    optimizer_params = optimizer_params_default

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight),
        (LPNorm(model.input), lp_norm_weight),
        (TotalVariation(model.input), tv_weight)
    ]

    opt = Optimizer(model.input, losses)
    img = opt.minimize(**optimizer_params)[0]
    return img


def visualize_saliency(model, layer_idx, filter_indices,
                       seed_img, alpha=0.5):
    """Generates an attention heatmap over the `seed_img` for maximizing `filter_indices` output in the given `layer`.
     For a full description of saliency, see the paper:
     [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034v2.pdf)

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)` 
            if data_format='channels_first' or `(samples, image_dims..., channels)` if data_format='channels_last'.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_img: The input image for which activation map needs to be visualized.
        alpha: The alpha value of image as overlayed onto the heatmap. 
            This value needs to be between [0, 1], with 0 being heatmap only to 1 being image only (Default value = 0.5)

    Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image, overlayed with `seed_img` using `alpha`, indicating image regions that, when changed, 
        would contribute the most towards maximizing the output of `filter_indices`.
    """
    if alpha < 0. or alpha > 1.:
        raise ValueError("`alpha` needs to be between [0, 1]")

    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), 1)
    ]
    opt = Optimizer(model.input, losses, norm_grads=False)
    grads = opt.minimize(max_iter=1, verbose=False, seed_img=seed_img)[1]

    # We are minimizing loss as opposed to maximizing output as with the paper.
    # So, negative gradients here mean that they reduce loss, maximizing class probability.
    grads *= -1

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    grads = np.max(np.abs(grads), axis=channel_idx)

    # Normalize and create heatmap
    grads = utils.normalize(grads)
    heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    heatmap = np.uint8(seed_img * alpha + heatmap * (1. - alpha))
    return heatmap[0]


def visualize_cam(model, layer_idx, filter_indices,
                  seed_img, penultimate_layer_idx=None, alpha=0.5):
    """Generates a gradient based class activation map (CAM) as described in paper
    [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391v1.pdf).
    Unlike [class activation mapping](https://arxiv.org/pdf/1512.04150v1.pdf), which requires minor changes to
    network architecture in some instances, grad-CAM has a more general applicability.

    Compared to saliency maps, grad-CAM is class discriminative; i.e., the 'cat' explanation exclusively highlights
    cat regions and not the 'dog' region and vice-versa.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)` 
            if data_format='channels_first' or `(samples, image_dims..., channels)` if data_format='channels_last'.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_img: The input image for which activation map needs to be visualized.
        penultimate_layer_idx: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            wrt filter output. If not provided, it is set to the nearest penultimate `Convolutional` or `Pooling` layer.
        alpha: The alpha value of image as overlayed onto the heatmap. 
            This value needs to be between [0, 1], with 0 being heatmap only to 1 being image only (Default value = 0.5)

     Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Notes:
        This technique deprecates occlusion maps as it gives similar results, but with one-pass gradient computation
        as opposed inefficient sliding window approach.

    Returns:
        The heatmap image, overlayed with `seed_img` using `alpha`, indicating image regions that, when changed, 
        would contribute the most towards maximizing the output of `filter_indices`.
    """
    if alpha < 0. or alpha > 1.:
        raise ValueError("`alpha` needs to be between [0, 1]")

    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    # Search for the nearest penultimate `Convolutional` or `Pooling` layer.
    if penultimate_layer_idx is None:
        for idx, layer in utils.reverse_enumerate(model.layers[:layer_idx-1]):
            if isinstance(layer, (_Conv, _Pooling1D, _Pooling2D, _Pooling3D)):
                penultimate_layer_idx = idx
                break

    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Convolution` or `Pooling` '
                         'layer for layer_idx: {}'.format(layer_idx))
    assert penultimate_layer_idx < layer_idx

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), 1)
    ]

    penultimate_output = model.layers[penultimate_layer_idx].output
    opt = Optimizer(model.input, losses, wrt=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_img, max_iter=1, verbose=False)

    # We are minimizing loss as opposed to maximizing output as with the paper.
    # So, negative gradients here mean that they reduce loss, maximizing class probability.
    grads *= -1

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grads = utils.normalize(grads)

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output
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

    # The penultimate feature map size is definitely smaller than input image.
    input_dims = utils.get_img_shape(model.input)[2:]
    heatmap = imresize(heatmap, input_dims, interp='bicubic', mode='F')

    # ReLU thresholding, normalize between (0, 1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Create jet heatmap.
    heatmap_colored = np.uint8(cm.jet(heatmap)[..., :3] * 255)
    heatmap_colored = np.uint8(seed_img * alpha + heatmap_colored * (1. - alpha))
    return heatmap_colored
