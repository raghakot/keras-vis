from __future__ import absolute_import

import numpy as np
import matplotlib.cm as cm
import pprint

from scipy.misc import imresize
from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras import backend as K

from .losses import ActivationMaximization, RegressionTarget
from .optimizer import Optimizer
from .regularizers import TotalVariation, LPNorm
from .modifiers import Jitter
from .utils import utils


_DEFAULT_IMG_MODIFIERS = [
    Jitter()
]


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


def get_num_filters(layer):
    """Determines the number of filters within the give `layer`.

    Returns:
        Total number of filters within `layer`.
        For `keras.layers.Dense` layer, this is the total number of outputs.
    """
    # Handle layers with no channels.
    if K.ndim(layer.output) == 2:
        return K.int_shape(layer.output)[-1]

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    return K.int_shape(layer.output)[channel_idx]


def overlay(array1, array2, alpha=0.5):
    """Overlays `array1` onto `array2` with `alpha` blending.

    Args:
        array1: The first numpy array.
        array2: The second numpy array.
        alpha: The alpha value of `array1` as overlayed onto `array2`. This value needs to be between [0, 1],
            with 0 being `array2` only to 1 being `array1` only (Default value = 0.5).

    Returns:
        The `array1`, overlayed with `array2` using `alpha` blending.
    """
    if alpha < 0. or alpha > 1.:
        raise ValueError("`alpha` needs to be between [0, 1]")
    if array1.shape != array2.shape:
        raise ValueError('`array1` and `array2` mush have the same shapes')

    return (array1 * alpha + array2 * (1. - alpha)).astype(array1.dtype)


def visualize_activation(input_tensor, losses, seed_input=None, input_range=(0, 255), **optimizer_params):
    """Generates the `input_tensor` that minimizes the weighted `losses`.

    This function is intended for advanced use cases where a custom loss is desired. For common use cases,
    refer to `visualize_class_activation` or `visualize_regression_activation`.

    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: Seeds the optimization with a starting image. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer.md##optimizerminimize). Will default to
            reasonable values when required keys are not found.

    Returns:
        The model input that minimizes the weighted `losses`.
    """
    # Default optimizer kwargs.
    optimizer_params_default = {
        'seed_input': seed_input,
        'max_iter': 200,
        'verbose': False,
        'image_modifiers': _DEFAULT_IMG_MODIFIERS
    }
    optimizer_params_default.update(optimizer_params)
    optimizer_params = optimizer_params_default

    opt = Optimizer(input_tensor, losses)
    img = opt.minimize(**optimizer_params)[0]

    # If range has integer numbers, cast to 'uint8'
    if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        img = np.clip(img, input_range[0], input_range[1]).astype('uint8')

    if K.image_data_format() == 'channels_first':
        img = np.moveaxis(img, 0, -1)
    return img


def visualize_class_activation(model, layer_idx, filter_indices=None, seed_input=None,
                               input_range=(0, 255),
                               act_max_weight=1, lp_norm_weight=10, tv_weight=10,
                               **optimizer_params):
    """Generates the model input that maximizes the output of all `filter_indices` in the given `layer_idx`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)

            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_input: Seeds the optimization with a starting input. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        act_max_weight: The weight param for `ActivationMaximization` loss. Not used if 0 or None. (Default value = 1)
        lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None. (Default value = 10)
        tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None. (Default value = 10)
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer.md##optimizerminimize). Will default to
            reasonable values when required keys are not found.

    Example:
        If you wanted to visualize the input image that would maximize the output index 22, say on
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer_idx = dense_layer_idx`.

        If `filter_indices = [22, 23]`, then it should generate an input image that shows features of both classes.

    Returns:
        The model input that maximizes the output of `filter_indices` in the given `layer_idx`.
    """
    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight),
        (LPNorm(model.input), lp_norm_weight),
        (TotalVariation(model.input), tv_weight)
    ]
    return visualize_activation(model.input, losses, seed_input, input_range, **optimizer_params)


def visualize_regression_activation(model, layer_idx, output_indices, targets, seed_img=None,
                                    input_range=(0, 255),
                                    reg_target_weight=1., lp_norm_weight=1e-3, tv_weight=1e-3,
                                    **optimizer_params):
    """Generates a model input that drives the outputs of `output_indices` in the given `layer_idx` to the
    corresponding regression `targets`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose output indices needs to be considered.
        output_indices: Output indices within the layer for the corresponding regression `targets`.
        targets: The regression targets for the corresponding `output_indices`.
        seed_img: Seeds the optimization with a starting image. Initialized with a random value when set to None.
            (Default value = None)
        input_range: Specifies the input range as a `(min, max)` tuple. This is used to rescale the
            final optimized input to the given range. (Default value=(0, 255))
        reg_target_weight: The weight param for `RegressionTarget` loss. Not used if 0 or None.
            (Default value = 1)
        lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None.
            (Default value = 1e-3)
        tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None.
            (Default value = 1e-3)
        optimizer_params: The **kwargs for optimizer [params](vis.optimizer.md##optimizerminimize). Will default to
            reasonable values when required keys are not found.

    Example:
        Consider a model with continuous regression output such as the self driving car.

        If you wanted to visualize the input image that would cause the final `Dense` layer output_index 0 to output
        45 degrees, then you would set `output_indices = 0`, `layer_idx = dense_layer_idx` and `targets = 45`

        Suppose this model has two regression outputs, one for the steering angle and another for acceleration.
        Setting `output_indices = [0, 1]`, `layer_idx = dense_layer_idx` and `targets = [45, -5]` would generate
        the model input that would cause the steering angle to increase but the acceleration to decrease.

    Returns:
        The model input that causes regression `output_indices` outputs to approach their corresponding `targets`.
    """
    output_indices = utils.listify(output_indices)
    print("Working on output indices: {}".format(pprint.pformat(output_indices)))

    losses = [
        (RegressionTarget(model.layers[layer_idx], output_indices, targets), reg_target_weight),
        (LPNorm(model.input), lp_norm_weight),
        (TotalVariation(model.input), tv_weight)
    ]
    return visualize_activation(model.input, losses, seed_img, input_range, **optimizer_params)


def visualize_saliency(input_tensor, losses, seed_input):
    """Generates an attention heatmap over the `seed_input` by using positive gradients of `input_tensor`
    with respect to weighted `losses`.

    This function is intended for advanced use cases where a custom loss is desired. For common use cases,
    refer to `visualize_class_saliency` or `visualize_regression_saliency`.

    For a full description of saliency, see the paper:
    [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034v2.pdf)

    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model input for which activation map needs to be visualized.

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards minimizing
        weighted `losses`.
    """
    opt = Optimizer(input_tensor, losses, norm_grads=False)
    grads = opt.minimize(max_iter=1, verbose=False, seed_input=seed_input)[1]

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    grads = np.max(np.abs(grads), axis=channel_idx)

    # Normalize and create heatmap.
    grads = utils.normalize(grads)
    return np.uint8(cm.jet(grads)[..., :3] * 255)[0]


def visualize_class_saliency(model, layer_idx, filter_indices, seed_input):
    """Generates an attention heatmap over the `seed_input` for maximizing `filter_indices`
    output in the given `layer_idx`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_input: The model input for which activation map needs to be visualized.

    Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
    """
    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    # `ActivationMaximization` loss reduces as outputs get large, hence negative gradients indicate the direction
    # for increasing activations. Multiply with -1 so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    return visualize_saliency(model.input, losses, seed_input)


def visualize_regression_saliency(model, layer_idx, output_indices, targets, seed_input):
    """Generates an attention heatmap over the `seed_input` for driving the outputs of `output_indices`
    in the given `layer_idx` to the corresponding regression `targets`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        output_indices: Output indices within the layer for the corresponding regression `targets`.
        targets: The regression targets for the corresponding `output_indices`.
        seed_input: The model input for which activation map needs to be visualized.

    Example:
        Consider a model with continuous regression output such as the self driving car.

        If you wanted to visualize the attention over input image that would cause the final `Dense` layer output_index
        0 to output 45 degrees, then you would set `output_indices = 0`, `layer_idx = dense_layer_idx` and `targets = 45`.

        Suppose this model has two regression outputs, one for the steering angle and another for acceleration.
        Setting `output_indices = [0, 1]`, `layer_idx = dense_layer_idx` and `targets = [45, -5]` would generate
        the attention heatmap over input that would cause the steering angle to increase and acceleration to decrease.

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards `output_indices`
        outputs to approach their corresponding `targets`.
    """
    output_indices = utils.listify(output_indices)
    print("Working on filters: {}".format(pprint.pformat(output_indices)))

    # `RegressionTarget` loss reduces as outputs approach target, hence negative gradients indicate this direction.
    # Multiply with -1 so that positive gradients indicate this direction instead.
    losses = [
        (RegressionTarget(model.layers[layer_idx], output_indices, targets), -1)
    ]
    return visualize_saliency(model.input, losses, seed_input)


def visualize_cam(input_tensor, losses, seed_input, penultimate_layer):
    """Generates a gradient based class activation map (CAM) by using positive gradients of `input_tensor`
    with respect to weighted `losses`.

    For details on grad-CAM, see the paper:
    [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391v1.pdf).

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

    Notes:
        This technique deprecates occlusion maps as it gives similar results, but with one-pass gradient computation
        as opposed inefficient sliding window approach.

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards minimizing the
        weighted `losses`.
    """
    penultimate_output = penultimate_layer.output
    opt = Optimizer(input_tensor, losses, wrt_tensor=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=1, verbose=False)

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
    heatmap = imresize(heatmap, input_dims, interp='bicubic', mode='F')

    # Normalize and create heatmap.
    heatmap = utils.normalize(heatmap)
    return np.uint8(cm.jet(heatmap)[..., :3] * 255)


def visualize_class_cam(model, layer_idx, filter_indices,
                        seed_input, penultimate_layer_idx=None):
    """Generates a gradient based class activation map (grad-CAM) that maximizes the outputs of
    `filter_indices` in `layer_idx`.

    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_input: The input image for which activation map needs to be visualized.
        penultimate_layer_idx: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            wrt filter output. If not provided, it is set to the nearest penultimate `Conv` or `Pooling` layer.

     Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Notes:
        This technique deprecates occlusion maps as it gives similar results, but with one-pass gradient computation
        as opposed inefficient sliding window approach.

    Returns:
        The heatmap image indicating the input regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
    """
    penultimate_layer = _find_penultimate_layer(model, layer_idx, penultimate_layer_idx)

    # `ActivationMaximization` outputs negative gradient values for increase in activations. Multiply with -1
    # so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    return visualize_cam(model.input, losses, seed_input, penultimate_layer)


def visualize_regression_cam(model, layer_idx, output_indices, targets, seed_input, penultimate_layer_idx=None):
    """Generates gradient based class activation map (grad-CAM) over the `seed_input` for driving the outputs of
    `output_indices` in the given `layer_idx` to the corresponding regression `targets`.

   Args:
       model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
       layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
       output_indices: Output indices within the layer for the corresponding regression `targets`.
       targets: The regression targets for the corresponding `output_indices`.
       seed_input: The input image for which activation map needs to be visualized.
       penultimate_layer_idx: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            wrt filter output. If not provided, it is set to the nearest penultimate `Conv` or `Pooling` layer.

   Example:
       Consider a model with continuous regression output such as the self driving car.

       If you wanted to visualize the attention over input image that would cause the final `Dense` layer output_index
       0 to output 45 degrees, then you would set `output_indices = 0`, `layer_idx = dense_layer_idx` and `targets = 45`.

       Suppose this model has two regression outputs, one for the steering angle and another for acceleration.
       Setting `output_indices = [0, 1]`, `layer_idx = dense_layer_idx` and `targets = [45, -5]` would generate
       the attention heatmap over input that would cause the steering angle to increase and acceleration to decrease.

   Returns:
       The heatmap image indicating the `seed_input` regions whose change would most contribute towards `output_indices`
       outputs to approach their corresponding `targets`.
   """
    penultimate_layer = _find_penultimate_layer(model, layer_idx, penultimate_layer_idx)

    # `RegressionTarget` loss reduces as outputs approach target, hence negative gradients indicate this direction.
    # Multiply with -1 so that positive gradients indicate this direction instead.
    losses = [
        (RegressionTarget(model.layers[layer_idx], output_indices, targets), -1)
    ]
    return visualize_cam(model.input, losses, seed_input, penultimate_layer)
