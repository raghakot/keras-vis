import cv2
import numpy as np
import pprint

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import _Pooling2D
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
    # For all other layers it is 4
    isDense = K.ndim(layer.output) == 2

    if isDense:
        return layer.output.shape[1]
    else:
        if K.image_dim_ordering() == 'th':
            return layer.output._keras_shape[1]
        else:
            return layer.output._keras_shape[3]


def visualize_activation(model, layer_idx, filter_indices=None,
                         seed_img=None, text=None,
                         act_max_weight=1, lp_norm_weight=10, tv_weight=10,
                         **optimizer_params):
    """Generates stitched input image(s) over all `filter_indices` in the given `layer` that maximize
    the filter output activation.

    Args:
        model: The `keras.models.Model` instance. Model input is expected to be a 4D image input of shape:
            `(samples, channels, rows, cols)` if dim_ordering='th' or `(samples, rows, cols, channels)` if dim_ordering='tf'.
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
        text: The text to overlay on top of the generated image. (Default Value = None)
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
    if text:
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return img


def visualize_saliency(model, layer_idx, filter_indices,
                       seed_img, text=None, overlay=True):
    """Generates an attention heatmap over the `seed_img` for maximizing `filter_indices` output in the given `layer`.
     For a full description of saliency, see the paper:
     [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034v2.pdf)

    Args:
        model: The `keras.models.Model` instance. Model input is expected to be a 4D image input of shape:
            `(samples, channels, rows, cols)` if dim_ordering='th' or `(samples, rows, cols, channels)` if dim_ordering='tf'.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_img: The input image for which activation map needs to be visualized.
        text: The text to overlay on top of the generated image. (Default Value = None)
        overlay: If true, overlays the heatmap over the original image (Default value = True)

    Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Returns:
        The heatmap image indicating image regions that, when changed, would contribute the most towards maximizing
        a the filter output.
    """
    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), 1)
    ]
    opt = Optimizer(model.input, losses)
    grads = opt.minimize(max_iter=1, verbose=False, jitter=0, seed_img=seed_img)[1]

    # We are minimizing loss as opposed to maximizing output as with the paper.
    # So, negative gradients here mean that they reduce loss, maximizing class probability.
    grads *= -1

    s, c, row, col = utils.get_img_indices()
    grads = np.max(np.abs(grads), axis=c, keepdims=True)

    # Smoothen activation map
    grads = utils.deprocess_image(grads[0])
    grads /= np.max(grads)

    # Convert to heatmap and zero out low probabilities for a cleaner output.
    heatmap = cv2.applyColorMap(cv2.GaussianBlur(grads * 255, (3, 3), 0), cv2.COLORMAP_JET)
    heatmap[np.where(grads <= 0.2)] = 0

    if overlay:
        heatmap = cv2.addWeighted(seed_img, 0.5, heatmap, 1, 0)
    if text:
        cv2.putText(heatmap, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return heatmap


def visualize_cam(model, layer_idx, filter_indices,
                  seed_img, penultimate_layer_idx=None,
                  text=None, overlay=True):
    """Generates a gradient based class activation map (CAM) as described in paper
    [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391v1.pdf).
    Unlike [class activation mapping](https://arxiv.org/pdf/1512.04150v1.pdf), which requires minor changes to
    network architecture in some instances, grad-CAM has a more general applicability.

    Compared to saliency maps, grad-CAM is class discriminative; i.e., the 'cat' explanation exclusively highlights
    cat regions and not the 'dog' region and vice-versa.

    Args:
        model: The `keras.models.Model` instance. Model input is expected to be a 4D image input of shape:
            `(samples, channels, rows, cols)` if dim_ordering='th' or `(samples, rows, cols, channels)` if dim_ordering='tf'.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_img: The input image for which activation map needs to be visualized.
        penultimate_layer_idx: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            wrt filter output. If not provided, it is set to the nearest penultimate `Convolutional` or `Pooling` layer.
        text: The text to overlay on top of the generated image. (Default Value = None)
        overlay: If true, overlays the heatmap over the original image (Default value = True)

     Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.

    Notes:
        This technique deprecates occlusion maps as it gives similar results, but with one-pass gradient computation
        as opposed inefficient sliding window approach.

    Returns:
        The heatmap image indicating image regions that, when changed, would contribute the most towards maximizing
        a the filter output.
    """
    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    # Search for the nearest penultimate `Convolutional` or `Pooling` layer.
    if penultimate_layer_idx is None:
        for idx, layer in utils.reverse_enumerate(model.layers[:layer_idx-1]):
            if isinstance(layer, (Convolution2D, _Pooling2D)):
                penultimate_layer_idx = idx
                break

    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Convolution2D` or `Pooling2D` '
                         'layer for layer_idx: {}'.format(layer_idx))
    assert penultimate_layer_idx < layer_idx

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), 1)
    ]

    penultimate_output = model.layers[penultimate_layer_idx].output
    opt = Optimizer(model.input, losses, wrt=penultimate_output)
    _, grads, penultimate_output_value = opt.minimize(seed_img, max_iter=1, verbose=False)

    # We are minimizing loss as opposed to maximizing output as with the paper.
    # So, negative gradients here mean that they reduce loss, maximizing class probability.
    grads *= -1

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output
    s_idx, c_idx, row_idx, col_idx = utils.get_img_indices()
    weights = np.mean(grads, axis=(s_idx, row_idx, col_idx))

    # Generate heatmap by computing weight * output over feature maps
    s, ch, rows, cols = utils.get_img_shape(penultimate_output)
    heatmap = np.ones(shape=(rows, cols), dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * penultimate_output_value[utils.slicer[0, i, :, :]]

    # The penultimate feature map size is definitely smaller than input image.
    s, ch, rows, cols = utils.get_img_shape(model.input)
    heatmap = cv2.resize(heatmap, (cols, rows), interpolation=cv2.INTER_CUBIC)

    # ReLU thresholding, normalize between (0, 1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Convert to heatmap and zero out low probabilities for a cleaner output.
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap_colored[np.where(heatmap <= 0.2)] = 0

    if overlay:
        heatmap_colored = cv2.addWeighted(seed_img, 1, heatmap_colored, 0.5, 0)
    if text:
        cv2.putText(heatmap_colored, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return heatmap_colored
