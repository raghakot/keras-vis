import cv2
import numpy as np
from keras import backend as K

from losses import ActivationMaximization
from optimizer import Optimizer
from regularizers import TotalVariation, LPNorm
from utils import utils


def _get_num_filters(layer):
    """
    Returns: Total number of filters within this layer
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


def visualize_saliency(img, layer, filter_indices,
                       seed_img, overlay=True):
    """Generates a heatmap indicating the pixels that contributed the most towards
    maximizing `filter_indices` output in the given `layer`.

    For example, if you wanted to visualize the which pixels contributed to classifying an image as 'bird', say output
    index 22 on final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

    Alternatively one could use `filter_indices = [22, 23]` and hope to see image regions that are common to output
    categories 22, 23 to show up in the heatmap.

    Args:
        img: 4D input image tensor with shape: `(samples, channels, rows, cols)` if dim_ordering='th'
            or `(samples, rows, cols, channels)` if dim_ordering='tf'.
        layer: The `keras.Layer` layer whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

            If you are visualizing final `keras.layers.Dense` layer, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.

        seed_img: The input image for which activation map needs to be visualized.
        overlay: If true, overlays the heatmap over the original image (Default value = True)

    Returns:
        The heatmap image indicating image regions that, when changed, would contribute the most towards maximizing
        a the filter output.
    """

    losses = [
        (ActivationMaximization(layer, filter_indices), 1)
    ]
    opt = Optimizer(img, losses)
    _, grads = opt.minimize(max_iter=1, verbose=True, jitter=0, seed_img=seed_img)

    s, c, w, h = utils.get_img_indices()
    grads = np.max(np.abs(grads), axis=c, keepdims=True)

    # Smoothen activation map
    grads = utils.deprocess_image(grads[0])
    grads /= np.max(grads)

    # Convert to heatmap and zero out low probabilities for a cleaner output.
    heatmap = cv2.applyColorMap(cv2.GaussianBlur(grads * 255, (3, 3), 0), cv2.COLORMAP_JET)
    heatmap[np.where(grads <= 0.2)] = 0

    if overlay:
        return cv2.addWeighted(seed_img, 1, heatmap, 0.5, 0)
    else:
        return heatmap


def visualize_activation(img, layer, filter_indices=None,
                         seed_img=None, max_iter=200,
                         act_max_weight=1, lp_norm_weight=10, tv_weight=10, verbose=False,
                         show_filter_idx_text=True, idx_label_map=None, cols=5):
    """Generates stitched input image(s) over all `filter_indices` in the given `layer` that maximize
    the filter output activation.

    For example, if you wanted to visualize the input image that would maximize the output index 22, say on
    final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.

    If `filter_indices = [22, 23]`, then a stitched image comprising of two images are generated, each
    corresponding to the entry in `filter_indices`.

    Args:
        img: 4D input image tensor with shape: `(samples, channels, rows, cols)` if dim_ordering='th'
            or `(samples, rows, cols, channels)` if dim_ordering='tf'.
        layer: The `keras.Layer` layer whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
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
        max_iter: The maximum number of gradient descent iterations. (Default value = 200)
        act_max_weight: The weight param for `ActivationMaximization` loss. Not used if 0 or None. (Default value = 1)
        lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None. (Default value = 10)
        tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None. (Default value = 10)
        verbose: Shows verbose loss output for each filter. (Default value = False)
                Very useful to estimate loss weight factor. (Default value = True)
        show_filter_idx_text: Adds filter_idx text to the image if set to True. (Default value = True)
            If the entry in `filter_indices` is an array, then comma separated labels are generated.
        idx_label_map: Map of filter_idx to text label. If not None, this map is used to translate filter_idx
            to text value when show_filter_idx_text = True. (Default value = None)
        cols: Max number of image cols. New row is created when number of images exceed the column size.
            (Default value = 5)

    Returns:
        Stitched image output visualizing input images that maximize the filter output(s). (Default value = 10)
    """
    if filter_indices is None:
        filter_indices = np.arange(_get_num_filters(layer))

    imgs = []
    for i, idx in enumerate(filter_indices):
        indices = idx if isinstance(idx, list) else [idx]

        losses = [
            (ActivationMaximization(layer, indices), act_max_weight or 0),
            (LPNorm(), lp_norm_weight or 0),
            (TotalVariation(), tv_weight or 0)
        ]

        opt = Optimizer(img, losses)
        print('Working on filter {}/{}'.format(i + 1, len(filter_indices)))
        opt_img, g = opt.minimize(seed_img=seed_img, max_iter=max_iter, verbose=verbose)

        # Add filter text to image if applicable.
        if show_filter_idx_text:
            label = None
            if idx_label_map:
                label = ', '.join([idx_label_map.get(i) for i in indices])
            if label is None:
                label = "Filter {}".format(', '.join([str(i) for i in indices]))
            cv2.putText(opt_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        imgs.append(opt_img)

    return utils.stitch_images(imgs, cols=cols)
