from __future__ import absolute_import


from .activation_maximization import visualize_activation_with_losses
from .activation_maximization import visualize_activation

from .saliency import visualize_saliency_with_losses
from .saliency import visualize_saliency
from .saliency import visualize_cam_with_losses
from .saliency import visualize_cam

from keras import backend as K


def get_num_filters(layer):
    """Determines the number of filters within the given `layer`.

    Args:
        layer: The keras layer to use.

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
        raise ValueError('`array1` and `array2` must have the same shapes')

    return (array1 * alpha + array2 * (1. - alpha)).astype(array1.dtype)
