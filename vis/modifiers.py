from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import shift
from keras import backend as K


class ImageModifier(object):
    """Abstract class for defining an image modifier. An image modifier can be used with the
    [Optimizer.minimize](vis.optimizer/#optimizerminimize) to make `pre` and `post` image changes with the
    gradient descent update step.

    ```python
    modifier.pre(img)
    # gradient descent update to img
    modifier.post(img)
    ```
    """

    def pre(self, img):
        """Implement pre gradient descent update modification to the image. If pre-processing is not desired,
        simply ignore the implementation. It returns the unmodified `img` by default.

        Args:
            img: An N-dim numpy array of shape: `(samples, channels, image_dims...)` if data_format='channels_first' or
                `(samples, image_dims..., channels)` if data_format='channels_last'.

        Returns:
            The modified pre image.
        """
        return img

    def post(self, img):
        """Implement post gradient descent update modification to the image. If post-processing is not desired,
        simply ignore the implementation. It returns the unmodified `img` by default.

        Args:
            img: An N-dim numpy array of shape: `(samples, channels, image_dims...)` if data_format='channels_first' or
                `(samples, image_dims..., channels)` if data_format='channels_last'.

        Returns:
            The modified post image.
        """
        return img


class Jitter(ImageModifier):

    def __init__(self, jitter=16):
        """Implements an image modifier that introduces random jitter in `pre` and undoes in `post`.
        Jitter has been shown to produce crisper activation maximization images.

        Args:
            jitter: Number of pixels to jitter in rows and cols dimensions.
        """
        super(Jitter, self).__init__()
        self.jitter = jitter

    def pre(self, img):
        image_dims = len(img.shape) - 2
        dim_offsets = np.random.randint(-self.jitter, self.jitter+1, image_dims).tolist()

        if K.image_data_format() == 'channels_first':
            shift_vector = np.array([0, 0] + dim_offsets)
        else:
            shift_vector = np.array([0] + dim_offsets + [0])

        return shift(img, shift_vector, mode='wrap', order=0)
