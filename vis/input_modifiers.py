from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import shift
from .utils import utils
from keras import backend as K


class InputModifier(object):
    """Abstract class for defining an input modifier. An input modifier can be used with the
    [Optimizer.minimize](vis.optimizer#optimizerminimize) to make `pre` and `post` changes to the optimized input
    during the optimization process.

    ```python
    modifier.pre(seed_input)
    # gradient descent update to img
    modifier.post(seed_input)
    ```
    """

    def pre(self, inp):
        """Implement pre gradient descent update modification to the input. If pre-processing is not desired,
        simply ignore the implementation. It returns the unmodified `inp` by default.

        Args:
            inp: An N-dim numpy array of shape: `(samples, channels, image_dims...)` if `image_data_format=
                channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.

        Returns:
            The modified pre input.
        """
        return inp

    def post(self, inp):
        """Implement post gradient descent update modification to the input. If post-processing is not desired,
        simply ignore the implementation. It returns the unmodified `inp` by default.

        Args:
            inp: An N-dim numpy array of shape: `(samples, channels, image_dims...)` if `image_data_format=
                channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.

        Returns:
            The modified post input.
        """
        return inp


class Jitter(InputModifier):

    def __init__(self, jitter=0.05):
        """Implements an input modifier that introduces random jitter in `pre`.
        Jitter has been shown to produce crisper activation maximization images.

        Args:
            jitter: The amount of jitter to apply, scalar or sequence.
                If a scalar, same jitter is applied to all image dims. If sequence, `jitter` should contain a value
                per image dim.

                A value between `[0., 1.]` is interpreted as a percentage of the image dimension. (Default value: 0.05)
        """
        super(Jitter, self).__init__()
        self.jitter = np.array(utils.listify(jitter))
        if np.any(jitter < 0.):
            raise ValueError('Jitter value should be positive')
        self._processed = False

    def _process_jitter_values(self, image_dims):
        if len(self.jitter) == 1:
            self.jitter = np.repeat(self.jitter, len(image_dims))
        if len(self.jitter) != len(image_dims):
            raise RuntimeError('Jitter {}, does not match the number of image dims: {}'
                               .format(self.jitter, len(image_dims)))

        # Convert percentage to absolute values.
        for i, jitter_value in enumerate(self.jitter):
            if jitter_value < 1.:
                self.jitter[i] = image_dims[i] * jitter_value

        # Round to int.
        self.jitter = np.int32(self.jitter)
        self._processed = True

    def pre(self, img):
        if not self._processed:
            image_dims = utils.get_img_shape(img)[2:]
            self._process_jitter_values(image_dims)

        dim_offsets = [np.random.randint(-value, value + 1) for value in self.jitter]
        if K.image_data_format() == 'channels_first':
            shift_vector = np.array([0, 0] + dim_offsets)
        else:
            shift_vector = np.array([0] + dim_offsets + [0])

        return shift(img, shift_vector, mode='wrap', order=0)
