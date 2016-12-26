import numpy as np
from .utils import utils


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

    def __init__(self):
        # These indices are required to handle difference in 'th'/'tf' dim orderings.
        self._ch_idx, self._row_idx, self._col_idx = utils.get_img_indices()[1:]

    @property
    def channel_idx(self):
        """Returns the proper channel index based on image dim ordering.
        """
        return self._ch_idx

    @property
    def row_idx(self):
        """Returns the proper row index based on image dim ordering.
        """
        return self._row_idx

    @property
    def col_idx(self):
        """Returns the proper col index based on image dim ordering.
        """
        return self._col_idx

    def pre(self, img):
        """Implement pre gradient descent update modification to the image. If pre-processing is not desired,
        simply ignore the implementation. It returns the unmodified `img` by default.

        Properties `self.channel_idx`, `self.row_idx`, `self.col_idx` can be used to handle 'th'/'tf'
        image dim ordering differences.

        Args:
            img: 4D numpy array with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'

        Returns:
            The modified pre image.
        """
        return img

    def post(self, img):
        """Implement post gradient descent update modification to the image. If post-processing is not desired,
        simply ignore the implementation. It returns the unmodified `img` by default.

        Properties `self.channel_idx`, `self.row_idx`, `self.col_idx` can be used to handle 'th'/'tf'
        image dim ordering differences.

        Args:
            img: 4D numpy array with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'

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
        self.jx = 0
        self.jy = 0

    def pre(self, img):
        self.jx, self.jy = np.random.randint(-self.jitter, self.jitter+1, 2)
        return np.roll(np.roll(img, self.jx, self.row_idx), self.jy, self.col_idx)

    def post(self, img):
        # Un-shift the jitter.
        return np.roll(np.roll(img, -self.jx, self.row_idx), -self.jy, self.col_idx)
