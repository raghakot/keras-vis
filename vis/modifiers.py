import numpy as np
from .utils import utils


class ImageModifier(object):
    """Abstract class for defining an image modifier. An image modifier can be used with the
    [Optimizer.minimize](../vis.optimizer/#optimizerminimize) to make `pre` and `post` image changes with the
    gradient descent update step.

    ```python
    modifier.pre(img)
    # gradient descent update to img
    modifier.post(img)
    ```
    """

    def pre(self, img):
        """Implement pre gradient descent update modification to the image.

        Args:
            img: 4D numpy array with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'

        Returns:
            The modified pre image.
        """
        raise NotImplementedError()

    def post(self, img):
        """Implement post gradient descent update modification to the image. If post processing is not desired,
        simply return the unmodified `img`

        Args:
            img: 4D numpy array with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'

        Returns:
            The modified post image.
        """


class Jitter(ImageModifier):

    def __init__(self, jitter=16):
        """Implements an image modifier that introduces random jitter in `pre` and undoes in `post`.
        Jitter has been shown to produce crisper activation maximization images.

        Args:
            jitter: Number of pixels to jitter in rows and cols dimensions.
        """
        super(Jitter, self).__init__()
        self.jitter = jitter

        # Maintain row and col indices for use in `pre` and `post`
        s, ch, row, col = utils.get_img_indices()
        self.row_idx = row
        self.col_idx = col

        # Jitter amounts in x and y directions.
        self.jx = 0
        self.jy = 0

    def pre(self, img):
        self.jx, self.jy = np.random.randint(-self.jitter, self.jitter+1, 2)
        return np.roll(np.roll(img, self.jx, self.row_idx), self.jy, self.col_idx)

    def post(self, img):
        # Un-shift the jitter.
        return np.roll(np.roll(img, -self.jx, self.row_idx), -self.jy, self.col_idx)
