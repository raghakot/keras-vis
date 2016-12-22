from __future__ import division

import numpy as np
import os
import math
import json
import cv2

from keras import backend as K


# Globals
_CLASS_INDEX = None


def deprocess_image(img):
    """Utility function to convert optimized image output into a valid image.

    Args:
        img: 3D numpy array with shape: `(channels, rows, cols)` if dim_ordering='th' or
            `(rows, cols, channels)` if dim_ordering='tf'.

    Returns:
        A valid image output.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1

    # clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB array
    img *= 255

    # TF image format if channels = (1 or 3) towards the last rank.
    if img.shape[-1] != 3 and img.shape[-1] != 1:
        img = img.transpose((1, 2, 0))

    img = np.clip(img, 0, 255).astype('uint8')
    return img


def stitch_images(images, margin=5, cols=5):
    """Utility function to stitch images together with a `margin`.

    Args:
        images: The array of images to stitch.
        margin: The black border margin size between images (Default value = 5)
        cols: Max number of image cols. New row is created when number of images exceed the column size.
            (Default value = 5)

    Returns:
        A single numpy image array comprising of input images.
    """
    if len(images) == 0:
        return None

    h, w, c = images[0].shape
    n_rows = int(math.ceil(len(images) / cols))
    n_cols = min(len(images), cols)

    out_w = n_cols * w + (n_cols - 1) * margin
    out_h = n_rows * h + (n_rows - 1) * margin
    stitched_images = np.zeros((out_h, out_w, c), dtype=images[0].dtype)

    for row in range(n_rows):
        for col in range(n_cols):
            img_idx = row * cols + col
            if img_idx >= len(images):
                break

            stitched_images[(h + margin) * row : (h + margin) * row + h,
            (w + margin) * col : (w + margin) * col + w, :] = images[img_idx]

    return stitched_images


def generate_rand_img(c, w, h):
    """Generates a random image.

    Args:
      c: image channels
      w: image width
      h: image height

    Returns:
        A numpy array of shape: `(channels, rows, cols)` if dim_ordering='th' or
            `(rows, cols, channels)` if dim_ordering='tf'.
    """
    if K.image_dim_ordering() == 'th':
        x = np.random.random((c, w, h))
    else:
        x = np.random.random((w, h, c))
    x = (x - 0.5) * 20 + 128
    return x


def get_img_shape(img):
    """Returns image shape in a backend agnostic manner.

    Args:
        img: The image tensor in 'th' or 'tf' dim ordering.

    Returns:
        Tuple containing image shape information in `(samples, channels, width, height)` order.
    """
    if K.image_dim_ordering() == 'th':
        return K.int_shape(img)
    else:
        samples, w, h, c = K.int_shape(img)
        return samples, c, w, h


def get_img_indices():
    """Returns image indices in a backend agnostic manner.

    Returns:
        A tuple representing indices for image in `(samples, channels, width, height)` order.
    """
    if K.image_dim_ordering() == 'th':
        return 0, 1, 2, 3
    else:
        return 0, 3, 1, 2


def load_img(path, grayscale=False, target_size=None):
    """Utility function to load an image from disk.

    Args:
      path: The image file path.
      grayscale: True to convert to grayscale image (Default value = False)
      target_size: (w, h) to resize. (Default value = None)

    Returns:
        The loaded numpy image.
    """
    if grayscale:
        img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(path)
    if target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    return img


def get_imagenet_label(index):
    """Utility function to return the image net label for the final dense later output index.

    Args:
        index: The image net output category value,

    Returns:
        Image net label corresponding to the image category.
    """
    global _CLASS_INDEX
    if _CLASS_INDEX is None:
        with open(os.path.join(os.path.dirname(__file__), '../resources/imagenet_class_index.json')) as f:
            _CLASS_INDEX = json.load(f)
    return _CLASS_INDEX[str(index)][1]


class _BackendAgnosticImageSlice(object):
    """Utility class to make image slicing across 'th'/'tf' backends easier.
    """

    def __getitem__(self, item_slice):
        """Assuming a slice for shape `(samples, channels, width, height)`
        """
        assert len(item_slice) == 4
        if K.image_dim_ordering() == 'th':
            return item_slice
        else:
            return tuple([item_slice[0], item_slice[2], item_slice[3], item_slice[1]])


"""Slice utility to image slicing across 'th'/'tf' backends easier.
Example:
    conv_layer[utils.slicer[:, filter_idx, :, :]] will work for both theano and tensorflow backends
    even though, in tensorflow, slice should be conv_layer[utils.slicer[:, :, :, filter_idx]]
"""
slicer = _BackendAgnosticImageSlice()
