from __future__ import absolute_import
from __future__ import division

import numpy as np
import matplotlib.font_manager as fontman

import os
import math
import json

from skimage import io, transform
from collections import Iterable
from keras import backend as K

import logging
logger = logging.getLogger(__name__)

try:
    import PIL as pil
    from PIL import ImageFont
    from PIL import Image
    from PIL import ImageDraw
except ImportError:
    pil = None


# Globals
_CLASS_INDEX = None


def _check_pil():
    if not pil:
        raise ImportError('Failed to import PIL. You must install Pillow')


def _find_font_file(query):
    """Utility to find font file.
    """
    return list(filter(lambda path: query.lower() in os.path.basename(path).lower(), fontman.findSystemFonts()))


def set_random_seed(seed_value=1337):
    """Sets random seed value for reproducibility.

    Args:
        seed_value: The seed value to use. (Default Value = infamous 1337)
    """
    np.random.seed(seed_value)


def reverse_enumerate(iterable):
    """Enumerate over an iterable in reverse order while retaining proper indexes, without creating any copies.
    """
    return zip(reversed(range(len(iterable))), reversed(iterable))


def listify(value):
    """Ensures that the value is a list. If it is not a list, it creates a new list with `value` as an item.
    """
    if not isinstance(value, Iterable):
        value = [value]
    return value


def random_array(shape, mean=128., std=20.):
    """Creates a uniformly distributed random array with the given mean and std.
    
    Args:
        shape: The desired shape
        mean: The desired mean (Default value = 128)
        std: The desired std (Default value = 20)

    Returns: Random numpy array of given `shape` uniformly distributed with desired `mean` and `std`.
    """
    x = np.random.random(shape)
    # normalize around mean=0, std=1
    x = (x - np.mean(x)) / np.std(x)
    # and then around the desired mean/std
    x = (x * std) + mean
    return x


def deprocess_image(img):
    """Utility function to convert optimized image output into a valid image.

    Args:
        img: N dim numpy image array with shape: `(channels, image_dims...)` if data_format='channels_first' or
            `(image_dims..., channels)` if data_format='channels_last'.

    Returns:
        A valid image output.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    img = img.copy()
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1

    # clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB array
    img *= 255
    return np.clip(img, 0, 255).astype('uint8')


def stitch_images(images, margin=5, cols=5):
    """Utility function to stitch images together with a `margin`.

    Args:
        images: The array of 2D images to stitch.
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

            stitched_images[(h + margin) * row: (h + margin) * row + h,
            (w + margin) * col: (w + margin) * col + w, :] = images[img_idx]

    return stitched_images


def get_img_shape(img):
    """Returns image shape in a backend agnostic manner.

    Args:
        img: An image tensor of shape: `(channels, image_dims...)` if data_format='channels_first' or
            `(image_dims..., channels)` if data_format='channels_last'.

    Returns:
        Tuple containing image shape information in `(samples, channels, image_dims...)` order.
    """
    if isinstance(img, np.ndarray):
        shape = img.shape
    else:
        shape = K.int_shape(img)

    if K.image_data_format() == 'channels_last':
        shape = list(shape)
        shape.insert(1, shape[-1])
        shape = tuple(shape[:-1])
    return shape


def load_img(path, grayscale=False, target_size=None):
    """Utility function to load an image from disk.

    Args:
      path: The image file path.
      grayscale: True to convert to grayscale image (Default value = False)
      target_size: (w, h) to resize. (Default value = None)

    Returns:
        The loaded numpy image.
    """
    img = io.imread(path, grayscale)
    if target_size:
        img = transform.resize(img, target_size, preserve_range=True).astype('uint8')
    return img


def get_imagenet_label(indices, join=', '):
    """Utility function to return the image net label for the final `dense` layer output index.

    Args:
        indices: Could be a single value or an array of indices whose labels needs looking up.
        join: When multiple indices are passed, the output labels are joined using this value. (Default Value = ', ')

    Returns:
        Image net label corresponding to the image category.
    """
    global _CLASS_INDEX
    if _CLASS_INDEX is None:
        with open(os.path.join(os.path.dirname(__file__), '../../resources/imagenet_class_index.json')) as f:
            _CLASS_INDEX = json.load(f)

    indices = listify(indices)
    return join.join([_CLASS_INDEX[str(idx)][1] for idx in indices])


def draw_text(img, text, position=(10, 10), font='FreeSans.ttf', font_size=14, color=(0, 0, 0)):
    """Draws text over the image. Requires PIL.
    
    Args:
        img: The image to use.
        text: The text string to overlay.
        position: The text (x, y) position. (Default value = (10, 10)) 
        font: The ttf or open type font to use. (Default value = 'FreeSans.ttf')
        font_size: The text font size. (Default value = 12)
        color: The (r, g, b) values for text color. (Default value = (0, 0, 0))

    Returns: Image overlayed with text.
    """
    _check_pil()

    font_files = _find_font_file(font)
    if len(font_files) == 0:
        logger.warn("Failed to lookup font '{}', falling back to default".format(font))
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_files[0], font_size)

    # Don't mutate original image
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text(position, text, fill=color, font=font)
    return np.asarray(img)


def bgr2rgb(img):
    """Converts an RGB image to BGR and vice versa
    
    Args:
        img: Numpy array in RGB or BGR format

    Returns: The converted image format
    """
    return img[..., ::-1]


def normalize(array, min_value=0., max_value=1.):
    """Normalizes the numpy array to (min_value, max_value)

    Args:
        array: The numpy array
        min_value: The min value in normalized array (Default value = 0)
        max_value: The max value in normalized array (Default value = 1)

    Returns:
        The array normalized to range between (min_value, max_value)
    """
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min)
    return (max_value - min_value) * normalized + min_value


class _BackendAgnosticImageSlice(object):
    """Utility class to make image slicing uniform across various `data_format`.
    """

    def __getitem__(self, item_slice):
        """Assuming a slice for shape `(samples, channels, image_dims...)`
        """
        if K.image_data_format() == 'channels_first':
            return item_slice
        else:
            # Move channel index to last position.
            item_slice = list(item_slice)
            item_slice.append(item_slice.pop(1))
            return tuple(item_slice)


"""Slice utility to make image slicing uniform across various `data_format`.
Example:
    conv_layer[utils.slicer[:, filter_idx, :, :]] will work for both `channels_first` and `channels_last` data formats
    even though, in tensorflow, slice should be conv_layer[utils.slicer[:, :, :, filter_idx]]
"""
slicer = _BackendAgnosticImageSlice()
