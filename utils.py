from __future__ import division

import numpy as np
import os
import math
import json
import cv2
from keras import backend as K


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255

    # TF image format if channels = (1 or 3) towards the last rank.
    if x.shape[2] != 3 and x.shape[2] != 1:
        x = x.transpose((1, 2, 0))

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def stitch_images(images, margin=5, cols=5):
    """
    Util to stitch images together. Will fold over images to the next row when max_width is exceeded.
    :param images: The images to stitch
    :param margin: The black border margin size between images
    :param cols: max nu
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
    if K.image_dim_ordering() == 'th':
        x = np.random.random((1, c, w, h))
    else:
        x = np.random.random((1, w, h, c))
    x = (x - 0.5) * 20 + 128
    return x


def get_img_shape(img):
    """
    Returns shape in a backend agnostic manner.
    :param img: The image tensor
    :return: The image shape in form (samples, channels, width, height)
    """
    if K.image_dim_ordering() == 'th':
        return K.int_shape(img)
    else:
        samples, w, h, c = K.int_shape(img)
        return samples, c, w, h


def get_image_indices():
    """
    Returns image indices in a backend agnostic manner.
    :return: A tuple representing indices for (samples, channels, width, height)
    """
    if K.image_dim_ordering() == 'th':
        return 0, 1, 2, 3
    else:
        return 0, 3, 1, 2


def load_img(path, grayscale=False, target_size=None):
    if grayscale:
        img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(path)
    if target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    return img


CLASS_INDEX = None
def get_imagenet_label(index):
    global CLASS_INDEX
    if CLASS_INDEX is None:
        with open(os.path.join(os.path.dirname(__file__), 'resources/imagenet_class_index.json')) as f:
            CLASS_INDEX = json.load(f)
    return CLASS_INDEX[str(index)][1]


class BackendAgnosticImageSlice(object):
    """
    Assuming a slice for shape (samples, channels, width, height)
    """
    def __getitem__(self, item_slice):
        assert len(item_slice) == 4
        if K.image_dim_ordering() == 'th':
            return item_slice
        else:
            return tuple([item_slice[0], item_slice[2], item_slice[3], item_slice[1]])


slicer = BackendAgnosticImageSlice()
