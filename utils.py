import numpy as np
import cv2
from keras import backend as K


def save_filters(filters, img_width, img_height):
    margin = 5
    n = int(len(filters)**0.5)
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index < len(filters):
                img = filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    cv2.imwrite('stitched_filters_%dx%d.png' % (n, n), stitched_filters)


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

    if x.shape[2] != 3:
        x = x.transpose((1, 2, 0))

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


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
