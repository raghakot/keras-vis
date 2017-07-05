import pytest
from vis.utils import utils
from keras import backend as K


def test_get_img_shape_on_2d_image():
    n = 5
    channels = 4
    dim1 = 1
    dim2 = 2

    K.set_image_data_format('channels_first')
    assert (n, channels, dim1, dim2) == utils.get_img_shape(K.ones(shape=(n, channels, dim1, dim2)))

    K.set_image_data_format('channels_last')
    assert (n, channels, dim1, dim2) == utils.get_img_shape(K.ones(shape=(n, dim1, dim2, channels)))


def test_get_img_shape_on_3d_image():
    n = 5
    channels = 4
    dim1 = 1
    dim2 = 2
    dim3 = 3

    K.set_image_data_format('channels_first')
    assert (n, channels, dim1, dim2, dim3) == utils.get_img_shape(K.ones(shape=(n, channels, dim1, dim2, dim3)))

    K.set_image_data_format('channels_last')
    assert (n, channels, dim1, dim2, dim3) == utils.get_img_shape(K.ones(shape=(n, dim1, dim2, dim3, channels)))


def test_reverse_iterable():
    assert list(utils.reverse_enumerate('abcde')) == [(4, 'e'), (3, 'd'), (2, 'c'), (1, 'b'), (0, 'a')]


if __name__ == '__main__':
    pytest.main([__file__])
