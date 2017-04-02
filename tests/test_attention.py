""" Ensures that examples run properly.
"""
import pytest
from vis.utils.test_utils import across_data_formats

from examples.attention_maps import generate_cam
from examples.attention_maps import generate_saliceny_map


@across_data_formats
def test_cam():
    generate_cam(show=False)


@across_data_formats
def test_saliency():
    generate_saliceny_map(show=False)


if __name__ == '__main__':
    pytest.main([__file__])
