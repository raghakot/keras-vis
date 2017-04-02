""" Ensures that examples run properly.
"""
import pytest
from vis.utils.test_utils import across_data_formats

from examples.visualize_layer import visualize_random
from examples.visualize_layer import visualize_multiple_categories
from examples.visualize_layer import visualize_multiple_same_filter


@across_data_formats
def test_visualize_random():
    visualize_random(1, show=False)


@across_data_formats
def test_visualize_multiple_same_filter():
    visualize_multiple_same_filter(1, show=False)


@across_data_formats
def test_visualize_multiple_categories():
    visualize_multiple_categories(show=False)


if __name__ == '__main__':
    pytest.main([__file__])
