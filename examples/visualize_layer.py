import cv2
import numpy as np

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation


def visualize_random(num_categories=10):
    """Example to show how to visualize multiple filters via activation maximization.

    Args:
        num_categories: The number of random categories to visualize. (Default Value = 5)
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    # Visualize couple random categories from imagenet.
    indices = np.random.permutation(1000)[:num_categories]
    images = [visualize_activation(model, layer_idx, filter_indices=idx,
                                   text=utils.get_imagenet_label(idx),
                                   max_iter=500) for idx in indices]

    # Easily stitch images via `utils.stitch_images`
    cv2.imshow('Random imagenet categories', utils.stitch_images(images))
    cv2.waitKey(0)


def visualize_multiple_same_filter(num_runs=3):
    """Example to show how to visualize same filter multiple times via different runs.

    Args:
        num_runs: The number of times the same filter is visualized
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    # 20 is the imagenet category for 'ouzel'
    indices = [20] * num_runs
    images = [visualize_activation(model, layer_idx, filter_indices=idx,
                                   text=utils.get_imagenet_label(idx),
                                   max_iter=500) for idx in indices]
    cv2.imshow('Multiple runs of ouzel', utils.stitch_images(images))
    cv2.waitKey(0)


def visualize_multiple_categories():
    """Example to show how to visualize images that activate multiple categories
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    # Visualize [20] (ouzel) and [20, 71] (An ouzel-scorpion :D)
    indices = [20, [20, 71]]
    images = [visualize_activation(model, layer_idx, filter_indices=idx,
                                   text=utils.get_imagenet_label(idx),
                                   max_iter=500) for idx in indices]
    cv2.imshow('Multiple category visualization', utils.stitch_images(images))
    cv2.waitKey(0)


if __name__ == '__main__':
    print('Visualizing random imagenet output categories')
    visualize_random(3)

    print('Visualizing same filter over multiple runs')
    visualize_multiple_same_filter()

    print('Visualizing multiple categories')
    visualize_multiple_categories()
