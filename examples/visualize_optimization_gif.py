from vis.utils.vggnet import VGG16
from vis.visualization import visualize_class_activation
from vis.callbacks import GifGenerator


def generate_opt_gif():
    """Example to show how to generate the gif of optimization progress.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
    output_class = [20]

    optimizer_params = {
        'max_iter': 500,
        'verbose': True,
        'callbacks': [GifGenerator('opt_progress')]
    }
    visualize_class_activation(model, layer_idx, filter_indices=output_class, **optimizer_params)


if __name__ == '__main__':
    generate_opt_gif()
