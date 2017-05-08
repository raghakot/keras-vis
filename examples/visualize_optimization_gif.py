from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.modifiers import Jitter
from vis.optimizer import Optimizer

from vis.callbacks import GifGenerator
from vis.utils.vggnet import VGG16


def generate_opt_gif():
    """Example to show how to generate the gif of optimization progress.
    This example also shows how to use the optimizer directly with losses.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    output_class = [20]

    losses = [
        (ActivationMaximization(layer_dict[layer_name], output_class), 2),
        (LPNorm(model.input), 10),
        (TotalVariation(model.input), 10)
    ]
    opt = Optimizer(model.input, losses)
    opt.minimize(max_iter=500, verbose=True, image_modifiers=[Jitter()], callbacks=[GifGenerator('opt_progress')])


if __name__ == '__main__':
    generate_opt_gif()
