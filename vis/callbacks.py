from __future__ import absolute_import
import pprint
from .utils import utils

try:
    import imageio as imageio
except ImportError:
    imageio = None


def _check_imageio():
    if not imageio:
        raise ImportError('Failed to import imageio. You must install imageio')


class OptimizerCallback(object):
    """Abstract class for defining callbacks for use with [optimizer.minimize](vis.optimizer.md#minimize).
    """

    def callback(self, i, named_losses, overall_loss, grads, wrt_value):
        """This function will be called within [optimizer.minimize](vis.optimizer.md#minimize).
        
        Args:
            i: The optimizer iteration.
            named_losses: List of `(loss_name, loss_value)` tuples.
            overall_loss: Overall weighted loss.
            grads: The gradient of input image with respect to `wrt_value`.
            wrt_value: The current `wrt_value`.
        """
        raise NotImplementedError()

    def on_end(self):
        """Called at the end of optimization process. This function is typically used to cleanup / close any
        opened resources at the end of optimization.
        """
        pass


class Print(OptimizerCallback):
    """Callback to print values during optimization    
    """
    def callback(self, i, named_losses, overall_loss, grads, wrt_value):
        print('Iteration: {}, named_losses: {}, overall loss: {}'.format(i+1, pprint.pformat(named_losses), overall_loss))


class GifGenerator(OptimizerCallback):
    """Callback to construct gif of optimized image.
    """
    def __init__(self, path):
        """        
        Args:
            path: The file path to save gif. 
        """
        _check_imageio()
        if not path.endswith('.gif'):
            path += '.gif'
        self.writer = imageio.get_writer(path, mode='I', loop=1)

    def callback(self, i, named_losses, overall_loss, grads, wrt_value):
        img = utils.deprocess_image(wrt_value[0])
        img = utils.draw_text(img, "Step {}".format(i + 1))
        self.writer.append_data(img)

    def on_end(self):
        self.writer.close()
