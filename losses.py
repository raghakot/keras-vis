from utils import slicer
from keras import backend as K


class Loss(object):
    """
    Abstract class for defining the loss function to be minimized.
    """
    def __init__(self):
        self.name = "Unnamed Loss"

    def build_loss(self, img, layer_dict, **kwargs):
        """
        Define the loss that needs to be minimized.
        :param img: The image of shape (1, C, W, H)
        :param layer_dict: Named dictionary of various layers on the network.
        :return: The loss expression.
        """
        raise NotImplementedError()


class ActivationMaximization(Loss):
    """
    Finds an input image such that activation at a particular layer/filter is maximized.
    """
    def __init__(self):
        super(ActivationMaximization, self).__init__()
        self.name = "Activation Maximization Loss"

    def build_loss(self, img, layer_dict, **kwargs):
        """
        Builds a loss function that maximizes activation map of
        :param img: The image of shape (1, C, W, H)
        :param layer_dict: Named dictionary of various layers on the network.
        :param kwargs: Must contain `layer_name` and `filter_idx` that needs to be maximized.
            for Dense layers, `filter_idx` is interpreted as output index.
        :return: The loss function.
        """
        layer_name = kwargs.get('layer_name')
        filter_idx = kwargs.get('filter_idx')
        if layer_name is None:
            raise ValueError("layer_name must be provided")
        if filter_idx is None:
            raise ValueError("filter_idx must be provided")

        layer_output = layer_dict[layer_name].output

        # For all other layers it is 4
        isDense = K.ndim(layer_output) == 2

        if isDense:
            loss = -K.mean(layer_output[:, filter_idx])
        else:
            loss = -K.mean(layer_output[slicer[:, filter_idx, :, :]])

        return loss
