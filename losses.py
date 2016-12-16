from keras import backend as K
from utils import utils


class Loss(object):
    """
    Abstract class for defining the loss function to be minimized.
    """
    def __init__(self):
        self.name = "Unnamed Loss"

    def build_loss(self, img):
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
    def __init__(self, layer, filter_indices):
        """
        :param layer: The keras layer to optimize.
        :param filter_indices: filter indices within the layer to be maximized.
            For `Dense` layers, `filter_idx` is interpreted as output index.

            If you are optimizing final Dense layer to maximize class output, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.
        """
        super(ActivationMaximization, self).__init__()
        self.name = "Activation Maximization Loss"
        self.layer = layer
        self.filter_indices = filter_indices

    def build_loss(self, img):
        """
        Builds a loss function that maximizes activation map of
        :param img: The image of shape (1, C, W, H)
        :param layer_dict: Named dictionary of various layers on the network.
        :param kwargs: Must contain `layer_name` and `filter_idx` that needs to be maximized.
            for Dense layers, `filter_idx` is interpreted as output index.
        :return: The loss function.
        """

        layer_output = self.layer.output

        # For all other layers it is 4
        isDense = K.ndim(layer_output) == 2

        loss = 0.
        for idx in self.filter_indices:
            if isDense:
                loss += -K.mean(layer_output[:, idx])
            else:
                loss += -K.mean(layer_output[utils.slicer[:, idx, :, :]])

        return loss
