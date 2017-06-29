from __future__ import absolute_import

from keras import backend as K
from .utils import utils


class Loss(object):
    """Abstract class for defining the loss function to be minimized.
    The loss function should be built by defining `build_loss` function.

    The attribute `name` should be defined to identify loss function with verbose outputs.
    Defaults to 'Unnamed Loss' if not overridden.
    """
    def __init__(self):
        self.name = "Unnamed Loss"

    def __str__(self):
        return self.name

    def build_loss(self):
        """Implement this function to build the loss function expression.
        Any additional arguments required to build this loss function may be passed in via `__init__`.

        Ideally, the function expression must be compatible with all keras backends and `channels_first` or
        `channels_last` image_data_format(s). `utils.slicer` can be used to define data format agnostic slices.
        (just define it in `channels_first` format, it will automatically shuffle indices for tensorflow
        which uses `channels_last` format).

        ```python
        # theano slice
        conv_layer[:, filter_idx, ...]

        # TF slice
        conv_layer[..., filter_idx]

        # Backend agnostic slice
        conv_layer[utils.slicer[:, filter_idx, ...]]
        ```

        [utils.get_img_shape](vis.utils.utils.md#get_img_shape) and
        [utils.get_img_indices](vis.utils.utils.md#get_img_indices) are other optional utilities that make this easier.

        Returns:
            The loss expression.
        """
        raise NotImplementedError()


class ActivationMaximization(Loss):
    """A loss function that maximizes the activation of a set of filters within a particular layer.

    Typically this loss is used to ask the reverse question - What kind of input image would increase the networks
    confidence, for say, dog class. This helps determine what the network might be internalizing as being the 'dog'
    image space.

    One might also use this to generate an input image that maximizes both 'dog' and 'human' outputs on the final
    `keras.layers.Dense` layer.
    """
    def __init__(self, layer, filter_indices):
        """
        Args:
            layer: The keras layer whose filters need to be maximized. This can either be a convolutional layer
                or a dense layer.
            filter_indices: filter indices within the layer to be maximized.
                For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.

                If you are optimizing final `keras.layers.Dense` layer to maximize class output, you tend to get
                better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
                output can be maximized by minimizing scores for other classes.
        """
        super(ActivationMaximization, self).__init__()
        self.name = "ActivationMax Loss"
        self.layer = layer
        self.filter_indices = utils.listify(filter_indices)

    def build_loss(self):
        layer_output = self.layer.output

        # For all other layers it is 4
        is_dense = K.ndim(layer_output) == 2

        loss = 0.
        for idx in self.filter_indices:
            if is_dense:
                loss += -K.mean(layer_output[:, idx])
            else:
                # slicer is used to deal with `channels_first` or `channels_last` image data formats
                # without the ugly conditional statements.
                loss += -K.mean(layer_output[utils.slicer[:, idx, ...]])

        return loss


class RegressionTarget(Loss):
    """A loss function that drives regression outputs to targets.

    Unlike classification, we cannot simply maximize the class output as it only serves to increase the regression
    node output. We might want to visualize change in input that would cause the output to:

    - Increase
    - Decrease
    - Remain the same

    For example, with self driving car steering angle output, we might want to see what causes it to turn more right
    than predicted (increase output), more left (decrease output), or maintain its prediction (same).
    """
    def __init__(self, layer, output_indices, targets, threshold=0.05):
        """
        Args:
            layer: The keras layer whose regression outputs needs to be visualized. This can be any layer with output
                of shape `(batch_size, outputs)`.
            output_indices: Output indices within the layer.
            targets: The desired regression output targets corresponding to `output_indices` outputs.
                To visualize input where output is same as the target, a value of `None` can be passed. This saves
                additional computation of predicting the actual output and then passing it as the target.
            threshold: The threshold within which output should be considered the same as target.
                ie., |output - target| < threshold means that regression output should not be changed.
        """
        super(RegressionTarget, self).__init__()
        self.name = "RegressionTarget Loss"

        self.layer = layer
        self.output_indices = utils.listify(output_indices)
        self.targets = utils.listify(targets)
        self.threshold = K.variable(threshold)

        if K.ndim(layer.output) != 2:
            raise ValueError('`layer` output should have shape `(batch_size, num_outputs)`. Found output '
                             'dim of shape {}'.format(K.int_shape(layer.output)))
        if len(self.output_indices) != len(self.targets):
            raise ValueError('There should be targets corresponding to each `output_indices`')

    def build_loss(self):
        """The rationale for this loss is as follows:

        - When we consider the gradient of input pixel with respect to output, a positive gradient indicates the
        direction that would cause the output to increase. Hence we try to increase the output when `target > output`.

        - To decrease the value we need to consider negative gradients. So if we step in negative of that direction,
        the output will decrease.

        - To maintain the same output, we want to focus on the gradients that are small. Hence we consider inverse of
        output to highlight gradients that would be small.

        Basically, we want positive gradient to indicate the direction that achieves the increase, decrease or
        remains same cases so that the optimization code can be uniform.

        Returns:
            The loss expression.
        """
        overall_loss = None
        for i, idx in enumerate(self.output_indices):
            output = self.layer.output[0, idx]
            target = self.targets[idx]
            if target is None:
                target = output
            else:
                target = K.variable(target)

            diff = target - output
            is_greater = K.greater(diff, self.threshold)
            is_lesser = K.less(diff, -self.threshold)

            # Rationale is as follows:
            # if diff > th:
            #   return output (since positive gradients here indicate how to increase output)
            # elif diff < -th:
            #   return -output (since positive gradients here indicate how to decrease output)
            # else:
            #   return sign(output) / output (since positive gradients here highlight output gradients that are small
            #                                 and hence keep the output unchanged)
            #
            # Negative sign to overall expression since reduction in loss should get us closer to our goal of
            # driving outputs closer to targets.
            loss = -K.switch(is_greater, output,
                             K.switch(is_lesser, -output, K.sign(diff) / (output + K.epsilon())))
            overall_loss = loss if overall_loss is None else overall_loss + loss

        return overall_loss
