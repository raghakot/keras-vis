from keras import backend as K
from collections import OrderedDict
from utils import get_img_shape, deprocess_image, normalize, generate_rand_img
import numpy as np
import pprint


class Optimizer(object):

    def __init__(self, model, regularizers, losses, **kwargs):
        """
        Creates an optimizer that minimizes weighted regularized losses + weighted loss function losses.
        :param model: Keras neural network `Model` object
        :param regularizers: List of (regularizer, weight) tuples
        :param losses: List of (loss, weight) tuples
        """
        self.img = model.input
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

        self.loss_functions = []
        overall_loss = K.variable(0.)

        for regularizer, weight in regularizers:
            loss_fn = weight * regularizer.build_loss(self.img)
            overall_loss += loss_fn
            self.loss_functions.append((regularizer.name, K.function([self.img], [loss_fn])))

        for loss, weight in losses:
            loss_fn = loss.build_loss(self.img, layer_dict, **kwargs)
            overall_loss += loss_fn
            self.loss_functions.append((loss.name, K.function([self.img], [loss_fn])))

        grads = K.gradients(overall_loss, self.img)[0]
        # Normalization makes it less sensitive to learning rate during gradient descent.
        grads = normalize(grads)

        self.overall_loss_grad_function = K.function([self.img], [overall_loss, grads])

    def _eval_losses(self, img):
        """
        Evaluates losses with respect to input image.
        :param img: The image to compute loss against.
        :return: A tuple of (name, loss_value) dictionary of various regularization losses and loss function losses.
        """
        losses = OrderedDict()
        for name, fn in self.loss_functions:
            losses[name] = fn([img])
        return losses

    def _eval_loss_and_grads(self, img):
        """
        Evaluates overall loss and its gradients with respect to input image.
        :param img: The image to compute loss, grads against.
        :return: Tuple of (loss, grads)
        """
        return self.overall_loss_grad_function([img])

    def minimize(self, seed_img=None, max_iter=100, verbose=True):
        """
        Performs gradient descent on the input image with respect to defined losses and regularizations.
        :param seed_img: The seed image to start with, for optimization. Seeded with a random noise if None.
        :param max_iter: The maximum number of gradient descent iterations.
        :param verbose: Prints losses/regularization losses at every gradient descent iteration. Very useful to
            estimate weight factor(s).
        :return: The optimized image.
        """
        samples, c, w, h = get_img_shape(self.img)
        if seed_img is None:
            seed_img = generate_rand_img(c, w, h)

        cache = None
        for i in range(max_iter):
            loss, grads = self._eval_loss_and_grads(seed_img)

            if verbose:
                losses = self._eval_losses(seed_img)
                print('losses: {}, overall loss: {}'.format(pprint.pformat(losses), loss))

            # Noob gradient descent update.
            step, cache = self.rmsprop(grads, cache)
            seed_img += step

        return deprocess_image(seed_img[0])

    def rmsprop(self, grads, cache=None, decay_rate=0.95):
        """
        Use RMSProp to compute a step from gradients.
        Inputs:
        - grads: numpy array of gradients.
        - cache: numpy array of same shape as dx giving RMSProp cache
        - decay_rate: How fast to decay cache
        Returns a tuple of:
        - step: numpy array of the same shape as dx giving the step. Note that this
          does not yet take the learning rate into account.
        - cache: Updated RMSProp cache.
        """
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = -grads / np.sqrt(cache + 1e-8)
        return step, cache
