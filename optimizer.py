from keras import backend as K
from collections import OrderedDict
from scipy.optimize import fmin_l_bfgs_b
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
        self.grads_function = []

        for regularizer, weight in regularizers:
            loss_fn = weight * regularizer.build_loss(self.img)
            grads = K.gradients(loss_fn, self.img)[0]
            grads = normalize(grads)
            self.loss_functions.append((regularizer.name, K.function([self.img], [loss_fn, grads])))

        for loss, weight in losses:
            loss_fn = loss.build_loss(self.img, layer_dict, **kwargs)
            grads = K.gradients(loss_fn, self.img)[0]
            grads = normalize(grads)
            self.loss_functions.append((loss.name, K.function([self.img], [loss_fn, grads])))

    def eval_loss_and_grads(self, img):
        """
        Evaluates loss and gradients with respect to input image.
        :return: A tuple of
        - losses: (name, loss_value) dictionary of various regularization losses and loss function losses.
        - grads: The aggregate loss gradients with respect to image input
        """
        losses = OrderedDict()
        grads_list = []
        for name, fn in self.loss_functions:
            loss_value, grads = fn([img])
            losses[name] = loss_value
            grads_list.append(grads)

        grads = np.sum(np.array(grads_list), 0)
        return losses, grads

    def minimize(self, seed_img=None, max_iter=100, verbose=True):
        """
        Performs gradient descent on the input image with respect to defined losses and regularizations.
        :param seed_img: The seed image to start with, for optimization. Seeded with a random noise if None.
        :param verbose: Prints losses/regularization losses at every gradient descent iteration. Very useful to
            estimate weight factor(s).
        :return: The optimized image.
        """
        # Run scipy-based optimization (L-BFGS) over the pixels of the input image
        samples, c, w, h = get_img_shape(self.img)
        if seed_img is None:
            seed_img = generate_rand_img(c, w, h)

        # evaluator = Evaluator(self)
        for i in range(max_iter):
            # x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
            #                                  fprime=evaluator.grads, maxfun=self.max_iter)
            losses, grads = self.eval_loss_and_grads(seed_img)

            # l2 norm, makes it easier to perform updates
            # grads = grads / (np.sqrt(np.mean(grads ** 2, -1, keepdims=True)) + K.epsilon())
            seed_img -= grads * 0.1

            if verbose:
                print('losses: {}, overall loss: {}'.format(pprint.pformat(losses), np.mean(losses.values())))

        return deprocess_image(seed_img[0])


class Evaluator(object):
    """
    This Evaluator class makes it possible to compute loss and gradients in one pass
    while retrieving them via two separate functions, "loss" and "grads".
    This is done because scipy.optimize requires separate functions for loss and gradients,
    but computing them separately would be inefficient.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.losses = None
        self.loss = None
        self.grads = None

    def loss(self, x):
        assert self.loss is None
        self.losses, self.grads = self.optimizer.eval_loss_and_grads(x)
        self.loss = np.mean(self.losses)
        return self.loss

    def grads(self, x):
        assert self.loss is not None
        grads = np.copy(self.grads)
        self.losses = None
        self.loss = None
        self.grads = None
        return grads
