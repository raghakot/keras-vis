import pprint
import imageio
from utils import utils
import numpy as np

from collections import OrderedDict
from keras import backend as K


class Optimizer(object):

    def __init__(self, input_layer, losses):
        """
        Creates an optimizer that minimizes weighted regularized losses + weighted loss function losses.
        :param input_layer: 4D Keras image input layer (including #samples)
        :param losses: List of (`Redularizer`/`Loss` instances, weight) tuples
        """
        self.img = input_layer
        self.loss_functions = []
        overall_loss = K.variable(0.)

        for loss, weight in losses:
            loss_fn = weight * loss.build_loss(self.img)
            overall_loss += loss_fn
            # Learning phase is added so that 'test' phase can be used to disable dropout.
            self.loss_functions.append((loss.name, K.function([self.img, K.learning_phase()], [loss_fn])))

        grads = K.gradients(overall_loss, self.img)[0]
        # Normalization makes it less sensitive to learning rate during gradient descent.
        grads = utils.normalize(grads)

        self.overall_loss_grad_function = K.function([self.img, K.learning_phase()], [overall_loss, grads])

    def _eval_losses(self, img):
        """
        Evaluates losses with respect to input image.
        :param img: The image to compute loss against.
        :return: A tuple of (name, loss_value) dictionary of various regularization losses and loss function losses.
        """
        losses = OrderedDict()
        for name, fn in self.loss_functions:
            # 0 learning phase for 'test'
            losses[name] = fn([img, 0])
        return losses

    def _eval_loss_and_grads(self, img):
        """
        Evaluates overall loss and its gradients with respect to input image.
        :param img: The image to compute loss, grads against.
        :return: Tuple of (loss, grads)
        """
        # 0 learning phase for 'test'
        return self.overall_loss_grad_function([img, 0])

    def _rmsprop(self, grads, cache=None, decay_rate=0.95):
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

    def _jitter(self, img, jitter=32):
        s, c, w, h = utils.get_image_indices()
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        return np.roll(np.roll(img, ox, w), oy, h)

    def _get_seed_img(self, seed_img):
        """
        Creates the seed_img, along with other sanity checks.
        """
        samples, c, w, h = utils.get_img_shape(self.img)
        if seed_img is None:
            seed_img = utils.generate_rand_img(c, w, h)
        else:
            if K.image_dim_ordering() == 'th':
                seed_img = seed_img.transpose(2, 0, 1)
            # Convert to image tensor containing samples.
            seed_img = np.array([seed_img], dtype=np.float32)
        return seed_img

    def minimize(self, seed_img=None, max_iter=200,
                 jitter=32, verbose=True, progress_gif_path=None):
        """
        Performs gradient descent on the input image with respect to defined losses and regularizations.
        :param seed_img: 3D numpy array in tf or th image ordering to be used as the initial
            seed image for optimization. Seeded with a random noise if None.
        :param max_iter: The maximum number of gradient descent iterations.
        :param jitter: The amount of pixels to jitter in each iteration. Jitter is known to generate crisper images.
        :param verbose: Prints losses/regularization losses at every gradient descent iteration. Very useful to
            estimate weight factor(s).
        :param progress_gif_path: Saves a gif of input image being optimized.
            This slows down perf quite a bit, use with care.
        :return: The tuple of optimized image, gradients of image with respect to losses.
        """
        seed_img = self._get_seed_img(seed_img)

        cache = None
        best_loss = float('inf')
        best_img = None

        writer = None
        grads = None

        try:
            if progress_gif_path:
                if not progress_gif_path.endswith('.gif'):
                    progress_gif_path += '.gif'
                writer = imageio.get_writer(progress_gif_path, mode='I', loop=1)

            for i in range(max_iter):
                if jitter > 0:
                    seed_img = self._jitter(seed_img, jitter)

                loss, grads = self._eval_loss_and_grads(seed_img)

                if verbose:
                    losses = self._eval_losses(seed_img)
                    print('Iteration: {}, losses: {}, overall loss: {}'.format(i+1, pprint.pformat(losses), loss))

                # Gradient descent update.
                step, cache = self._rmsprop(grads, cache)
                seed_img += step

                if writer:
                    writer.append_data(utils.deprocess_image(seed_img.copy()[0]))

                if loss < best_loss:
                    best_loss = loss
                    best_img = seed_img.copy()
        finally:
            if writer:
                print('Saving gif to {}'.format(progress_gif_path))
                writer.close()

        return utils.deprocess_image(best_img[0]), grads
