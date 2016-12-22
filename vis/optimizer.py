import pprint
import cv2
import imageio
import numpy as np

from collections import OrderedDict
from keras import backend as K
from utils import utils


class Optimizer(object):

    def __init__(self, img, losses):
        """Creates an optimizer that minimizes weighted loss function.

        Args:
            img: 4D tensor with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'.
            losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        """
        self.img = img
        self.loss_functions = []
        overall_loss = K.variable(0.)

        for loss, weight in losses:
            # Perf optimization. Don't build loss function with 0 weight.
            if weight != 0:
                loss_fn = weight * loss.build_loss(self.img)
                overall_loss += loss_fn
                # Learning phase is added so that 'test' phase can be used to disable dropout.
                self.loss_functions.append((loss.name, K.function([self.img, K.learning_phase()], [loss_fn])))

        grads = K.gradients(overall_loss, self.img)[0]
        # Normalization avoids very small or large gradients and ensures a smooth gradient gradient descent process.
        grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

        self.overall_loss_grad_function = K.function([self.img, K.learning_phase()], [overall_loss, grads])

    def _eval_losses(self, img):
        """Evaluates losses with respect to numpy input image.

        Args:
            img: 4D numpy array with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'.

        Returns:
            A dictionary of ([Loss](vis.losses#Loss).name, loss_value) values for various losses.
        """
        losses = OrderedDict()
        for name, fn in self.loss_functions:
            # 0 learning phase for 'test'
            losses[name] = fn([img, 0])
        return losses

    def _eval_loss_and_grads(self, img):
        """Evaluates overall loss and its gradients with respect to input image.

        Args:
           img: 4D numpy array with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'

        Returns:
            Tuple containing (overall_loss, grads) with respect to provided input image.
        """
        # 0 learning phase for 'test'
        return self.overall_loss_grad_function([img, 0])

    def _rmsprop(self, grads, cache=None, decay_rate=0.95):
        """Uses RMSProp to compute step from gradients.

        Args:
            grads: numpy array of gradients.
            cache: numpy array of same shape as `grads` as RMSProp cache
            decay_rate: How fast to decay cache

        Returns:
            A tuple of
                step: numpy array of the same shape as `grads` giving the step.
                    Note that this does not yet take the learning rate into account.
                cache: Updated RMSProp cache.
        """
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = -grads / np.sqrt(cache + K.epsilon())
        return step, cache

    def _jitter(self, img, jitter=32):
        """Jitters the numpy input image randomly in width and height dimensions.
        This kind of regularization is known to produce crisper images via guided backprop.

        Args:
          img: 4D numpy array with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'
          jitter: Number of pixels to jitter in width and height directions.

        Returns:
            The jittered numpy image array.
        """
        s, c, w, h = utils.get_img_indices()
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        return np.roll(np.roll(img, ox, w), oy, h)

    def _get_seed_img(self, seed_img):
        """Creates the seed_img, along with other sanity checks.
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
                 jitter=8, verbose=True, progress_gif_path=None):
        """Performs gradient descent on the input image with respect to defined losses.

        Args:
            seed_img: 3D numpy array with shape: `(channels, rows, cols)` if dim_ordering='th' or
                `(rows, cols, channels)` if dim_ordering='tf'.
                Seeded with random noise if set to None. (Default value = None)
            max_iter: The maximum number of gradient descent iterations. (Default value = 200)
            jitter: The number of pixels to jitter between subsequent gradient descent iterations.
                Jitter is known to generate crisper images. (Default value = 8)
            verbose: Logs individual losses at the end of every gradient descent iteration.
                Very useful to estimate loss weight factor. (Default value = True)
            progress_gif_path: Saves a gif of input image being optimized.
                This slows down perf quite a bit, use with care.

        Returns:
            The tuple of `(optimized image, gradients)` of image with respect to losses.
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
                    seed_img_copy = utils.deprocess_image(seed_img.copy()[0])
                    cv2.putText(seed_img_copy, "{}".format(i + 1), (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                    writer.append_data(seed_img_copy)

                if loss < best_loss:
                    best_loss = loss
                    best_img = seed_img.copy()
        finally:
            if writer:
                print('Saving gif to {}'.format(progress_gif_path))
                writer.close()

        return utils.deprocess_image(best_img[0]), grads
