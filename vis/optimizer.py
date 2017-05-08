import numpy as np

from keras import backend as K
from .callbacks import Print
from .utils import utils


_PRINT_CALLBACK = Print()


class Optimizer(object):

    def __init__(self, img_input, losses, wrt=None, norm_grads=True):
        """Creates an optimizer that minimizes weighted loss function.

        Args:
            img_input: An image input tensor of shape: `(samples, channels, image_dims...)` if 
                data_format='channels_first' or `(samples, image_dims..., channels)` if data_format='channels_last'.
            losses: List of ([Loss](vis.losses#Loss), weight) tuples.
            wrt: Short for, with respect to. This instructs the optimizer that the aggregate loss from `losses`
                should be minimized with respect to `wrt`. `wrt` can be any tensor that is part of the model graph.
                Default value is set to None which means that loss will simply be minimized with respect to `img_input`.
            norm_grads: True to normalize gradients. Normalization avoids very small or large gradients and ensures 
                a smooth gradient gradient descent process. If you want the actual gradient, set this to false.
        """
        self.img = img_input
        self.loss_names = []
        self.loss_functions = []
        self.wrt = self.img if wrt is None else wrt

        overall_loss = None
        for loss, weight in losses:
            # Perf optimization. Don't build loss function with 0 weight.
            if weight != 0:
                loss_fn = weight * loss.build_loss()
                overall_loss = loss_fn if overall_loss is None else overall_loss + loss_fn
                self.loss_names.append(loss.name)
                self.loss_functions.append(loss_fn)

        # Compute gradient of overall with respect to `wrt` tensor.
        grads = K.gradients(overall_loss, self.wrt)[0]
        if norm_grads:
            grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

        # The main function to compute various quantities in optimization loop.
        self.compute_fn = K.function([self.img, K.learning_phase()],
                                     self.loss_functions + [overall_loss, grads, self.wrt])

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

    def _get_seed_img(self, seed_img):
        """Creates a random seed_img if None. Otherwise:
            - Ensures batch_size dim on provided `seed_img`.
            - Shuffle axis according to expected `data_format`.  
        """
        desired_shape = (1, ) + K.int_shape(self.img)[1:]
        if seed_img is None:
            return utils.random_array(desired_shape)

        # Add batch dim if needed.
        if len(seed_img.shape) != len(desired_shape):
            seed_img = np.expand_dims(seed_img, 0)

        # Only possible if channel idx is out of place.
        if seed_img.shape != desired_shape:
            seed_img = np.moveaxis(seed_img, -1, 1)
        return seed_img.astype(K.floatx())

    def minimize(self, seed_img=None, max_iter=200, image_modifiers=None, callbacks=None, verbose=True):
        """Performs gradient descent on the input image with respect to defined losses.

        Args:
            seed_img: An N-dim numpy array of shape: `(samples, channels, image_dims...)` 
                if data_format='channels_first' or `(samples, image_dims..., channels)` if data_format='channels_last'. 
                Seeded with random noise if set to None. (Default value = None)
            max_iter: The maximum number of gradient descent iterations. (Default value = 200)
            image_modifiers: A list of [../vis/modifiers/#ImageModifier](ImageModifier) instances specifying `pre` and
                `post` image processing steps with the gradient descent update step. `pre` is applied in list order while
                `post` is applied in reverse order. For example, `image_modifiers = [f, g]` means that
                `pre_img = g(f(img))` and `post_img = f(g(img))`
            callbacks: A list of [../vis/callbacks/#OptimizerCallback](OptimizerCallback) to trigger during optimization.
            verbose: Logs individual losses at the end of every gradient descent iteration.
                Very useful to estimate loss weight factor. (Default value = True)

        Returns:
            The tuple of `(optimized_image, grads with respect to wrt, wrt_value)` after gradient descent iterations.
        """
        seed_img = self._get_seed_img(seed_img)
        if image_modifiers is None:
            image_modifiers = []

        callbacks = callbacks or []
        if verbose:
            callbacks.append(_PRINT_CALLBACK)

        cache = None
        best_loss = float('inf')
        best_img = None

        grads = None
        wrt_value = None

        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in image_modifiers:
                seed_img = modifier.pre(seed_img)

            # 0 learning phase for 'test'
            computed_values = self.compute_fn([seed_img, 0])
            losses = computed_values[:len(self.loss_names)]
            named_losses = zip(self.loss_names, losses)
            overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]

            # TODO: theano grads shape in inconsistent for some reason. Patch for now and investigate later.
            if grads.shape != seed_img.shape:
                grads = np.reshape(grads, seed_img.shape if self.wrt == self.img else wrt_value.shape)

            # Trigger callbacks
            for c in callbacks:
                c.callback(i, named_losses, overall_loss, grads, wrt_value)

            # Gradient descent update.
            # It only makes sense to do this if wrt is image. Otherwise shapes wont match for the update.
            if self.wrt is self.img:
                step, cache = self._rmsprop(grads, cache)
                seed_img += step

            # Apply modifiers `post` step
            for modifier in reversed(image_modifiers):
                seed_img = modifier.post(seed_img)

            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_img = seed_img.copy()

        # Trigger on_end
        for c in callbacks:
            c.on_end()

        return utils.deprocess_image(best_img[0]), grads, wrt_value
