from keras import backend as K


# Import backend depending on config
if K.backend() == 'tensorflow':
    from .tensorflow_backend import *
elif K.backend() == 'theano':
    from .theano_backend import *
else:
    raise ValueError("Backend '{}' not supported".format(K.backend()))
