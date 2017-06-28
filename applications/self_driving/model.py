from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.models import Model
from keras.layers import Input, Dense

FRAME_H = 70
FRAME_W = 180


def build_model():
    inp = Input(shape=(FRAME_H, FRAME_W, 3))
    x = Conv2D(filters=8, kernel_size=(5, 5), activation='relu')(inp)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='tanh')(x)
    return Model(inputs=[inp], outputs=[x])


if __name__ == '__main__':
    model = build_model()
    model.summary()
