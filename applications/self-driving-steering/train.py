from keras.callbacks import ModelCheckpoint
from model import get_model
import data


def train():
    model = get_model()
    print("Loaded model")

    X, y = data.get_validation_dataset()
    print("Loaded validation dataset")
    print("Total of", len(y) * 4)
    model.summary()

    print("Training model")
    checkpoint_path = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)
    model.fit_generator(driving_data.generate_arrays_from_file(),
                        validation_data=(X, y), samples_per_epoch=len(y) * 4,
                        nb_epoch=150, verbose=2,
                        callbacks=[checkpoint])


if __name__ == "__main__":
    train()
