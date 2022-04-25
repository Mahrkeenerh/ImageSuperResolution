import time
import argparse
from types import SimpleNamespace

import tensorflow as tf
from tensorflow import keras as tfk
import keras
from keras import layers
from keras import backend as K
from wandb.keras import WandbCallback

import nn_plus


class DRCSR_model:
    def __init__(self, simple_namespace, config=None, verbose=True):
        self.epochs = simple_namespace.epochs
        self.batch_size = simple_namespace.batch_size
        self.path = simple_namespace.path
        self.model_name = simple_namespace.model_name

        self.verbose = verbose
        self.config = config

        self.learning_rate = 0.001
        self.optimizer_name = "adam"
        self.loss_name = "mean_squared_error"
        self.callbacks = []
        self.kernel_sizes = SimpleNamespace(
            first_kernel=(3, 3),
            outer_kernel=(3, 3),
            middle_kernel=(3, 3),
            last_kernel=(3, 3)
        )

        if config is not None:
            self.learning_rate = config["learning_rate"]
            self.optimizer_name = config["optimizer"]
            self.loss_name = config["loss"]
            self.callbacks = [WandbCallback()]

        self.build_model()
        self.build_datasets()
   

    def build_model(self):

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            self.model = get_model(self.kernel_sizes)
            self.callbacks += nn_plus.get_callbacks()

            self.model.compile(
                optimizer=get_optimizer(self.optimizer_name, self.learning_rate),
                loss=get_loss(self.loss_name),
                metrics=[nn_plus.PSNR]
            )

            self.model.summary()


    def build_datasets(self):
        if self.verbose:
            print("=" * 98)
            print("LOADING DATASETS")
            start_time = time.time()

        nn_plus.set_data(self.path, "train_LR", "train_HR", "valid_LR", "valid_HR")
        random_generator = nn_plus.get_sized_train_generator(10)
        valid_generator = nn_plus.get_patch_valid_generator(2)
        # randomized_generator = nn_plus.get_randomized_generator(random_generator)

        # self.train_dataset = nn_plus.dataset_from_generator(randomized_generator).batch(32)
        self.train_dataset = nn_plus.dataset_from_generator(random_generator).batch(32)
        self.validate_dataset = nn_plus.dataset_from_generator(valid_generator).batch(32)

        if self.verbose:
            print(
                f"DATASETS LOADED, time: "
                f"{int((time.time() - start_time) // 60)}m "
                f"{round((time.time() - start_time) % 60)}s"
            )


    def train(self):
        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            shuffle=True,
            callbacks=[self.callbacks],
            validation_data=self.validate_dataset
        )


    def save(self):
        nn_plus.dumpHistory(self.history)
        self.model.save(f'./outputs/model_{self.model_name}.h5')


def conv_block(
    input_layer: layers.Layer,
    kernel_sizes: SimpleNamespace,
    block_name: str = "conv_block"
) -> layers.Layer:
    with K.name_scope(block_name):
        layer = layers.Conv2D(
            128,
            kernel_sizes.outer_kernel,
            activation='relu',
            padding='same'
        )(input_layer)

        layer = layers.Conv2D(
            256,
            kernel_sizes.middle_kernel,
            activation='relu',
            padding='same'
        )(layer)

        layer = layers.Conv2D(
            128,
            kernel_sizes.outer_kernel,
            activation='relu',
            padding='same'
        )(layer)

        return layer


def get_model(kernel_sizes: SimpleNamespace) -> keras.Model:
    input_layer = keras.Input(shape=(None, None, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    first = layers.Conv2D(128, kernel_sizes.first_kernel, activation='relu', padding='same')(x)

    x = conv_block(first, kernel_sizes)
    merged = layers.Add()([first, x])
    x = conv_block(x, kernel_sizes)
    merged = layers.Add()([merged, x])
    x = conv_block(x, kernel_sizes)
    merged = layers.Add()([merged, x])
    x = conv_block(x, kernel_sizes)
    merged = layers.Add()([merged, x])
    x = conv_block(x, kernel_sizes)
    merged = layers.Add()([merged, x])
    x = conv_block(x, kernel_sizes)
    merged = layers.Add()([merged, x])
    x = conv_block(x, kernel_sizes)
    merged = layers.Add()([merged, x])
    x = conv_block(x, kernel_sizes)
    merged = layers.Add()([merged, x])

    x = layers.Conv2D(32, kernel_sizes.last_kernel, activation='relu', padding='same')(merged)
    out = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return keras.Model(input_layer, out)


def get_optimizer(optimizer_name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    if optimizer_name == 'adam':
        return tfk.optimizers.Adam(learning_rate)

    if optimizer_name == 'sgd':
        return tfk.optimizers.SGD(learning_rate)
    
    return tfk.optimizers.Adam(learning_rate)


def get_loss(loss_name: str) -> tfk.losses.Loss:
    if loss_name == 'mean_squared_error':
        return tfk.losses.MeanSquaredError()

    if loss_name == 'binary_crossentropy':
        return tfk.losses.BinaryCrossentropy()
    
    return tf.keras.losses.MeanSquaredError()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, required=True)
    parser.add_argument("--path", "-p", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    parser.add_argument("--wandb_api_key", "-wak", type=str, required=False)
    args = parser.parse_args()

    simple_namespace = SimpleNamespace(
        epochs=args.epochs,
        batch_size=32,
        path=args.path,
        model_name=args.model_name
    )

    drcsr_model = DRCSR_model(simple_namespace)
    drcsr_model.train()
    drcsr_model.save()
