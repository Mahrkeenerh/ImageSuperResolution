import tensorflow as tf
import numpy as np
from tensorflow import keras
from azureml.core import Run
import json, cv2
import os
import random
import sys
import wandb


def PSNR(a, b, max_val=1):
    return tf.image.psnr(a, b, max_val)


def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1.0)


class LogToAzure(tf.keras.callbacks.Callback):
    '''Keras Callback for realtime logging to Azure'''

    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        # Log all log data to Azure
        for k, v in logs.items():
            self.run.log(k, v)


class LogImage(keras.callbacks.Callback):
    """Callback for logging images"""

    def on_epoch_end(self, epoch, logs=None):
        prediction = np.squeeze(self.model(np.expand_dims(valid_low[2], axis=0)), axis=0)
        prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

        # if ./outputs/images doesn't exist, create it
        if not os.path.exists("./outputs/images"):
            os.makedirs("./outputs/images")

        cv2.imwrite(f'./outputs/images/prediction_{epoch}.png', prediction * 255)

        # wandb log prediction image
        wandb.log({'prediction': wandb.Image(prediction_bgr, caption=f'Prediction: {epoch}')}, step=epoch)


def get_callbacks():
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="./outputs/checkpoint",
        save_weights_only=False,
        save_best_only=False
    )

    azure_log = LogToAzure(Run.get_context())
    image_log = LogImage()

    return [checkpoint_callback, azure_log, image_log]


def dumpHistory(history):
    try:
        with open("./outputs/history.json", "w") as history_out:
            json.dump(history.history, history_out)
    except:
        print("Failed to dump history")
        print(sys.exc_info()[0])


train_low = []
train_high = []
valid_low = []
valid_high = []


def set_data(path, low_train, high_train, low_valid, high_valid):

    global train_low, train_high, valid_low, valid_high

    train_low = [cv2.imread(os.path.join(path, low_train, f)) / 255 for f in os.listdir(os.path.join(path, low_train))]
    train_high = [cv2.imread(os.path.join(path, high_train, f)) / 255 for f in os.listdir(os.path.join(path, high_train))]

    valid_low = [cv2.imread(os.path.join(path, low_valid, f)) / 255 for f in os.listdir(os.path.join(path, low_valid))]
    valid_high = [cv2.imread(os.path.join(path, high_valid, f)) / 255 for f in os.listdir(os.path.join(path, high_valid))]


def rotate(low, high):
    yield low, high
    yield cv2.rotate(low, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(high, cv2.ROTATE_90_CLOCKWISE)
    yield cv2.rotate(low, cv2.ROTATE_180), cv2.rotate(high, cv2.ROTATE_180)
    yield cv2.rotate(low, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(high, cv2.ROTATE_90_COUNTERCLOCKWISE)


def flip(low, high):
    yield low, high
    yield np.flip(low), np.flip(high)


def patch(low, high, multiply=1):
    x_patch_size = 128
    y_patch_size = 64
    x = x_patch_size

    shape = low.shape

    while x < shape[0]:
        y = y_patch_size

        while y < shape[1]:
            yield (low[x - x_patch_size: x, y - y_patch_size: y], 
                high[(x - x_patch_size) * multiply: x * multiply, (y - y_patch_size) * multiply: y * multiply])

            y += y_patch_size

        x += x_patch_size


def full_train_generator_at_i(i, multiply=1):
    """Patch generator yielding all results at index i"""

    low, high = train_low[i], train_high[i]

    for rot_low, rot_high in rotate(low, high):
        for flip_low, flip_high in flip(rot_low, rot_high):
            for low_patch, high_patch in patch(flip_low, flip_high, multiply=multiply):
                yield low_patch, high_patch


def get_full_train_generator(multiply=1):
    """Patch generator yielding all results"""

    def full_train_generator(multiply=1):
        for i in range(len(train_low)):
            results = full_train_generator_at_i(i, multiply=multiply)

            for low, high in results:
                yield low, high

    return lambda: full_train_generator(multiply=multiply)


def get_sized_train_generator(size, multiply=1):
    """Patch generator yielding results of size `size`"""

    def sized_train_generator(size, multiply=1):
        for i in range(len(train_low)):
            results = random.sample(list(full_train_generator_at_i(i, multiply=multiply)), size)
            for low, high in results:
                yield low, high

    return lambda: sized_train_generator(size, multiply=multiply)


def get_original_valid_generator():
    """Valid generator yielding all results"""

    def full_valid_generator():
        for i in range(len(valid_low)):
            yield valid_low[i], valid_high[i]

    return lambda: full_valid_generator()


def get_patch_valid_generator(size, multiply=1):
    """Valid generator yielding a single result"""

    def patch_valid_generator(size, multiply=1):
        for i in range(len(valid_low)):
            low, high = valid_low[i], valid_high[i]
            results = random.sample(list(patch(low, high, multiply=multiply)), size)

            for low, high in results:
                yield low, high

    return lambda: patch_valid_generator(size, multiply=multiply)


def get_randomized_generator(generator):
    """Randomize generator"""

    def randomized_generator(random_order, generated_dataset):
        for i in random_order:
            yield generated_dataset[i]

    generated_dataset = list(generator())
    random_order = random.sample([i for i in range(len(generated_dataset))], len(generated_dataset))

    return lambda: randomized_generator(random_order, generated_dataset)


def dataset_from_generator(generator):
    """Convert generator to dataset"""

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE

    return tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, None, 3), (None, None, 3))
    ).with_options(options)
