import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tkinter import filedialog
import time
import ctypes

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import backend as K

import keras2


kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


def PSNR(a, b, max_val=1):
    return tf.image.psnr(a, b, max_val)

def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1.0)


layers.UpSampling2D = keras2.UpSampling2D
K.resize_images = keras2.resize_images


def get_image_path():
    return filedialog.askopenfile(filetypes=[("Image", ["*.jpg", "*.png"])])


def get_factor():
    return input("Enter factor: ")


print("Choose image to upscale:")

image_path = get_image_path()
factor = get_factor()

while factor == "" or not factor.isdigit():
    print("Invalid factor, try again:")
    factor = get_factor()

factor = int(factor)
image = cv2.imread(image_path.name) / 255

bi_image = cv2.resize(
    image, 
    (int(image.shape[1] * factor),(image.shape[0] * factor)),
    cv2.INTER_CUBIC
)

splits = np.array_split(bi_image, factor * 10, axis=1)
splits = [np.array_split(split, factor * 10, axis=0) for split in splits]

model = keras.models.load_model(
    'DRCSR.h5', 
    custom_objects={
        'PSNR': PSNR,
        'SSIM': SSIM,
        'UpSampling2D': keras2.UpSampling2D,
        'resize_images': keras2.resize_images
    }
)

out = []

start_time = time.time()
c = 0
total = (10 * factor) ** 2
print("HELLo")

for split in splits:
    temp_out = []
    for i in split:
        temp_out.append(np.squeeze(model(np.expand_dims(i, axis=0))))

        c += 1
        percentage = c * 100 // total
        fract = c / total
        elapsed_time = time.time() - start_time
        print(f"\033[F{percentage}% | {c}/{total} | ETA: {round(elapsed_time / fract - elapsed_time, 2)}s\033[K")

    out.append(temp_out)

out = [np.concatenate(i, axis=0) for i in out]
out = np.concatenate(out, axis=1)

print(f"Predicting time: {round(time.time() - start_time, 2)}s")

out_name = "out_001.png"

while os.path.exists(out_name):
    out_name = "out_" + str(int(out_name[4:-4]) + 1).zfill(3) + ".png"

cv2.imwrite(out_name, out * 255)
