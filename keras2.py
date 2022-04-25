import tensorflow as tf

# from keras import activations
from keras import backend
# from keras import constraints
# from keras import initializers
# from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
# from keras.layers.pooling import AveragePooling1D
# from keras.layers.pooling import AveragePooling2D
# from keras.layers.pooling import AveragePooling3D
# from keras.layers.pooling import MaxPooling1D
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from keras.utils import conv_utils
# from keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.UpSampling2D')
class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data
    by `size[0]` and `size[1]` respectively.
    Examples:
    >>> input_shape = (2, 2, 1, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[ 0    1    2]]
        [[ 3    4    5]]]
     [[[ 6    7    8]]
        [[ 9 10 11]]]]
    >>> y = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
    >>> print(y)
    tf.Tensor(
        [[[[ 0    1    2]
             [ 0    1    2]]
            [[ 3    4    5]
             [ 3    4    5]]]
         [[[ 6    7    8]
             [ 6    7    8]]
            [[ 9 10 11]
             [ 9 10 11]]]], shape=(2, 2, 2, 3), dtype=int64)
    Args:
        size: Int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        interpolation: A string, one of `nearest` or `bilinear`.
    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
                `(batch_size, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
                `(batch_size, channels, rows, cols)`
    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
                `(batch_size, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
                `(batch_size, channels, upsampled_rows, upsampled_cols)`
    """

    def __init__(self,
                             size=(2, 2),
                             data_format=None,
                             interpolation='nearest',
                             **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        if interpolation not in {'nearest', 'bilinear', 'bicubic'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` '
                                             f'or `"bilinear"`. Received: "{interpolation}".')
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[
                    2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[
                    3] if input_shape[3] is not None else None
            return tf.TensorShape(
                    [input_shape[0], input_shape[1], height, width])
        else:
            height = self.size[0] * input_shape[
                    1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[
                    2] if input_shape[2] is not None else None
            return tf.TensorShape(
                    [input_shape[0], height, width, input_shape[3]])

    def call(self, inputs):
        return backend.resize_images(
                inputs, self.size[0], self.size[1], self.data_format,
                interpolation=self.interpolation)

    def get_config(self):
        config = {
                'size': self.size,
                'data_format': self.data_format,
                'interpolation': self.interpolation
        }
        base_config = super(UpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# import collections
# import itertools
# import json
# import os
# import random
# import sys
# import threading
# import warnings
# import weakref

import numpy as np

# from tensorflow.core.protobuf import config_pb2
# from tensorflow.python.eager import context
# from tensorflow.python.eager.context import get_config
# from tensorflow.python.framework import config
# from keras import backend_config
# from keras.distribute import distribute_coordinator_utils as dc
# from keras.engine import keras_tensor
# from keras.utils import control_flow_util
# from keras.utils import object_identity
# from keras.utils import tf_contextlib
# from keras.utils import tf_inspect
# from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@keras_export('keras.backend.permute_dimensions')
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.
    Args:
            x: Tensor or variable.
            pattern: A tuple of
                    dimension indices, e.g. `(0, 2, 1)`.
    Returns:
            A tensor.
    Example:
        >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        >>> a
        <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
        array([[ 1,    2,    3],
                     [ 4,    5,    6],
                     [ 7,    8,    9],
                     [10, 11, 12]], dtype=int32)>
        >>> tf.keras.backend.permute_dimensions(a, pattern=(1, 0))
        <tf.Tensor: shape=(3, 4), dtype=int32, numpy=
        array([[ 1,    4,    7, 10],
                     [ 2,    5,    8, 11],
                     [ 3,    6,    9, 12]], dtype=int32)>
    """
    return tf.compat.v1.transpose(x, perm=pattern)


@keras_export('keras.backend.resize_images')
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def resize_images(x, height_factor, width_factor, data_format,
                                    interpolation='nearest'):
    """Resizes the images contained in a 4D tensor.
    Args:
            x: Tensor or variable to resize.
            height_factor: Positive integer.
            width_factor: Positive integer.
            data_format: One of `"channels_first"`, `"channels_last"`.
            interpolation: A string, one of `nearest` or `bilinear`.
    Returns:
            A tensor.
    Raises:
            ValueError: in case of incorrect value for
                `data_format` or `interpolation`.
    """
    if data_format == 'channels_first':
        rows, cols = 2, 3
    elif data_format == 'channels_last':
        rows, cols = 1, 2
    else:
        raise ValueError('Invalid `data_format` argument: %s' % (data_format,))

    new_shape = x.shape[rows:cols + 1]
    if new_shape.is_fully_defined():
        new_shape = tf.constant(new_shape.as_list(), dtype='int32')
    else:
        new_shape = tf.shape(x)[rows:cols + 1]
    new_shape *= tf.constant(
            np.array([height_factor, width_factor], dtype='int32'))

    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
    if interpolation == 'nearest':
        x = tf.image.resize(
                x, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    elif interpolation == 'bilinear':
        x = tf.image.resize(
                x, new_shape, method=tf.image.ResizeMethod.BILINEAR)
    elif interpolation == 'bicubic':
        x = tf.image.resize(
                x, new_shape, method=tf.image.ResizeMethod.BICUBIC)
    else:
        raise ValueError('interpolation should be one '
                                         'of "nearest" or "bilinear".')
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 3, 1, 2])

    return x
