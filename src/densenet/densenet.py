from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from keras.optimizers import *


def conv_factory(x, concat_axis, nb_filter, dropout_rate, weight_decay):

    """BatchNorm -> Relu -> 3x3 Conv2D (Dropout Optionally)"""

    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer='he_uniform',
               padding='same',
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4):

    """BatchNorm -> Relu -> 1x1 Conv2D (Drop Optionally)"""

    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1),
               kernel_initializer='he_uniform',
               padding='same',
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):

    """Calculate x layer by layer in a denseblock"""

    list_fest = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, nb_filter, dropout_rate, weight_decay)
        list_fest.append(x)
        # Concatenate each output x from list_fest
        x = Concatenate(axis=concat_axis)(list_fest)
        nb_filter += growth_rate

    return x, nb_filter


def densenet(num_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate, learning_rate, weight_decay=1E-4):

    """Build a 3 dense block DenseNet"""

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    assert (depth-4) % nb_dense_block == 0, 'Depth must be 3N + 4'

    nb_layers = int((depth-4) / nb_dense_block)

    model_input = Input(shape=img_dim)  # check it

    # Initial Conv2D
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer='he_uniform',
               padding='same',
               name='initial_conv2d',
               kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense block, nb_dense_block-1 because last dense block doesn't followed by transition layer
    for i in range(nb_dense_block-1):
        # Block
        x, nb_filter = denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate, weight_decay)
        # Transition
        x = transition(x, concat_axis, nb_filter, dropout_rate, weight_decay)

    # Check the *(nb_filter)

    # The last dense block BatchNorm -> Relu -> Pooling -> Classification
    x, nb_filter = denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate, weight_decay)
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay)
                           )(x)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(K.image_data_format())(x)
    x = Dense(num_classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=[x], name='DenseNet')

    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def test_dense_net():

    num_classes = 10
    img_dim = [32, 32, 3]
    depth = 16
    nb_dense_block = 3
    growth_rate = 12
    nb_filter = 16
    dropout_rate = 0.2
    weight_decay = 1E-4

    img = np.arange(3072)
    img = np.reshape(img, (32, 32, 3)).astype('float32')
    img /= 3072

    model = densenet(num_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate, weight_decay)

    return model
