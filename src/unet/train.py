from __future__ import print_function
import os

from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
import unet


# Settings for unet
batch_size = 32
num_classes = 10
epochs = 1
learning_rate = 1E-4
weight_decay = 1E-4

plot_architecture = True
save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'trained_model_unet.h5'
model_path = os.path.join(save_dir, model_name)

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# process data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = x_train.shape[1:]

print(input_shape)

# if model not exists
model_exists = False

if not model_exists:

    # build network
    model = unet.unet(input_shape, num_classes, weight_decay)

    # train the model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

    # save model
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # plot model
    net_plot_path = os.path.join(os.path.join(os.getcwd(), 'figures'))
    print(net_plot_path)
    if plot_architecture:
        if not os.path.isdir(net_plot_path):
            os.mkdir(net_plot_path)
        plot_model(model, to_file=os.path.join(net_plot_path, 'unet.png'), show_shapes=True)

else:

    print('..')

    model = load_model(model_path)

    # score trained model
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss: ' + str(scores[0]))
    print('Test accuracy: ' + str(scores[1]))

