import os
import click
import pickle
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from cnn import cnn
from densenet import densenet
from unet import unet


def load_data(data_path):

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test


@click.command()
@click.option('--model_type',
              prompt='model_type',
              type=click.Choice(['cnn', 'densenet', 'unet']))
@click.option('--augmentation',
              prompt='data augmentation',
              default=False)
@click.option('--architecture',
              prompt='plot architecture',
              default=True)


def main(model_type, augmentation, architecture):

    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed_data.pkl')

    if not os.path.exists(data_path):
        print('data not exists, please generate the data')

    else:
        x_train, y_train, x_test, y_test = load_data(data_path)

        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

    trained_model_dir = os.path.join(os.getcwd(), 'saved_models', model_type)

    if os.path.exists(trained_model_dir) and os.listdir(trained_model_dir):
        print(model_type + ' trained model or model figures are already exists')

    else:
        if not os.path.exists(trained_model_dir):
            os.makedirs(trained_model_dir)

        input_shape = x_train.shape[1:]
        num_classes = y_test.shape[1]

        if model_type == 'cnn':
            """ setting model parameters """
            batch_size = 32
            epochs = 1
            learning_rate = 0.0001
            weight_decay = 1e-6
            dropout_rate = 0.2
            """"""""""""""""""""""""""""""""
            model = cnn.cnn(input_shape, num_classes, learning_rate, weight_decay, dropout_rate)
            model_name = 'trained_model_cnn.h5'

        elif model_type == 'densenet':
            """ setting model parameters """
            batch_size = 32
            epochs = 1
            learning_rate = 0.0001
            weight_decay = 1E-4
            depth = 16
            nb_dense_block = 3
            growth_rate = 12
            nb_filter = 16
            dropout_rate = 0.2
            """"""""""""""""""""""""""""""""
            model = densenet.densenet(num_classes, input_shape, depth, nb_dense_block, growth_rate,
                                      nb_filter, dropout_rate, learning_rate, weight_decay)
            model_name = 'trained_model_densenet.h5'

        elif model_type == 'unet':
            """ setting model parameters """
            batch_size = 32
            epochs = 1
            learning_rate = 0.0001
            weight_decay = 1e-6
            dropout_rate = 0.2
            """"""""""""""""""""""""""""""""
            model = unet.unet(input_shape, num_classes, learning_rate, weight_decay, dropout_rate)
            model_name = 'trained_model_unet.h5'

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)

        print(trained_model_dir)
        trained_model_path = os.path.join(trained_model_dir, model_name)
        print(trained_model_path)

        model.save(trained_model_path)
        print('trained model saved')

        if architecture:
            architecture_path = os.path.join(os.path.join(os.getcwd(), 'figures', model_type))
            if not os.path.exists(architecture_path):
                os.makedirs(architecture_path)
            plot_model(model, to_file=os.path.join(architecture_path, model_type + '.png'), show_shapes=True)
            print('architecture saved')

        print(model_type)
        print(augmentation)
        print(architecture)


if __name__ == '__main__':
    main()
