import os
import pickle
import keras
from keras.datasets import cifar10


def load_data(num_classes, data_path):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    processed_data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    return processed_data


def write_data_to_pickle(data):

    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(data, f)


def main():

    num_classes = 10

    processed_data_path = os.path.join(os.getcwd(), 'processed_data.pkl')

    if os.path.exists(processed_data_path):
        print('data already exists')

    else:
        data_path = ''

        data = load_data(num_classes, data_path)

        write_data_to_pickle(data)

        print(data['x_train'].shape)
        print(data['y_train'].shape)
        print(data['x_test'].shape)
        print(data['y_test'].shape)


if __name__ == '__main__':
    main()
