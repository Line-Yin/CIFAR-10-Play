import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

image = load_img('cat.jpg')
img_arr = img_to_array(image)

print(img_arr.shape)

img_arr = np.expand_dims(img_arr, axis=0)

i = 0
for batch in datagen.flow(img_arr,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix='cat',
                          save_format='jpeg'):
    i += 1
    if i > 32:
        break


