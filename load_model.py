


from keras.preprocessing import image
import numpy as np
from keras.models import load_model


def model(path,size):
    saved_model = load_model("model_vgg2.h5")
    img = image.load_img(path, target_size=(size, size))
    img = np.asarray(img)

    img = np.expand_dims(img, axis=0)
    # print(img)

    # classes = ['airplain', 'bird', 'drone', 'helicopter']
    output = saved_model.predict(img)

    i = np.argmax(output)
    # print(classes[i])
    return i

# path='airplain_tryModel.jpg'
#
# model(path)

# out:
# airplain

# def model_2(path,size):
#     saved_model = load_model("model_vgg_s81.h5")
#     img = image.load_img(path, target_size=(size, size))
#     img = np.asarray(img)
#
#     img = np.expand_dims(img, axis=0)
#     # print(img)
#
#     classes = ['yes','no']
#     output = saved_model.predict(img)
#
#     i = np.argmax(output)
#     print(classes[i])
#     return i

# path='create_dataset_s81/drone_medium1_csv/image_0000.jpg'
#
# model_2(path,81)
#
# import os
# for filename in os.listdir('try'):
#     path='try'+'/'+filename
#     print(path)
#     model_2(path,81)