


from keras.preprocessing import image
import numpy as np
from keras.models import load_model


def model(path):
    saved_model = load_model("model_vgg2.h5")
    img = image.load_img(path, target_size=(224, 224))
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

