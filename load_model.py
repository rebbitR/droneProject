


from keras.preprocessing import image
import numpy as np
from keras.models import load_model

def my_model(path,size,type):
    models=['model_vgg_s81.h5','model_vgg_categorical_s81.h5']
    if type=='category':
        model=models[1]
        classes = ['airplane', 'bird', 'drone', 'helicopter','other']
    elif type=='binary':
        model=models[0]
        classes = ['yes', 'no']
    else:
        print('ERROR')
        return 1,'ERROR'
    saved_model = load_model(model)
    img = image.load_img(path, target_size=(size, size))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = saved_model.predict(img)
    i = np.argmax(output)
    return i,classes[i]

# i,kind=my_model('create_dataset_s81/airplane_close_csv/image_0226.jpg',81,'hbh')
# print(str(i)+' '+kind)

