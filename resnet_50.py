# from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
import tensorflow.keras
import pandas as pd
# import sys
# file = open('logFile.log', 'w')
# sys.stdout = file
IMG_WIDTH=81
IMG_HEIGHT=81
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = restnet.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
restnet = Model(restnet.input,output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()
# ----
from keras.preprocessing.image import ImageDataGenerator
batch_size = 32

train_path = 'dataset/train'
valid_path = 'dataset/validate'
test_path = 'dataset/test'


classes = ['airplane', 'bird', 'drone','helicopter','other']

train_generator = ImageDataGenerator().flow_from_directory(directory=train_path,
                                            classes=classes,
                                            class_mode='categorical',
                                            target_size=(81,81),
                                            batch_size=batch_size,
                                            shuffle=True)

val_generator = ImageDataGenerator().flow_from_directory(directory=valid_path,
                                            classes=classes,
                                            class_mode='categorical',
                                            target_size=(81,81),
                                            batch_size=batch_size,
                                            shuffle=True)

test_generator = ImageDataGenerator().flow_from_directory(directory=test_path,
                                               classes=classes,
                                               class_mode='categorical',
                                               target_size=(81,81),
                                               batch_size=batch_size,
                                               shuffle=False)
# ----
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
# import keras.optimizer_v2.rmsprop as rms
# from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

model = Sequential()
model.add(restnet)

model.add(Dense(512, activation='relu', input_dim=(IMG_HEIGHT,IMG_WIDTH,3)))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])
model.build((None, 81, 81, 3))

model.summary()

checkpoint = ModelCheckpoint('pred_drone_5_classes_restnet_50_2.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

history = model.fit(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=test_generator,
                              validation_steps=50,
                              verbose=1, callbacks=[checkpoint])

# model.save('pred_drone_5_classes_restnet_50_1.h5')

#
# restnet.trainable = True
# set_trainable = False
# for layer in restnet.layers:
#     if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
# layers = [(layer, layer.name, layer.trainable) for layer in restnet.layers]
# pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
#
# # from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
# # from tensorflow.python.keras.models import Sequential
# # # import keras.optimizer_v2.rmsprop as rms
# #
# # from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
# from tensorflow.keras.optimizers import RMSprop
#
# model_finetuned = Sequential()
# model_finetuned.add(restnet)
# model_finetuned.add(Dense(512, activation='relu', input_dim=(IMG_HEIGHT,IMG_WIDTH,3)))
# model_finetuned.add(Dropout(0.3))
# model_finetuned.add(Dense(512, activation='relu'))
# model_finetuned.add(Dropout(0.3))
# model_finetuned.add(Dense(5, activation='softmax'))
# model_finetuned.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(learning_rate=1e-5),
#               metrics=['accuracy'])
#
# model_finetuned.build((None, 81, 81, 3))
#
# model_finetuned.summary()
#
# history_1 = model_finetuned.fit(train_generator,
#                                   steps_per_epoch=100,
#                                   epochs=15,
#                                   validation_data=test_generator,
#                                   validation_steps=20,
#                                   verbose=1)
#
# model_finetuned.save('pred_drone_5_classes_restnet_50.h5')

# https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38
