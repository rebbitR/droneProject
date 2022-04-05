

import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator



batch_size = 32

# train_path = 'create_dataset_2/train'
# test_path = 'create_dataset_2/test'
# valid_path='create_dataset_2/validate'

train_path='dataset_2_classes/dataset/train'
test_path='dataset_2_classes/dataset/test'
valid_path='dataset_2_classes/dataset/validate'

# classes = ['yes','no']
classes = ['airplain', 'bird', 'drone', 'helicopter','other']

train_batches = ImageDataGenerator().flow_from_directory(directory=train_path,
                                            classes=classes,
                                            class_mode='categorical',
                                            target_size=(81,81),
                                            batch_size=batch_size,
                                            shuffle=True)

valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path,
                                            classes=classes,
                                            class_mode='categorical',
                                            target_size=(81,81),
                                            batch_size=batch_size,
                                            shuffle=True)

test_batches = ImageDataGenerator().flow_from_directory(directory=test_path,
                                               classes=classes,
                                               class_mode='categorical',
                                               target_size=(81,81),
                                               batch_size=batch_size,
                                               shuffle=False)



# still dont runing
# vgg16_model = VGG16()
vgg16_model = VGG16(weights='imagenet',include_top=False,input_shape=(81,81,3))


model = Sequential()

for layer in vgg16_model.layers:
  model.add(layer)

#כדי לראות את מבנה המודל:
model.summary()

conv_model = Sequential()

for layer in vgg16_model.layers[:-6]:
  conv_model.add(layer)

conv_model.summary()

transfer_layer = model.get_layer('block5_pool')

# define the conv_model inputs and outputs
conv_model = Model(inputs=conv_model.input,
                   outputs=transfer_layer.output)

# # # the 2 classes:
# num_classes = 2

# # the 5 classes:
num_classes = 5

# start a new Keras Sequential model.
#יצירת מודל מסוג שכבות
new_model = Sequential()

# add the convolutional layers of the VGG16 model
new_model.add(conv_model)

# flatten the output of the VGG16 model because it is from a
# convolutional layer
new_model.add(Flatten())

# add a dense (fully-connected) layer.
# this is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(81, activation='relu'))

# add a dropout layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test set
new_model.add(Dropout(0.5))

# add the final layer for the actual classification
new_model.add(Dense(num_classes, activation='softmax'))


optimizer = Adam(learning_rate=1e-5)

# loss function should by 'categorical_crossentropy' for multiple classes
# but here we better use 'binary_crossentropy' because we need to distinguish between 2 classes
# loss = 'binary_crossentropy '

# loss = 'binary_crossentropy'
loss = 'categorical_crossentropy'
print("compile_model")
new_model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])



es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=2,
                   verbose=1,
                   mode='auto')

step_size_train=train_batches.n//train_batches.batch_size
step_size_valid=valid_batches.n//valid_batches.batch_size

history = new_model.fit_generator(train_batches,
                        epochs=30,
                        steps_per_epoch=step_size_train,
                        validation_data=valid_batches,
                        validation_steps=step_size_valid,
                        callbacks = [es],
                        verbose=1)

step_size_test=test_batches.n//test_batches.batch_size

result = new_model.evaluate_generator(test_batches, steps=step_size_test)

print("Test set classification accuracy: {0:.2%}".format(result[1]))

test_batches.reset()
predictions = new_model.predict_generator(test_batches,steps=step_size_test,verbose=1)


# # predicted class indices
y_pred = np.argmax(predictions,axis=1)


print("save_model")
new_model.save('model_vgg_categorical_s81.h5')
from sklearn.metrics import confusion_matrix,classification_report
# by the Confusion Matrix and Classification Report of sklearn
y_pred = np.argmax(predictions, axis=1)
print('Confusion Matrix')
print("test_batches.classes: {}".format(test_batches.classes))
print("y_pred: {}".format(y_pred))
# print(confusion_matrix(test_batches.classes, y_pred))
print('Classification Report')
# print(classification_report(test_batches.classes, y_pred, target_names=classes))

from keras.models import load_model
saved_model = load_model("model_vgg_categorical_s81.h5")

saved_model.summary()



