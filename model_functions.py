
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib as plt

def load_data(classes,batch_size,train_path,test_path,valid_path):
    train_batches = ImageDataGenerator().flow_from_directory(directory=train_path,
                                                             classes=classes,
                                                             class_mode='categorical',
                                                             target_size=(81, 81),
                                                             batch_size=batch_size,
                                                             shuffle=True)

    valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path,
                                                             classes=classes,
                                                             class_mode='categorical',
                                                             target_size=(81, 81),
                                                             batch_size=batch_size,
                                                             shuffle=True)

    test_batches = ImageDataGenerator().flow_from_directory(directory=test_path,
                                                            classes=classes,
                                                            class_mode='categorical',
                                                            target_size=(81, 81),
                                                            batch_size=batch_size,
                                                            shuffle=False)
    return train_batches,valid_batches,test_batches


def built_model_vgg16(input_shape,num_classes):

    # vgg16_model = VGG16()
    vgg16_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)

    model = Sequential()

    for layer in vgg16_model.layers:
      model.add(layer)

    model.summary()

    conv_model = Sequential()

    for layer in vgg16_model.layers[:-6]:
      conv_model.add(layer)

    conv_model.summary()

    for layer in conv_model.layers:
        layer.trainable = False

    transfer_layer = model.get_layer('block5_pool')

    # define the conv_model inputs and outputs
    conv_model = Model(inputs=conv_model.input,
                       outputs=transfer_layer.output)


    # # the 5 classes:
    num_classes = num_classes

    # start a new Keras Sequential model.
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

    loss = 'categorical_crossentropy'
    print("compile_model")
    new_model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return new_model

def fit_my_model(new_model,train_batches,valid_batches,test_batches,epochs,callback):


    step_size_train=train_batches.n//train_batches.batch_size
    step_size_valid=valid_batches.n//valid_batches.batch_size

    history = new_model.fit_generator(train_batches,
                            epochs=epochs,
                            steps_per_epoch=step_size_train,
                            validation_data=valid_batches,
                            validation_steps=step_size_valid,
                            callbacks = [callback],
                            verbose=1)

    step_size_test=test_batches.n//test_batches.batch_size

    result = new_model.evaluate_generator(test_batches, steps=step_size_test)

    print("Test set classification accuracy: {0:.2%}".format(result[1]))
    return new_model,history



batch_size = 32

num_classes = 5

train_path='dataset/train'
test_path='dataset/test'
valid_path='dataset/validate'

classes = ['airplane', 'bird', 'drone', 'helicopter', 'other']
input_shape=(81,81,3)

es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=2,
                   verbose=1,
                   mode='auto')


train_batches,valid_batches,test_batches=load_data(classes,batch_size,train_path,test_path,valid_path)
my_model=built_model_vgg16(input_shape,num_classes)
model,history=fit_my_model(my_model,train_batches,valid_batches,test_batches,30,es)
print("save_model")
model.save('model_vgg_categorical_s81_2.h5')
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()