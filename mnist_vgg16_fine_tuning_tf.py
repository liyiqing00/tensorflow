import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

tf.random.set_seed(42)
np.random.seed(42)


# the model
def pretrained_model(img_shape, num_classes):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    
    # make vgg16 model layers as non trainable
#    model_vgg16_conv.trainable = False
    model_vgg16_conv.trainable = True
    
    set_trainable = False
    for layer in model_vgg16_conv.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    

    # create your own input format
    keras_input = Input(shape=img_shape, name = 'image_input')
    
    # use the generated model 
    output_vgg16_conv = model_vgg16_conv(keras_input)
    
    # add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dense(64, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # create your own model 
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss= 'sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    
    return pretrained_model

# loading the data
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# converting it to RGB
x_train_full = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_train_full]
x_train_full = np.concatenate([arr[np.newaxis] for arr in x_train_full]).astype('float32')
x_train_full = x_train_full / 255.0

x_test = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')
x_test = x_test / 255.0

x_train, x_valid = x_train_full[:-5000], x_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

x_train = x_train[..., np.newaxis]
x_valid = x_valid[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# training the model
model = pretrained_model(x_train.shape[1:], len(set(y_train)))

# training phase
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])

# check constructed model
model.summary()

model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))


# testing phase
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test accuracy ', test_acc)
