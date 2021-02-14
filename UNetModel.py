import keras
from keras.models import Sequential,Model, Input
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, UpSampling2D, concatenate, Input
from keras.optimizers import Adam, SGD
import tensorflow as tf


def unet(input_size, pretrained_weights=None,):

    input_shape = input_size
    inputs= Input(input_shape)
    conv1 = Conv2D(64, kernel_size=(7,7), activation='relu', kernel_initializer='he_normal',padding='same', data_format='channels_last',input_shape=input_shape) (inputs)
    pool1 =MaxPool2D(pool_size=(2,2), strides=(2,2),  padding='same', data_format='channels_last')(conv1)
    conv2 = Conv2D(128, kernel_size=(7,7), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(conv2)
    conv3 = Conv2D(256, kernel_size=(7,7), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2),  padding='same' )(conv3)
    conv4 = Conv2D(512,kernel_size=(7,7),activation='relu', padding='same', kernel_initializer='he_normal' )(pool3)
    drop4 = Dropout(0.5)(conv4)
    up5=UpSampling2D(size=(2,2))(drop4)
    conv5 = Conv2D(256, kernel_size=(7,7), activation='relu', padding='same', kernel_initializer='he_normal')(up5)
    merge5 = concatenate([conv5, conv3],  axis=-1)
    conv6 = Conv2D(256, kernel_size=(7,7), activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    up7 = UpSampling2D(size=(2,2))(conv6)
    conv7=Conv2D(128, kernel_size=(7,7), activation='relu',padding='same', kernel_initializer='he_normal')(up7)
    merge7= concatenate([conv7, conv2])
    conv8= Conv2D(128, kernel_size=(7,7), kernel_initializer='he_normal', padding='same', activation='relu')(merge7)
    up9= UpSampling2D(size=(2,2))(conv8)
    conv9 =Conv2D(64, kernel_size=(7,7), kernel_initializer='he_normal', padding='same',activation='relu')(up9)
    merge9 = concatenate([conv9, conv1])
    conv10 = Conv2D(64, kernel_size=(7,7), kernel_initializer='he_normal', padding='same',activation='relu')(merge9)
    conv11 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv12 = Conv2D(1, 1, activation='sigmoid')(conv11)
    model = Model(inputs = inputs, outputs = conv12)
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    optimizer = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(name='meanIoU',num_classes=2)], run_eagerly=False)
    return model

