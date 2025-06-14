import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Dropout
from keras.layers import Conv2DTranspose, concatenate, Input, Activation


#encoder block (establishing skip connections)
def contraction(inputs=None, filters=64, batchNorm=False, dropout=0):
    #conv1 -> leakyrelU -> conv2 -> leakyrelu -> maxpool

    #first Convolution
    conv = Conv2D(filters,3,padding="same")(inputs)
    if batchNorm:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)

    #second convolution
    conv = Conv2D(filters,3,padding="same")(conv)
    if batchNorm:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)

    #prevent overfitting
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    
    skipConnection = conv
    conv = MaxPool2D(strides=2)(conv)
    return conv, skipConnection

#decoder block (filters: start filter size * 16)
def expansion(inputs=None, filters=1024, batchNorm=False, dropout=0):
    #conv1 -> leakyrelU -> conv2 -> leakyrelu -> 2x2 upsample

    #first Convolution
    conv = Conv2D(filters,3,padding="same")(inputs)
    if batchNorm:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)

    #second convolution
    conv = Conv2D(filters,3,padding="same")(conv)
    if batchNorm:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)

    #prevent overfitting
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    
    
    conv = Conv2DTranspose(filters,3,strides=2, padding="same")(conv)
    return conv

def encoder(inputs=None, filters=64, batchNorm=False, dropouts=np.zeros(8)):
    block = inputs
    blocks = []
    for i in range(4):
        block, skip = contraction(block, filters, batchNorm, dropouts[i])
        filters *= 2
        blocks.append((block, skip))
    return blocks

#(filters: start filter size * 16)
def decoder(inputs=None, filters=1024, batchNorm=False, dropouts=np.zeros(8)):
    #start from correct dropout idx
    buffer = 4

    skipConnections = [block[1] for block in inputs][::-1]
    blocks = [block[0] for block in inputs][::-1]
    currBlock = blocks[0]

    for i in range(4):
        currBlock = expansion(blocks[i], filters, batchNorm, dropouts[i + buffer])
        filters //= 2
        currBlock = concatenate((skipConnections[i], currBlock))
    return currBlock

def unetModel(size=(256, 256, 3), filters = 64, classes = 1, batchNorm = True, dropouts=np.zeros(8)):
    inputs = Input(shape=size)
    blocks = encoder(inputs, filters, batchNorm, dropouts)
    output = decoder(blocks, filters*16, batchNorm, dropouts)

    if classes < 2:
        output = Conv2D(1,1, padding="same" )(output)
        output = Activation('sigmoid')(output)
    else:
        output = Conv2D(classes,1, padding="same" )(output)
        output = Activation('softmax')(output)
    
    model = tf.keras.Model(inputs = inputs, outputs=output, name="U-Net")
    return model

