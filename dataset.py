import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

batchSize = 8

imgPath = sorted([os.path.join('data/imgs', fname) for fname in os.listdir('data/imgs')])
maskPath = sorted([os.path.join('data/masks', fname) for fname in os.listdir('data/masks')])

xTrain, xTest, yTrain, yTest = train_test_split(imgPath, maskPath, test_size=0.2, random_state=42)

#augmentations
def transform(img):
    img = tf.image.random_brightness(img, delta=.2)
    img = tf.image.random_contrast(img, 0.2, 0.5)
    return img

#load images
def load_img(imgPath, maskPath):
    img = tf.io.read_file(imgPath)      
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(maskPath)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    
    img = transform(img)
    return img, mask

#create dataset
def training(xTrain, xTest, yTrain, yTest):
    trainDs = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))
    trainDs = trainDs.map(lambda img, mask: load_img(img, mask))
    trainDs = trainDs.batch(batchSize)

    valDs = tf.data.Dataset.from_tensor_slices((xTest, yTest))
    valDs = valDs.map(lambda img, mask: load_img(img, mask))
    valDs = valDs.batch(batchSize)

    return trainDs, valDs

