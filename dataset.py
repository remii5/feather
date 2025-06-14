import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


batchSize = 5

img_dir = './data/imgs'
mask_dir = './data/masks/SegmentationClass'

# Get basenames (without extension) for both
img_names = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')}
mask_names = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith('.png')}

# Keep only filenames that exist in both
valid_names = sorted(img_names & mask_names, key=lambda x: int(x.replace('img', '')))

# Generate full paths
imgPath = [os.path.join(img_dir, f"{name}.png") for name in valid_names]
maskPath = [os.path.join(mask_dir, f"{name}.png") for name in valid_names]

xTrain, xTest, yTrain, yTest = train_test_split(imgPath, maskPath, test_size=0.2, random_state=42)

#augmentations
def transform(img):
    img = tf.image.random_brightness(img, max_delta=.2)
    img = tf.image.random_contrast(img, 0.2, 0.5)
    return img

#load images
def load_img(imgPath, maskPath):
    img = tf.io.read_file(imgPath)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (256,256))
    img = transform(img)

    mask = tf.io.read_file(maskPath)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, (256,256))
    mask = tf.where(mask > 0.1, 1.0, 0.0)
    
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

