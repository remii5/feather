import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def deconstruct(model, imgPath):
    img = tf.io.read_file(imgPath)     
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    imgResize = tf.image.resize(img, (256,256))
    imgInput = tf.expand_dims(imgResize, 0)
    
    pred = model.predict(imgInput, verbose=0)[0] 
    '''
    #full scale image size
    pred_big = tf.image.resize(pred, img.shape[:2],method='nearest')  

    mask = (pred_big[...,0] > 0.5).numpy().astype(np.uint8) 

    imgNp  = np.squeeze(img.numpy())
    img8bit = (imgNp * 255).astype(np.uint8)
    '''
    mask = (pred > .5).astype(np.uint8)
    mask = np.squeeze(mask)

    imgNp = np.squeeze(imgResize.numpy())
    img8bit = (imgNp * 255).astype(np.uint8)

    return mask, imgNp, img8bit



def plot(predMask, img):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(np.squeeze(img), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Defect Mask")
    plt.imshow(predMask, cmap='magma')
    plt.colorbar()
    plt.show()
