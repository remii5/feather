import dataset
import model
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.optimizers import Adam
from keras.metrics import MeanIoU, Precision, Recall

#generalize dice loss function
#more robust for an imbalance than BCE
def gdl(yTrue, yPred, smooth = 1e-6):
    yTrue = tf.cast(yTrue, dtype=tf.float32)
    yPred = tf.cast(yPred, dtype=tf.float32)

    yTrue = tf.reshape(yTrue, [-1])
    yPred = tf.reshape(yPred, [-1])

    w = 1.0/(tf.reduce_sum(yTrue)**2 + smooth)
    top = tf.reduce_sum(w * yTrue * yPred)
    bot = tf.reduce_sum(w * (yTrue + yPred))

    loss = 1 - (2.0 * top + smooth)/(bot + smooth)
    return loss
if __name__ == "__main__":

    epochs = 100


    img_dir = './data/imgs'
    mask_dir = './data/masks/SegmentationClass'

    # Get basenames (without extension) for both
    img_names = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')}
    mask_names = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith('.png')}

    valid_names = sorted(img_names & mask_names, key=lambda x: int(x.replace('img', '')))

    # Generate full paths
    imgPath = [os.path.join(img_dir, f"{name}.png") for name in valid_names]
    maskPath = [os.path.join(mask_dir, f"{name}.png") for name in valid_names]

    print(f"Using {len(imgPath)} matched image/mask pairs.")


    xTrain, xTest, yTrain, yTest = train_test_split(imgPath, maskPath, test_size=0.2, random_state=42)


    trainDs, valDs = dataset.training(xTrain, xTest, yTrain, yTest)

    unet = model.unetModel()


    unet.compile(
    loss = gdl,
    optimizer = Adam(learning_rate=1e-4),
    metrics=['accuracy',
             MeanIoU(num_classes = 2),
             Precision(),
             Recall()]
    )


    unet.fit(trainDs, epochs=epochs, validation_data = valDs)
    unet.save('unet.keras')

