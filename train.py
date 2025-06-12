import dataset
import model
import os
from sklearn.model_selection import train_test_split
from keras.losses import BinaryCrossentropy
from keras.optimizers import SGD

epochs = 5
batchSize = 8



imgPath = sorted([os.path.join('data/imgs', fname) for fname in os.listdir('data/imgs')])
maskPath = sorted([os.path.join('data/masks', fname) for fname in os.listdir('data/masks')])

xTrain, xTest, yTrain, yTest = train_test_split(imgPath, maskPath, test_size=0.2, random_state=42)

trainDs, valDs = dataset.training(xTrain, xTest, yTrain, yTest)

unet = model.unetModel()

unet.compile(
    loss = BinaryCrossentropy(),
    optimizer = SGD(learning_rate=1e-4),
    metrics=['accuracy']
)

unet.fit(xTrain, yTrain, epochs, batchSize, validation_data = valDs)
unet.save('unet.keras')