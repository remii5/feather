import dataset
import model
import os
from sklearn.model_selection import train_test_split
from keras.losses import BinaryCrossentropy
from keras.optimizers import SGD
import numpy as np

epochs = 20
batchSize = 8


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
    loss = BinaryCrossentropy(),
    optimizer = SGD(learning_rate=1e-4),
    metrics=['accuracy']
)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)
unet.fit(xTrain, yTrain, epochs, batchSize, validation_data = valDs)
unet.save('unet.keras')