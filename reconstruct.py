import tensorflow as tf
import numpy as np
from keras.models import load_model
import train
import deconstruct
from PIL import Image
import matplotlib.pyplot as plt


#non-local context dependent algo
def reconstruct(defectMap, imgArr, D = 5, R = 100, threshold=100):
    if D % 2 == 0:
        raise ValueError("Size must be an odd number.")
    if D > R:
        raise ValueError("D must be less than R.")
    defectCoords = np.nonzero(defectMap)
    
    
    newImg = imgArr.copy()

    halfD = D//2
    halfR = R//2

    #pad for the edges
    pad_size = R
    imgArrPadded = np.pad(imgArr, pad_size, mode='reflect')
    defectMapPadded = np.pad(defectMap, pad_size, mode='constant')

    for coordY, coordX in zip(*defectCoords):
        paddedY = coordY + pad_size
        paddedX = coordX + pad_size

        bestSum = float('inf')
        bestPatch = None

        y1, y2 = paddedY - halfD, paddedY + halfD + 1
        x1, x2 = paddedX - halfD, paddedX + halfD + 1

        ci = imgArrPadded[y1:y2, x1:x2]
        ciMask = defectMapPadded[y1:y2, x1:x2]

        ctxStartR = paddedY - halfR
        ctxEndR = paddedY + halfR + 1
        ctxStartC = paddedX - halfR
        ctxEndC = paddedX + halfR + 1

        for row in range(ctxStartR + halfD, ctxEndR - halfD):
            for col in range(ctxStartC + halfD, ctxEndC - halfD):
                nj = imgArrPadded[row - halfD:row + halfD + 1, col - halfD:col + halfD + 1]
                njMask = defectMapPadded[row - halfD:row + halfD + 1, col - halfD:col + halfD + 1]

                if nj.shape != (D, D) or np.any(njMask == 1):
                    continue

                validMask = (ciMask == 0)
                if nj.shape != ci.shape or validMask.shape != ci.shape:
                    continue 

                currSum = np.sum((nj - ci)[validMask]**2)

                if currSum < bestSum:
                    bestSum = currSum
                    bestPatch = nj.copy()

        y1o, y2o = coordY - halfD, coordY + halfD + 1
        x1o, x2o = coordX - halfD, coordX + halfD + 1

        if (y1o >= 0 and y2o <= newImg.shape[0]) and (x1o >= 0 and x2o <= newImg.shape[1]):
            if bestPatch is not None and bestSum < threshold:
                newImg[coordY, coordX] = bestPatch[halfD, halfD]
            else:
                newImg[coordY, coordX] = np.median(ci)

    return newImg
    
        

if __name__ == "__main__":
    imgInputPath = './input/image.png'
    model = load_model('unet.keras', custom_objects={'gdl': train.gdl})
    defectMap, imgNp, img8bit = deconstruct.deconstruct(model, imgInputPath)
    deconstruct.plot(defectMap, imgNp)
    assert defectMap.shape == imgNp.shape, "Mask and image sizes must match!"
    reconstructed = reconstruct(defectMap, imgNp)
    diff = np.abs(reconstructed - imgNp)

    plt.imshow(diff, cmap='hot')
    plt.colorbar()
    plt.title("Difference Image")
    plt.show()

    changed = np.sum(~np.isclose(reconstructed, imgNp, atol=1e-3))
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Reconstructed Image")
    plt.show()

    print("Pixels changed:", changed)
    img = Image.fromarray((reconstructed * 255).astype(np.uint8), mode="L")
    img.save("./output/reconstructed.png")
    img.show()  


