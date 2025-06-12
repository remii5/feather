import tensorflow as tf
import numpy as np
from keras.models import load_model
import deconstruct
from PIL import Image


#non-local context dependent algo
def reconstruct(defectMap, imgArr, D = 5, R = 100, threshold=1000):
    if D % 2 == 0:
        raise ValueError("Size must be an odd number.")
    if D > R:
        raise ValueError("D must be less than R.")
    defectCoords = np.nonzero(defectMap)
    
    bestSum = float('inf')
    bestPatch = None
    newImg = imgArr.copy()

    halfD = D//2
    halfR = R//2

    for coordY, coordX in zip(*defectCoords):
        x1, x2 = max(0, (coordX - halfD)), min(imgArr.shape[0], (coordX + halfD + 1)) 
        y1, y2 = max(0, (coordY - halfD)), min(imgArr.shape[1], (coordY + halfD + 1))

        #D matrix around defective pixel and original image pixel
        ci = imgArr[y1:y2, x1:x2]
        ciMask = defectMap[y1:y2, x1:x2]

        ctxStartC = max(0, (coordX - halfR))
        ctxEndC = min(imgArr.shape[1], (coordX + halfR + 1))
        ctxStartR = max(0, (coordY - halfR))
        ctxEndR = min(imgArr.shape[0], (coordY + halfR + 1))

    
        for row in range(ctxStartR + halfD, ctxEndR - halfD + 1):
            for col in range(ctxStartC + halfD, ctxEndC - halfD+ 1):
                nj = imgArr[row-halfD:row+halfD + 1, col-halfD:col+halfD+1]
                njMask = defectMap[row-halfD:row+halfD + 1, col-halfD:col+halfD+1]
                
                if nj.shape != ci.shape or np.any(njMask == 1):
                    continue

                #sum of squares (non-defective ciMask pixels and corresponding nj)
                validMask = (ciMask == 0)
                currSum = np.sum((nj - ci)[validMask]**2)

                if currSum < bestSum:
                    bestSum = currSum
                    bestPatch = nj.copy()

        if bestPatch is not None and bestSum < threshold:
            replacement = newImg[y1:y2, x1:x2]
            replacement[ciMask==1] = bestPatch[ciMask==1]
        else:
            newImg [coordX, coordY] = np.median(ci)

    return newImg
    
        

if __name__ == "__main__":
    imgInputPath = '/input/image'
    model = load_model('unet.keras')
    defectMap, imgNp, img8bit = deconstruct.deconstruct(model, imgInputPath)
    deconstruct.plot(defectMap, imgInputPath)
    img = Image.fromarray(reconstruct(defectMap, imgNp))
    img.show()
    






