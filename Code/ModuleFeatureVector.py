import cv2
import numpy as np

def ColorHist(img):
    # calculate first histogram and normalize
    hist_img1 = cv2.calcHist([img], [2], None, [256], [0, 256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    #return histogram  
    return hist_img1 


def ComRed(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #grenswaarden voor rode kleur in HSV
    lower_red = np.array([0, 25, 50])
    upper_red = np.array([30, 255, 255])  
    #masker maken
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #controleren of er rode pixels zijn
    if cv2.countNonZero(mask) == 0:
        return 0, 0
    #binair beeld bekomen
    binary = cv2.bitwise_and(img, img, mask=mask)
    #zwaartepunt binair beeld
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return 0, 0


def ColorPatches(img):
    # stel matrix op 4 rijen, 8 kolommen, 6 elementen (lowerb_h,lowerb_s,lowerb_v,higherb_h,higherb_s,higherb_v) 
    arrColor=np.array([[[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0]],
                    [[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0]],
                    [[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0],[0, 0, 0,0,0,0]]])
    # empty arrar for color values
    arrValues=np.array([[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]])
    # for loop om waarden toe te kennen 0-180, 0-255, 50-255
    amount_h=25.71     #H=180 over 7 patches
    amount_s=85        #S=255 over 3 patches
    # patches for full HSV color range
    for i in range(arrColor.shape[0]):
        for j in range(arrColor.shape[1]):
            arrColor[i, j] = np.array([amount_h * j, amount_s * i, 50, amount_h * (j + 1), amount_s * (i + 1), 255]) 
    # Convert the image to HSV:
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #calculate percentage per patch
    for i in range(arrColor.shape[0]):
        for j in range(arrColor.shape[1]):
            # Calculate treshholds
            lowerValues = np.array([arrColor[i, j,0],arrColor[i, j,1],arrColor[i, j,2]]) 
            upperValues = np.array([arrColor[i, j,3],arrColor[i, j,4],arrColor[i, j,5]]) 
            boundaries= [([lowerValues[0], lowerValues[1], lowerValues[2]], 
                        [upperValues[0], upperValues[1],upperValues[2]])]          
            # Create HSV mask
            hsvMask = cv2.inRange(hsvImage, lowerValues, upperValues)
            # AND mask & input image
            hsvOutput = cv2.bitwise_and(img, img, mask=hsvMask)
            # ratio 
            hsvOutput = cv2.cvtColor(hsvOutput, cv2.COLOR_BGR2GRAY)
            ratio = cv2.countNonZero(hsvOutput) / (img.shape[0] * img.shape[1])
            # This is the color percent calculation
            colorPercent = ratio * 100
            # Assign value to empty array
            arrValues[i, j]=colorPercent
    return arrValues