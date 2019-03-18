# OS Imports
from os import listdir
from os.path import isfile, join, exists


# Imports
import cv2
import numpy as np 
import argparse
import imutils


from tqdm import tqdm

# CUDA
from numba import vectorize
from numba import *

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')

surf = cv2.xfeatures2d.SURF_create()
bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
kernel = np.ones((5,5),np.float32)/25

maxsize = 32767 

def getFileNames(path):
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    return files

def showImg(img, screenName="default"):
    cv2.imshow(screenName, img)
    cv2.waitKey()

def match(img1, img2, conf=0.7, mingood=4):
    kp1, des1 = surf.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = surf.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    matches = bf.knnMatch(des2, des1, k = 2)

    good = []
    for m,n in matches:
        if m.distance < conf*n.distance:
            good.append((m.trainIdx, m.queryIdx))

    if len(good) > mingood:
        curr = np.float32([kp2[i].pt for (_, i) in good])
        prev = np.float32([kp1[i].pt for (i, _) in good])
        H, s = cv2.findHomography(curr, prev, cv2.RANSAC, 4)
        return H
    else:
        return "err"

#@nb.vectorize(['uint8(uint8, uint8)'], target='cuda')
@jit(nopython=True)
def mix_and_match(currImg, newImg):
    y1, x1 = currImg.shape[:2]
    for i in range(0, x1):
        for j in range(0, y1):
            if (currImg[j,i] == np.array([0,0,0])).all() and  (newImg[j,i] == np.array([0,0,0])).all():
                newImg[j,i] = [0, 0, 0]
            else:
                if (newImg[j,i] == np.array([0,0,0])).all():
                    newImg[j,i] = currImg[j,i]
                else:
                    if not (currImg[j,i] ==  np.array([0,0,0])).all():
                        bw, gw, rw = newImg[j,i]
                        bl,gl,rl = currImg[j,i]
                        newImg[j, i] = [bl,gl,rl]
    return newImg

def main(path, debug=False, crop=False, extension=""):
    # Read images
    print("Reading Images")
    filenames = getFileNames(path)
    if debug: print(filenames)
    if extension:
        filenames = [fname for fname in filenames if extension in fname]
        if debug: print(filenames)
    imgs = [cv2.resize(cv2.imread(x), (450, 300)) for x in filenames]

    # Preprocess the images in lists
    print("Preprocessing data")
    centre = len(imgs) // 2
    left_imgs   = []
    right_imgs  = []

    for i in tqdm(range(len(imgs))):
        if i <= centre:
            left_imgs.append(imgs[i])
        else:
            right_imgs.append(imgs[i])
    
    # Start going to the left (reverse the homography matrix)
    print("Starting pano from the left")
    currPano = left_imgs[0]
    for img in tqdm(left_imgs[1:]):
        # Need to do inverse homography because going to the left

        H = match(currPano, img)
        if H == "err":
            continue
        iH = np.linalg.inv(H)

        # TODO: Find the variable names
        f1 = np.dot(iH, np.array([0, 0, 1]))
        f1 = f1/f1[-1]

        iH[0][-1] += abs(f1[0])
        iH[1][-1] += abs(f1[1])

        ds = np.dot(iH, np.array([currPano.shape[1], currPano.shape[0],1]))
        offset_y = abs(int(f1[1]))
        offset_x = abs(int(f1[0]))

        dsize = (int(ds[0]) + offset_x, int(ds[1]) + offset_y)
        tmp = cv2.warpPerspective(currPano, iH, dsize)
        tmp[offset_y:img.shape[0]+offset_y, offset_x:img.shape[1]+offset_x] = img
        currPano = tmp

        if debug:
            showImg(currPano)
    

    
    # Continue going to the right homography
    print("Continuing to the right")

    for img in tqdm(right_imgs):
        H = match(currPano, img)
        
        ds = np.dot(H, np.array([img.shape[1], img.shape[0],1]))
        ds = ds/ds[-1]

        dsize = (int(ds[0]) + currPano.shape[1], int(ds[1]) + currPano.shape[0])
        tmp = cv2.warpPerspective(img, H, dsize)

        #print("Passing to gpu")
        tmp = mix_and_match(currPano, tmp)
        currPano = tmp

        if debug:
            showImg(currPano)
    
    if crop:
        # Crop the image to look nice
        print("Cropping")
        currPano = cv2.copyMakeBorder(currPano, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(currPano, cv2.COLOR_BGR2GRAY)
        gray = cv2.filter2D(gray, -1, kernel)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        
        # Debug
        if debug:
            showImg(currPano, "Pano")
            showImg(gray, "Gray")
            showImg(thresh, "Threshhold")

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minRect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
            #showImg(minRect)
            #showImg(sub)

        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        (x, y, w, h) = cv2.boundingRect(c)
        currPano = currPano[y:y+h, x:x+w]

    print("Showing image")
    showImg(currPano)

    print("Saving Image")
    fname = filenames[0].split("\\")[-1].split(".")[0]
    cv2.imwrite(fname + "_pano.png", currPano)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CV Final Project")
    parser.add_argument("-f", "--folder", nargs='?', default="files/", type=str)
    parser.add_argument('-t', "--type", help='File type')
    parser.add_argument('--debug', action='store_true', help='Add debug statements')
    parser.add_argument('--crop', action='store_true', help='Auto crop the image')

    args = parser.parse_args()

    if args.type:
        main(args.folder, args.debug, args.crop, args.type)
    else:
        main(args.folder, args.debug, args.crop)