import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img1 = cv.imread('files/goldengate-00.png', 0)          # queryImage
img2 = cv.imread('files/goldengate-01.png', 0) # trainImage


cv.waitKey()
# Initiate SURF
surf = cv.xfeatures2d.SURF_create()

# find the keypoints and descriptors with ORB
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_L1,crossCheck=False)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
out = 0
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:30],out, flags=2 )
cv.imshow("Test", img3)
cv.waitKey()

cv.imwrite("matches.png", img3)
