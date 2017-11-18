import numpy as np
import cv2
from matplotlib import pyplot as plt
 
imgL = cv2.imread('image1.png',0)
imgR = cv2.imread('image4.png',0)
 
stereo = cv2.StereoSGBM(minDisparity=0, numDisparities=64, SADWindowSize=21,P1=800,P2=3200)
disparity = stereo.compute(imgL,imgR)

#print "disparity ",disparity[670,385]
plt.imshow(disparity,'gray')
plt.show()
