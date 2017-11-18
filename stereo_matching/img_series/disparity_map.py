import numpy as np
import cv2
from matplotlib import pyplot as plt

camMat = np.reshape(np.array([4.0364801365017615e+02, 0., 6.3146928830350316e+02, 0.,
       4.0620637674934119e+02, 3.6213656430315433e+02, 0., 0., 1. ]),(3,3))
distCoeffs = np.array([2.5817367285864371e-02, -6.4939093957227453e-02,
       -1.6997464300167172e-04, -8.7200818996337727e-05,
       1.8531601637796945e-02 ])

imgL = cv2.imread('image-61.png',0)
imgR = cv2.imread('image-59.png',0)
#imgL = cv2.undistort(imgL,cameraMatrix=camMat,distCoeffs=distCoeffs)
#imgR = cv2.undistort(imgR,cameraMatrix=camMat,distCoeffs=distCoeffs)
stereo = cv2.StereoSGBM(minDisparity=0, numDisparities=64, SADWindowSize=11,P1=800,P2=3200)
disparity = stereo.compute(imgL,imgR)
#disparity /= 16
print "disparity ",disparity[102,180]
plt.imshow(disparity,'gray')
plt.show()
