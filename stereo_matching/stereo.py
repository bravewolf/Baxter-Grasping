import cv2
import numpy as np
from matplotlib import pyplot as plt
 
imgL = cv2.imread('image1.png',0)
imgR = cv2.imread('image4.png',0)
 
stereo = cv2.StereoSGBM(minDisparity=0, numDisparities=256, SADWindowSize=21,P1=800,P2=3200)
disparity = stereo.compute(imgL,imgR)
#print "disparity ",disparity[670,385]
#plt.imshow(disparity,'gray')
#plt.show()
camMat = np.reshape(np.array([4.0364801365017615e+02, 0., 6.3146928830350316e+02, 0.,
       4.0620637674934119e+02, 3.6213656430315433e+02, 0., 0., 1. ]),(3,3))
distCoeffs = np.array([2.5817367285864371e-02, -6.4939093957227453e-02,
       -1.6997464300167172e-04, -8.7200818996337727e-05,
       1.8531601637796945e-02 ])
Q =np.zeros((4,4))
R = np.eye(3)
T = np.array([-0.02,0,0])
cv2.stereoRectify(camMat,distCoeffs,camMat,distCoeffs,(1280,800),R,T,Q=Q)
print Q

#reproject to 3D
# outImg = np.ones((1280,800,3))
outImg = cv2.reprojectImageTo3D(disparity=disparity,Q=Q,handleMissingValues=False)
plt.imshow(outImg[:,:,0],'gray')
plt.show()
# plt.show()
print outImg[700,450]
print outImg[680,500]
print outImg[680,300]
