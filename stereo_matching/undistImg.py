import cv2
import numpy as np


imgL = cv2.imread('image4.png',0)
camMat = [408.769942172, 0.0, 629.645766058, 0.0, 408.769942172, 361.84845779, 0.0, 0.0, 1.0]
distCoeffs = [0.0196331022067, -0.0542073038238, 0.000869497661924, -0.000465375594747, 0.0139493320243]
camMat = np.reshape(np.array(camMat),(3,3))
distCoeffs = np.array(distCoeffs)
#print camMat
#print distCoeffs
undistImg = cv2.undistort(imgL,cameraMatrix=camMat,distCoeffs=distCoeffs)
cv2.imwrite("image4-undist.png",undistImg)
cv2.imshow('origin',imgL)
cv2.imshow("undist",undistImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(imgL)
# plt.imshow(undistImg)
# plt.show()
