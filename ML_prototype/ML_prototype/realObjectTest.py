from extractDatum import getEdges2,predictGraspingPoints,getContour2
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
def getCannyEdges(img): #extract canny edge
    edges = cv2.Canny(img, 50, 300)
    return edges
def loadImage(imgPath): #load all image on folder
    listPath = os.listdir(imgPath)
    listPath.sort()
    X = []
    count = 0
    for img in listPath:
        if img == "Readme.txt": #excluse readme
            continue
        else:
            im = cv2.imread(imgPath+img,3)
            if im.shape[0] <= 480 and im.shape[1] <= 640:
                X.append(im)
                count += 1
    return np.asarray(X), count

if __name__ == "__main__":
		
		#enter object name
    # object_class = "thickpencil"
    object_class = "martini"
    # object_class = "eraser"

    coef = np.loadtxt('coef/coef_(30v500).txt',delimiter=",")#load coef file
    #relative path of folder, contain testing images
    pathFolder = "realobj/cylinder/" #change here, cylinder or screwdriver
    X, imgNumber = loadImage(pathFolder) # X contain all loaded images
    
    i = 0 #counting var
    for img in X:
        edges = getCannyEdges(img)

        img,grasping_points = predictGraspingPoints(img,coef,threshold=.7,step=10)
        #show image with predicted grasping points
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #show canny edge extracted from image
        plt.figure()
        plt.imshow(edges,'gray')
        i+= 1
        print i #order of testing image
    plt.show()