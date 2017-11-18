from extractDatum import getEdges2,predictGraspingPoints,getContour2
import numpy as np
import cv2
from matplotlib import pyplot as plt

def getCannyEdges(img):
    edges = cv2.Canny(img, 200, 300)
    return edges

if __name__ == "__main__":
    object_class = "thickpencil"
    coef = np.loadtxt('%s/coef/coef.txt'%(object_class),delimiter=",")#load coef file

    if 0:#test on synthetic img
        id = "0060"#id of img
        path = "%s/img/%s%s.png"%(object_class,object_class,id)
        path2 = "%s/graspingPoint/graspPriorityWidth_%s%s.png"%(object_class,object_class,id)
    # print path2
    if 1:#test on real object
        path = "realobj/screwdriver.jpg"
    img = cv2.imread(path,3)

    #resize object, this may help when object is small in image
    # box = cv2.boundingRect(getEdges2(path))
    # # cv2.rectangle(img, (box[0] - 20, box[1] - 20), (box[0] + box[2] + 20, box[1] + box[3] + 20), (0, 255, 0), 1)
    # obj_patch = img[box[1] - 20:box[1] + box[3] + 20,box[0] - 20:box[0] + box[2] + 20]
    # shape = obj_patch.shape
    # obj_patch = cv2.resize(obj_patch,(shape[1]*2,shape[0]*2))

    edges = getCannyEdges(img)
    # true_grasping_region,_ = getContour2(path2)
    # edges = getEdges2(path)

    img,grasping_points = predictGraspingPoints(img,coef,threshold=.5,step=10)

    #display by cv2
    # cv2.drawContours(img,[true_grasping_region],0,(0,0,255),1)
    # cv2.imshow("predict grasping point",img)
    # # cv2.imshow("edges patch",edges)
    # cv2.imshow("edges",edges)
    # if cv2.waitKey(0)>0:
    #     cv2.destroyAllWindows()

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(edges,'gray')
    plt.show()