import numpy as np
from matplotlib import pyplot as plt
import cv2

def generatePloygon(n):
    return np.random.randint(100,500,(n,2))

def plotPolygon(points,block):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], color="red")
    ax.set_title("n=%d"%len(points))
    plt.show(block=block)

def randomInit(K,l,hi):
    init = []
    for i in range(K):
        init.append(np.random.randint(hi,size=l))
    return init

def knn_oneIter(centers,K,X,k):
    clusters = [[] for i in range(K)]
    for x in X:
        dist = [distance(x,center) for center in centers]
        clusterIndex = dist.index(min(dist))
        clusters[clusterIndex].append(x)
    #compute new centers
    newCenters = []
    error_list = []
    for cluster in clusters:
        #print "numOfCluster",len(cluster)
        if len(cluster)>2:
            temp = np.array(cluster)
            newCenter = np.mean(temp,0)

            # plt.imshow(newCenter.reshape((243, 160, 4)))
            # plt.show()
            newCenters.append(newCenter)
            error = 0
            for data in cluster:
                error += distance(newCenter,data)
            error_list.append(error)
        else:
            newCenters.append(X[np.random.randint(0,high=len(X))])
            error_list.append(float("inf"))
            # print "discard center"
    #compute error

    return newCenters,error_list,clusters

def distance(v1,v2):
    dist = np.dot(v1-v2,(v1-v2).T)
    return dist


def knn(init,X,K,k,it):
    old_centers = list(init)
    clusters= []
    for i in range(it):
        centers,error_list,clusters = knn_oneIter(old_centers,K,X,k)
        old_centers = list(centers)
        # print "error ",error_list
        # for cluster in clusters:
        #     plt.imshow(np.reshape(cluster[0],(243, 160, 4)))
        #     plt.show()
    return old_centers,clusters



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while(False):
        _, frame = cap.read()
        cv2.imshow("cam",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    old_center = []
    while (True):
        # Capture frame-by-frame
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([100, 100, 100])  # blue range
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        obj_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret2, thresh = cv2.threshold(obj_gray, 60, 255, cv2.THRESH_BINARY)
        width, height, channel = frame.shape
        # print 'size',width,height
        cv2.drawMarker(frame, (height / 2, width / 2), (0, 255, 0), cv2.MARKER_CROSS, 50)
        im, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        # cnt = contours[0]
        maxAreas = 500  # threshold for minimum size of object
        maxCnt = None
        for cnt in contours:
            if cv2.contourArea(cnt) > maxAreas:
                maxAreas = cv2.contourArea(cnt)
                maxCnt = cnt
                # cnt = contours[0]

        # hull = cv2.convexHull(cnt)
        # epsilon = 0.1*cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        if maxCnt != None:

            # n=20 #number of points
            points = cv2.approxPolyDP(maxCnt,1,True)
            # points = maxCnt
            print "num of points",len(points)

            # print points
            # image = frame
            hull = cv2.convexHull(points)
            # plotPolygon(points, False)
            # rect = cv2.minAreaRect(points)
            # for p in points:
            #     cv2.circle(image,(p[0],p[1]),3,(0,0,255),1)
            K = max([2,len(points)/12])
            K =7
            print "K ",K
            # init = randomInit(K,2,500)
            # init = np.array([[200,100],[200,200],[200,300],[200,400],[200,500]])
            if old_center == []:
                old_center = [maxCnt[np.random.randint(low=0,high=len(maxCnt))] for i in range(K)]

            # print init
            centers,clusters = knn(old_center,np.array(points),K,2,20)
            old_center = list(centers)
            # print centers
            rect = cv2.minAreaRect(points)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            # cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
            cv2.drawContours(frame, [points], 0, (0, 255, 0), 2)

            for cluster in clusters:
                if cluster != []:
                    rect = cv2.minAreaRect(np.array(cluster))
                    box = np.int0(cv2.boxPoints(rect))
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)


            # print box
            # print hull
            # for p in points:
            #     cv2.circle(image,(p[0],p[1]),3,(0,0,255),1)
            # new_centers = [center for center,cluster in zip(centers,clusters) if cluster!=[]]
            # plotPolygon(np.array(new_centers),True)
            cv2.imshow("knn",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.imshow("polygon2", out)
    cap.release()
    cv2.destroyAllWindows()

