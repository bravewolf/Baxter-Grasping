import cv2
import numpy as np
from os import listdir
import re
import sys
sys.path.insert(0, 'Python-Shape-Context-master')
from SC import SC
import time
import thread
import multiprocessing
from math import exp


#apply color filter on s_edges image
#deprecated!!!
def getEdges(filename):
    img = cv2.imread(filename, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = 255 - cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))
    res = cv2.bitwise_and(img, img, mask=mask)
    obj_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(obj_gray, 0, 255, cv2.THRESH_BINARY)
    edges = cv2.findNonZero(thresh)
    return edges

#apply canny edge filter on origin image
#canny edge filter params needs tune!!!
def getEdges2(filename):
    img = cv2.imread(filename, 3)
    edges = cv2.Canny(img, 50, 300)#param
    edges = cv2.findNonZero(edges)
    return edges

#appy color filter on origin image,grasping point image
def getContour2(filename):
    img = cv2.imread(filename, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = 255 - cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,30]))
    res = cv2.bitwise_and(img, img, mask=mask)
    obj_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(obj_gray, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    # plt.imshow(thresh)
    # plt.show()
    im, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    if len(contours)>0:
        areas = [cv2.contourArea(contour) for contour in contours]
        maxIndex = areas.index(max(areas))
        # print max(areas)
        box = cv2.boundingRect(contours[maxIndex])
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # print cv2.contourArea(box)
        # print cv2.contourArea(rect)
        if box[2]*box[3]<50000:
            return cv2.approxPolyDP(contours[maxIndex],1,True),box
        else:
            return cv2.convexHull(contours[maxIndex]),box
    else:
        return None,None

#read img file name
def readImg(folder):
    mypath = folder
    onlyfiles = [f for f in listdir(mypath) if f!="Readme.txt"]
    return onlyfiles

#given img and coef
#predict grasping point pixel position
def predictGraspingPoints(img,coef,threshold=.5,n_samples=300,n_r = 5,n_theta = 4,step=10,
                          patch_size1=10 ,
                          patch_size2 = 40,
                            patch_size3 = 80,
                          shape=(480, 640, 3)):
    #
    a = SC(n_r,n_theta)
    #get edge image from canny edge filter
    edges = cv2.Canny(img, 50, 300)
    edges = cv2.findNonZero(edges)

    #get bound rect
    box = cv2.boundingRect(edges)
    box2 = cv2.minAreaRect(edges)
    boxPoints = cv2.boxPoints(box2)
    boxPoints = np.int0(boxPoints)

    # uniformly sample edge points
    edges_index = range(len(edges))
    index_samples = np.random.choice(edges_index, n_samples)

    # compute shape context for each edge point
    samples = [(edges[i, 0, 0], edges[i, 0, 1]) for i in index_samples]
    shape_context = a.compute(samples)

    #generate sampled edge image
    #used for get edge point inside certain patch
    sampled_edges = np.zeros((shape[0],shape[1]),dtype=np.uint8)
    for s in samples:
        sampled_edges[s[1], s[0]] = 255
    ret, sampled_edges = cv2.threshold(sampled_edges, 0, 255, cv2.THRESH_BINARY)

    #generate patch centers
    centers = []
    for i in range(box[2] / step):
        for j in range(box[3] / step):
            center = (box[0] + i * step, box[1] + j * step)
            # filter patch center which is outside box2(minAreaRect bound edges)
            if cv2.pointPolygonTest(boxPoints, center, False) == 1:
                centers.append(center)
    print "Number of patches:",len(centers)
    grasping_points = []
    #compute descriptor for each patch
    maxCenter = centers[0]
    maxPro = 0.0
    for i in range(len(centers)):
        center = centers[i]
        # cv2.circle(img, center, 2, (0, 0, 255), 2)

        # accumulate shape context for patch with different scale
        descriptor = np.empty(0)
        for patch_size in [patch_size1, patch_size2, patch_size3]:
            c1 = max([center[0] - patch_size / 2, 0])  # min,max to avoid patch outside image
            c2 = max([center[1] - patch_size / 2, 0])
            e1 = min([c1 + patch_size, shape[1]])
            e2 = min([c2 + patch_size, shape[0]])

            # find edge points inside patch
            inside = cv2.findNonZero(sampled_edges[c2:e2, c1:e1])
            if inside is not None:
                # print "inside"
                accumulated = np.zeros(n_r * n_theta)
                for p in inside:
                    indices = [i for i, x in enumerate(samples) if x == (c1 + p[0, 0], c2 + p[0, 1])]
                    # accumulate shape context
                    for index in indices:
                        accumulated += shape_context[index]
            else:
                accumulated = np.zeros(n_r * n_theta)
            descriptor = np.concatenate((descriptor, accumulated))#concatenate accumulated SC

        #predict grasping point
        prob = 1/(1+exp(-np.dot(descriptor,coef)))
        if prob > maxPro:
            maxCenter = center
            maxPro = prob

        if prob >threshold:#threshold for true grasping point
            cv2.circle(img, center, 2, (0, 0, 255), 2)
            grasping_points.append(center)
    cv2.circle(img, maxCenter, 2, (255, 0, 0), 2) #draw center with highest confident probability
    grasping_points.append(maxCenter)
    print "max Probability: ", maxPro
    return img,grasping_points

###extract descriptor given a set of img name
#generate data case (descriptor, label)
#save in datafile
def extractTrainingDatum(object_class,datafile_name,imgs,thread_name,flags):
    #folders path
    folder1 = "%s/img"%object_class
    folder2 = "%s/graspingPoint"%object_class
    folder3 = "%s/edges"%object_class

    #shape context params(needs tune!!!)
    n_r = 5
    n_theta = 4
    n_samples = 300
    a = SC(n_r,n_theta)

    #patch params(needs tune!!!)
    step = 10 #every 10 pixels one patch
    patch_size1 = 10 #width of three different scale
    patch_size2 = 40
    patch_size3 = 80

    #open txt file
    datafile = open('%s/data/%s'%(object_class,datafile_name), 'w')
    total_patches = 0
    num = len(imgs)
    for i in range(num):
        start_time = time.time()#record time
        name1 = imgs[i]
        id = re.search(r'(\d{4})\.png',name1).group(1)
        name2 = "graspPriorityWidth_%s%s.png"%(object_class,id)
        name3 = "%s%ss_edges.png"%(object_class,id)
        print name1,name2,name3
        shape = (480, 640, 3)
        cnt2, _ = getContour2(folder2 + "/" + name2)  ###contour of grasping point,box2 is not used


        #get edges and bounding box
        edges = getEdges2(folder1+"/"+name1)
        box = cv2.boundingRect(edges)
        box2 = cv2.minAreaRect(edges)
        boxPoints = cv2.boxPoints(box2)
        boxPoints = np.int0(boxPoints)

        #uniformly sample edge points
        edges_index = range(len(edges))
        index_samples = np.random.choice(edges_index,n_samples)

        #compute shape context for each edge points
        samples = [(edges[i,0,0],edges[i,0,1]) for i in index_samples]
        shape_context =  a.compute(samples)

        #create image of sampled edges
        # used for get edge point inside certain patch
        sampled_edges = np.zeros((shape[0],shape[1]),dtype=np.uint8)
        for s in samples:
            sampled_edges[s[1],s[0]] = 255
        ret, sampled_edges = cv2.threshold(sampled_edges, 0, 255, cv2.THRESH_BINARY)

        #generate patch center
        centers = []
        for i in range(box[2]/step):
            for j in range(box[3]/step):
                center = (box[0]+i*step,box[1]+j*step)
                # filter patch center which is outside box2(minAreaRect bound edges)
                if cv2.pointPolygonTest(boxPoints, center, False) == 1:
                    centers.append(center)
        print "[%s]Number of patches:"%thread_name, len(centers)
        total_patches += len(centers)

        graspingPoint = 0
        # compute descriptor for each patch
        for i in range(len(centers)):
            center = centers[i]

            #get label
            label = -1
            if cnt2 is not None:
                if cv2.pointPolygonTest(cnt2, center, False)>=0:
                    label = 1
            if label == 1:
                graspingPoint += 1

            #accumulate shape context for patch with scale size
            descriptor = np.empty(0)
            for patch_size in [patch_size1,patch_size2,patch_size3]:
                c1 = max([center[0]-patch_size/2,0])#min,max to avoid patch outside image
                c2 = max([center[1]-patch_size/2,0])
                e1 = min([c1+patch_size,shape[1]])
                e2 = min([c2+patch_size,shape[0]])

                #find edge points inside patch
                inside = cv2.findNonZero(sampled_edges[c2:e2,c1:e1])
                if inside is not None:
                    accumulated = np.zeros(n_r*n_theta)
                    for p in inside:
                        indices = [i for i, x in enumerate(samples) if x == (c1+p[0,0],c2+p[0,1])]
                        #accumulate shape context
                        for index in indices:
                            accumulated += shape_context[index]
                else:
                    accumulated = np.zeros(n_r * n_theta)
                descriptor = np.concatenate((descriptor,accumulated))#concatenate accumulated shape context
            descriptor = descriptor.tolist()
            descriptor.append(label)#append label
            out =[str(d) for d in descriptor]
            # print out
            datafile.write(','.join(out) + '\n')

        print "[%s]Grasping point:"%thread_name,graspingPoint
        print "[%s]Time:"%thread_name,time.time()-start_time
        print ""
    datafile.close()  # close file
    print "[%s]Finished"%thread_name
    flags.value += 1#increment flags to indicate finish

if __name__ == "__main__":
    object_class = "thickpencil"
    folder1 = "%s/img" % object_class
    imgs = readImg(folder1)
    manager = multiprocessing.Manager()
    flags = manager.Value('flags',0)#flags to indicate all processed finish

    #multi-processes to accelerate extracting training data
    n_process = 10#number of processes
    batch_size = len(imgs)/n_process
    try:
        for i in range(n_process):
            p = multiprocessing.Process(target=extractTrainingDatum,
                                        args=("thickpencil",
                                              "thickpencil-%d.txt"%i,
                                              imgs[batch_size*i:batch_size*(i+1)],
                                              "Range(%d,%d)"%(batch_size*i,batch_size*(i+1))
                                              ,flags,))
            p.start()
        # thread.start_new_thread(extractTrainingDatum, ("thickpencil","thickpencil-test.txt",imgs[0:1],"Range(0,2)",flags,0,))
        # thread.start_new_thread(extractTrainingDatum, ("thickpencil","thickpencil-test2.txt",imgs[2:3],"Range(2,4)",flags,1,))
    except:
        print "Error: unable to start process"
    # extractTrainingDatum("thickpencil","thickpencil-test.txt",0,2,"Range(0,2)")
    while flags.value<n_process:
        pass
    print "Finish"