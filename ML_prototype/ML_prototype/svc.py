import time
from sklearn.svm import SVC
import numpy as np
import cPickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

if __name__=="__main__":
    #objects name
    object_class = "martini"
    object_class2 = "eraser"
    object_class3 = "thickpencil"

    #import training datum
    X = np.zeros((1,60))
    y = np.ones(1)*(-1)
    for i in range(10):
        data = np.loadtxt('%s/50-300//%s-%d.txt' % (object_class, object_class,i), delimiter=",")
        X = np.concatenate((X,data[:,:-1]))
        y = np.concatenate((y,data[:,-1]))
    print object_class
    for i in range(10):
        data = np.loadtxt('%s/50-300//%s-%d.txt' % (object_class2, object_class2,i), delimiter=",")
        X = np.concatenate((X,data[:,:-1]))
        y = np.concatenate((y,data[:,-1]))
    print object_class2
    for i in range(10):
        data = np.loadtxt('%s/50-300//%s-%d.txt' % (object_class3, object_class3,i), delimiter=",")
        X = np.concatenate((X,data[:,:-1]))
        y = np.concatenate((y,data[:,-1]))
    print object_class3

    #balance training data
    pos_data = [] #create new empty list
    pos_label = []
    neg_data = []
    neg_label = []

    for x, l in zip(X, y): #separate positive and negative data
        if l == 1:
            pos_data.append(x)
            pos_label.append(l)
        else:
            neg_data.append(x)
            neg_label.append(l)

    new_X = []
    new_y = []
    if len(pos_data) < len(neg_data): #checking condition, whether number of positive (n) < number of negative
    	neg_index = np.random.randint(0,len(neg_data),len(pos_data)) #take (n) random index from negative data
    	neg_random_X = []
    	neg_random_y = []
    	for index in neg_index:
    		neg_random_X.append(neg_data[index]) #append (n) negative data to new set
    		neg_random_y.append(neg_label[index])
        new_X = pos_data + neg_random_X #merge positive and negative data
        new_y = pos_label + neg_random_y 
    new_X = np.array(new_X)
    new_y = np.array(new_y)

    X = new_X #assign data set to balanced set
    y = new_y

    print set(y.tolist())
    print X.shape
    print y.shape

    #-------grid search-----------
    start_time = time.time()

    print "Start svc"
    clf = SVC(C = 10, kernel = 'rbf', gamma = 0.01) #using sklearn
    clf.fit(X,y)
    score = clf.score(X, y) #evaluate score

    print score
    #save classifier
    # save the classifier to file
    with open('logerg_4x5_50_300_pencil.pkl', 'wb') as fid:
    	cPickle.dump(clf, fid)

    # with open('logerg.pkl', 'rb') as fid:
    # 	model = cPickle.load(fid)

    print "Time: ",time.time()-start_time