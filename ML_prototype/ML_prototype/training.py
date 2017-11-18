import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import time
import cPickle

if __name__=="__main__":
    object_class = "thickpencil"
    object_class2 = "eraser"
    object_class3 = "martini"
    #import test datum
    test_X = np.zeros((1, 60)) #need to take suitable dimension wrt shape context
    test_y = np.ones(1) * (-1)
    for i in [7,8,9]:
    # for i in range(10):
        data = np.loadtxt('%s/50-300/%s-%d.txt' % (object_class, object_class, i), delimiter=",")
        test_X = np.concatenate((test_X, data[:, :-1]))
        test_y = np.concatenate((test_y, data[:, -1]))

    #import training datum
    X = np.zeros((1,60))
    y = np.ones(1)*(-1)
    for i in range(10):
        data = np.loadtxt('%s/50-300/%s-%d.txt' % (object_class, object_class,i), delimiter=",")
        X = np.concatenate((X,data[:,:-1]))
        y = np.concatenate((y,data[:,-1]))
    for i in range(10):
        data = np.loadtxt('%s/50-300/%s-%d.txt' % (object_class2, object_class2,i), delimiter=",")
        X = np.concatenate((X,data[:,:-1]))
        y = np.concatenate((y,data[:,-1]))
    for i in range(10):
        data = np.loadtxt('%s/50-300/%s-%d.txt' % (object_class3, object_class3,i), delimiter=",")
        X = np.concatenate((X,data[:,:-1]))
        y = np.concatenate((y,data[:,-1]))

    # balance training data
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

    #-------fit data-----------
    start_time = time.time()
    print "Start fitting"

    #try different model
    # model = svm.SVC()
    #model = svm.libsvm.fit(X,y,kernel='rbf')
    model = LogisticRegression()
    model.fit(X,y)
    print "Time: ",time.time()-start_time
    print model.coef_
    # print model.coef_.shape
    # print logreg.predict_proba(X[0,:])

    #---------save coef file-------------
    coeffile_name = "coef.txt"
    coeffile = open('coef/%s' %coeffile_name, 'w')
    out = [str(d) for d in model.coef_[0]]
    coeffile.write(','.join(out) + '\n')
    coeffile.close()

    #save classifier
    # save the classifier
    with open('logerg.pkl', 'wb') as fid:
        cPickle.dump(model, fid)


    #------------test coef--------------
        # load it again
    # with open('logerg.pkl', 'rb') as fid:
    #     model = cPickle.load(fid)
    pos = 0
    neg = 0
    for label in test_y:
        if label == -1:
            neg+=1
        else:
            pos+=1
    print "pos% in test",pos/float(pos+neg)
    print "neg% in test",neg/float(pos+neg)

    predict = model.predict(test_X)
    correct = 0
    correct_pos = 0
    correct_neg = 0
    for p,y_label in zip(predict,test_y):
        if p == y_label:
            correct += 1
            if y_label == 1:
                correct_pos += 1
            else:
                correct_neg += 1

    print "total correct% in test",correct/float(len(test_y))
    print "pos correct% in test", correct_pos / float(pos)
    print "neg correct% in test", correct_neg / float(neg)
