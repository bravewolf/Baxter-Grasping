Folder:
- realobj: contain image of real testing object
- testOnReal: contain grAsping points prediction result on real object
- thickpencil: image and training datum of thickpencil class
- eraser/martini: contain extracted descriptor extracted from data
- Python-Shape-Context: python implementation of shape context from github user "creotiv", modified to suitable with our project
- svm: paramters of SVM model
- coef: coefficient of training logistic regression model

Python file:
- extractDatum.py: implement extracting training datum, predicting grasping point
- training.py: implement logistic regression
- svm.py: implement rbf kernel SVM(SVM is too slow)
- gridSearch.py: implement grid search for SVM
- testOnImage.py: implement test on novel image object
- realObjectTest.py: implement test on real object
