# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

def main():
    #load_iris method load and return the iris dataset
    iris = load_iris()
    #‘data’, the data to learn
    X = iris.data
    # ‘target’, the classification labels
    y = iris.target
    #train_test_split method split arrays or matrices into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #instantiation
    clf = Perceptron()#Single Perceptron
    clf2 = MLPClassifier()#Multi-layer Perceptron classifier
    clf3 = MLPClassifier(solver="sgd")#Multi-layer Perceptron classifier
    clf4 = LinearSVC()#Linear Support Vector Classification
    #Fit the model according to the given training data
    clf.fit(X_train, y_train)
    clf2.fit(X_train,y_train)
    clf3.fit(X_train,y_train)
    clf4.fit(X_train,y_train)
    #Returns the mean accuracy on the given test data and labels
    print("X_train")
    print X_train
    print("y_train")
    print y_train
    print("X_test")
    print X_test
    print("y_test")
    print y_test
    #Returns the mean accuracy on the given test data and labels
    print ("Perceptron Identification rate:", clf.score(X_test, y_test))
    print ("MLPC       Identification rate:", clf2.score(X_test, y_test))
    print ("MLPC(sgd)  Identification rate:", clf3.score(X_test, y_test))
    print ("LinearSVC  Identification rate:", clf4.score(X_test, y_test))
if __name__ == "__main__":
    main()
