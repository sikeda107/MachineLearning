# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron


def main():
    #load_iris method load and return the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    y_names = iris.target_names
    labels = y_names[y]

    # take Petal length and Petal width as features
    setosa_petal_length = X[labels == 'setosa', 2]
    setosa_petal_width = X[labels == 'setosa', 3]
    setosa = np.c_[setosa_petal_length, setosa_petal_width]
    versicolor_petal_length = X[labels == 'versicolor', 2]
    versicolor_petal_width = X[labels == 'versicolor', 3]
    versicolor = np.c_[versicolor_petal_length, versicolor_petal_width]
    virginica_petal_length = X[labels == 'virginica', 2]
    virginica_petal_width = X[labels == 'virginica', 3]
    virginica = np.c_[virginica_petal_length, virginica_petal_width]

    #Make a scatter plot of (x,y)
    plt.scatter(setosa[:, 0], setosa[:, 1], color='red')
    plt.scatter(versicolor[:, 0], versicolor[:, 1], color='blue')
    plt.scatter(virginica[:, 0], virginica[:, 1], color='green')

    #make training data sets as teacher signals
    training_data = np.r_[setosa, versicolor, virginica]
    training_labels = np.r_[
        np.zeros(len(setosa)),#Return a new array of given shape and type, filled with zeros.
        np.ones(len(versicolor)),#Return a new array of given shape and type, filled with ones.
        np.ones(len(virginica)) * 2,
    ]

    #instantiation
    clf = Perceptron()#Single Perceptron
    clf2 = MLPClassifier()#Multi-layer Perceptron classifier
    clf3 = MLPClassifier(solver="sgd")#Multi-layer Perceptron classifier
    clf4 = svm.LinearSVC()#Linear Support Vector Classification

    #Fit the model according to the given training data
    clf.fit(training_data, training_labels)
    clf2.fit(training_data,training_labels)
    clf3.fit(training_data, training_labels)
    clf4.fit(training_data, training_labels)

    print("training_data")
    print(training_data)
    print("training_labels")
    print(training_labels)

    training_x_min = training_data[:, 0].min() - 1
    training_x_max = training_data[:, 0].max() + 1
    training_y_min = training_data[:, 1].min() - 1
    training_y_max = training_data[:, 1].max() + 1
    grid_interval = 0.02
    #meshgrid method return coordinate matrices from coordinate vectors
    #arrange method return evenly spaced values within a given interval
    xx, yy = np.meshgrid(
        np.arange(training_x_min, training_x_max, grid_interval),
        np.arange(training_y_min, training_y_max, grid_interval),
    )

    #Predict class labels for samples in X
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z3 = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
    Z4 = clf4.predict(np.c_[xx.ravel(), yy.ravel()])

    #reshape method gives a new shape to an array without changing its data.
    Z = Z.reshape(xx.shape)
    Z2 = Z2.reshape(xx.shape)
    Z3 = Z3.reshape(xx.shape)
    Z4 = Z4.reshape(xx.shape)

    #contourf method draw contour lines and filled contours, respectively
    #plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)
    #plt.contourf(xx, yy, Z2, cmap=plt.cm.bone, alpha=0.2)
    #plt.contourf(xx, yy, Z3, cmap=plt.cm.bone, alpha=0.2)
    plt.contourf(xx, yy, Z4, cmap=plt.cm.bone, alpha=0.2)

    #Display figure
    plt.autoscale()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
