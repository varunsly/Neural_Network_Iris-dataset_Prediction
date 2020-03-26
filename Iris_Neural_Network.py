from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from random import normalvariate
import numpy as np
from sklearn import datasets
import numpy.random as random

#Function to separate the data into test and training data
def split_dataset(self, proportion=0.7):
    indicies = random.permutation(len(self))
    separator = int(len(self)* proportion)

    leftIndicies = indicies[:separator]
    rightIndicies = indicies[separator:]

    leftDataset = ClassificationDataSet(inp=self['input'][leftIndicies].copy(), target=self['target'][leftIndicies].copy())
    rightDataset = ClassificationDataSet(inp=self['input'][rightIndicies].copy(), target=self['target'][rightIndicies].copy())

    return leftDataset, rightDataset
#Fecting the data from sklearn

irisData = datasets.load_iris()
dataFeatures = irisData.data
dataTarget = irisData.target

#Classfification of the data
dataset = ClassificationDataSet(4 , 1 , nb_classes=3)
#Data is being noramilzed using the numpy
for i in range(len(dataFeatures)):
    dataset.addSample(np.ravel(dataFeatures[i]), dataTarget[i])



trainingData, testData = split_dataset(dataset,0.7)

#Again is data being noramilzed foe the neuralNetworks
testData._convertToOneOfMany()
trainingData._convertToOneOfMany()

neuralNetwork =buildNetwork(trainingData.indim,7, trainingData.outdim, outclass=SoftmaxLayer)

trainer=BackpropTrainer(neuralNetwork, dataset=trainingData, learningrate=0.04, momentum=0.02, verbose=True)

trainer.trainEpochs(10)

print('Error(test dataset):' , percentError(trainer.testOnClassData(dataset=testData), testData['class']))

print('\n\n')

count=0
#Predictions are being made using the neuralNetwork
for input in dataFeatures:
    print(count, " Output of the Neural Network is: " , neuralNetwork.activate(input))
    count=count+1
