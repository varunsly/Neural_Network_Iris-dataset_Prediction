from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

neuralNetwork = buildNetwork(2,3,1, bias=True)

dataset = SupervisedDataSet(2,1)

dataset.addSample((0,0),(0))
dataset.addSample((1,0),(1))
dataset.addSample((0,1),(1))
dataset.addSample((1,1),(0))


trainer = BackpropTrainer(neuralNetwork, dataset=dataset , learningrate=0.01, momentum=0.06)

for i in range(1,10000):
    
    error=trainer.train()
    
    if i%1000 == 0:
        print("Error in iteration ", i, "is" , error)
        print(neuralNetwork.activate([0,0]))
        print(neuralNetwork.activate([0,1]))
        print(neuralNetwork.activate([1,0]))
        print(neuralNetwork.activate([1,1]))
        
        
        
print("\n Final Error and Final Result \n")
print(error)
print(neuralNetwork.activate([0,0]))
print(neuralNetwork.activate([0,1]))
print(neuralNetwork.activate([1,0]))
print(neuralNetwork.activate([1,1]))