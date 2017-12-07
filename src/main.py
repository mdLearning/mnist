'''
Created on Dec 7, 2017

@author: lyf
test mnist_network
'''
import mnist_loader
training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
import networkMnist
net = networkMnist.Network([784,10])
net.SGD(training_data,5,10,5.0,test_data = test_data)
