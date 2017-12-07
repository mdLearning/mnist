'''
Created on Dec 7, 2017

@author: lyf
A library to load the mnist image data. for details of the data structures that are returned
see the doc strings for 'load_Data' and 'load_data_wrapper',in practice,'load_data_wrapper'
is the function usually called by our neural network code
'''
####Libraries
#Standard library
import cPickle
import gzip
#Third-party libraries
import numpy as np

def load_data():
    """
    Return the MNIST data as a tuple containing the training data,the validation
    data,and the test data.
    
    the training data is returned as a tuple with two entries
    the first entry contains the actual training images.this is a
    numpy ndarray with 50,000 entries,each entry is ,in truth , a
    numpy ndarray with 784 values,representing the 28*28 = 784
    pixels in a single MNIST image
    
    the second entry in the training data tuple is numpy ndarray
    containing 50,000 entries, those entries are just the digit
    values(0,...9) for the corresponding images contained in the first
    entry of the tuple
    
    the 'validation data' and 'test data' are similar,except
    each contains only 10,000 images
    
    this is a nice data format,but for use in neural networks it's
    helpful to modify the format of the 'training data' a little
    that's done in the wrapper function 'load_data_wrapper', see
    below
    """
    f = gzip.open('','rb')
    training_data,validation_data,test_data = cPickle.load(f)
    f.close()
    return (training_data,validation_data,test_data)

def load_data_wrapper():
    """
    Return a tuple containing 'training_data,validation_data,test_data'
    Based on 'load_data',but the format is more convenient for use in our
    implementation of neural networks.
    
    IN particular 'training_data' is a list containing 50,000
    2-tuples(x,y).x is a 784-dimensional numpy.ndarray containing
    the input images. y is a 10-dimensional numpy.ndarray representing
    the unit vector corresponding to the correct digit for 'x'
    
    Obviously,this means we're using slightly different formats for the 
    training data and validation/test data. these formats turn out to be the
    most convenient for use in our neural network code
    """
    tr_d,va_d,te_d = load_data()
    training_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs,training_results)
    validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = zip(validation_inputs,va_d[1])
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = zip(test_inputs,te_d[1])
    return (training_data,validation_data,test_data)

def vectorized_result(j):
    """
    return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeros elsewhere. This is used to convert a digit
    (0,...9) into a corresponding desired output from the neural
    network
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e