#!/usr/bin/env python

import os
import pandas as pd

from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/source/')


import source.classification  as cls

'''
The following function will be called to train and test your model.
The function name, signature and output type is fixed.
The first argument is file name that contain data for training.
The second argument is file name that contain data for test.
The function must return predicted values or emotion for each data in test dataset
sequentially in a list.
['sad', 'happy', 'fear', 'fear', ... , 'happy']
'''

def  aithon_level2_api(traingcsv, testcsv):

    # The following dummy code for demonstration.

    # Train the model with training data
    cls.train_a_model(traingcsv)

    # Test that model with test data
    # And return predicted emotions in a list
    return cls.test_the_model(testcsv)


df = pd.read_csv('./data/aithon2020_level2_traning.csv')
traincsv, testcsv = train_test_split(df, test_size=0.1)
print(aithon_level2_api(traincsv, testcsv))