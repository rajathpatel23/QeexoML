# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:33:55 2019

@author: rpsworker
"""
from utility import load_instances, load_labels, load_timestamps, convert_to_classlabels, write_results
import tensorflow as tf
from sklearn.externals import joblib
from keras.models import load_model
import numpy as np
from deep_cwt_learner import convert_cwt, generate_features, get_tuning
import os


def test_model(*args):
    """
        param X_test: a feature matrix
        param model: 
        return : predicted labels for test data to the file saved with model 
        name
    """
    data_location = args[0]
    model_location = args[1]
    test_instances = load_instances(data_location)
    X_test = generate_features(test_instances, feature='accel')
    X_test = convert_cwt(X_test)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    timestamps = load_timestamps(test_instances)
    for file in os.listdir(model_location):
        model = load_model(model_location + file)
        y_test = model.predict(X_test)
        y_pred = get_tuning(y_test)
        file_name = file.strip()
        classlabels = convert_to_classlabels(y_pred)
        write_results(timestamps, classlabels, file_name + ".csv")


if __name__ == '__main__':
    model_loc = '../model_saver'
    data_loc = '../data/test/'
    test_model(data_loc, model_loc)
