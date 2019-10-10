# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 01:14:26 2019

@author: rpsworker
"""

from keras.layers import Dense, Input
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D, SeparableConv2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dropout, DepthwiseConv2D, GlobalAveragePooling2D, Activation
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.models import load_model
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from utility import load_instances, load_labels, load_timestamps, convert_to_classlabels, write_results
import pywt
import random
from collections import defaultdict
from evaluation import performance_analysis

random.seed(100)


class CNN_model(object):
    def __init__(self, input_data):
        self.input_data = input_data
        self.model_name = model_name
        self.loss_name = "binary_crossentropy"
        self.optimizer_name = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=False)
        self.model = None
        self.config = tf.ConfigProto(device_count={'GPU': 1})
        self.config.gpu_options.allocator_type = 'BFC'
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        tf.keras.backend.set_session(self.sess)
        self.activity_regularizer_val = 0.01

    def build_CNN_Model_4(self):
        input_img = Input(shape=(self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3]))
        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(input_img)
        x = BatchNormalization()(x)

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = SeparableConv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(4, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(4, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Conv2D(2, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        x = Flatten()(x)
        x = Dense(1, activation="sigmoid", activity_regularizer=l2(self.activity_regularizer_val))(x)
        self.model = Model(input_img, x)
        print(self.model.summary())
        self.model.compile(loss=self.loss_name, optimizer=self.optimizer_name, metrics=['accuracy'])
        return self.model

def generate_features(instances, feature=None):
    """ generate features
        param instances: a list of Instance class objects
        return: a feature matrix
    """
    if feature == 'touch':
        return np.array([list(instance.touch.values()) for instance in instances])
    if feature == 'accel':
        return np.array([instance.accel.astype(float) for instance in instances])
    else:
        X1 = np.array([instance.accel.astype(float) for instance in instances])
        X2 = np.array([list(instance.touch.values()) for instance in instances])
        X = np.concatenate((X1, X2), axis=1)
        return X


def get_tuning(X_in):
    tuned_out = []
    for x in range(len(X_in)):
        if X_in[x][0] >= 0.5:
            tuned_out.append(1)
        else:
            tuned_out.append(0)
    return tuned_out


def get_reshape(X):
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X


def p_features(X_in, type_="MinMax"):
    if type_ == 'MinMax':
        X_in = (X_in - np.min(X_in)) / (np.max(X_in) - np.min(X_in))
    return X_in


def convert_cwt(X_in):
    scale = np.arange(1, 53)
    dt = 1 / 4000  # Sampling rate
    frequencies = pywt.scale2frequency('morl', scale) / dt
    X_cwt = []
    for x in X_in:
        coeff, freq = pywt.cwt(x, frequencies, 'morl')
        X_cwt.append(np.abs(coeff))
    return X_cwt


def train_model(X, y, X_val, y_val, gr_name):
    """ train a model (1 nearest neighbor)
        param X: a feature matrix
        param y: a vector contains labels
        return : trained model
    """
    train = CNN_model(X)
    model_1 = train.build_CNN_Model_4()
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None,
                               restore_best_weights=True),
                 ModelCheckpoint(filepath="../model_saver/CNN_model_" + gr_name + ".h5", monitor='loss',
                                 save_best_only=True)]
    model_1.fit(X, y, epochs=75, batch_size=64,
                shuffle=True, verbose=1,
                callbacks=callbacks,
                validation_data=(X_val, y_val))
    return model_1


def get_group(train_instance, gr_name):
    try:
        if gr_name in train_instance[0].info.keys():
            return [x.info[gr_name] for x in train_instance]
    except KeyError:
        print("please enter the correct group name")


def test_on_heldout(X_test, y_test, model, eval_dict):
    """
    :param X_test: Held out test data
    :param y_test: Held out true labels
    :param model: trained model
    :param eval_dict: evaluation dict
    :return:
    """
    y_ = model.predict(X_test)
    tuned_test = get_tuning(y_)
    eval_dict = performance_analysis(y_, y_test, tuned_test, eval_dict)

    return eval_dict


def main(*args):
    DATA_LOCATION = args[0]
    model_save_location = args[1]
    group_name = args[2]
    train_instances = load_instances(DATA_LOCATION)
    X = generate_features(train_instances, feature="accel")
    y = load_labels(train_instances)
    # convert to CWT
    X = convert_cwt(X)
    X = get_reshape(X)
    eval_dict = defaultdict(list)
    if group_name == 'surface':
        group = get_group(train_instances, group_name)
        train_indexes = np.where(group == 'table')
        test_indexes = np.where(group == 'hand')
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]
    elif group_name == 'general':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
    else:
        group = get_group(train_instances, group_name)
        test_users = random.sample(group, k=5)
        bool_list = np.array([True if x in test_users else False for x in group])
        train_indexes = np.where(bool_list == False)
        test_indexes = np.where(bool_list == True)
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]
    # training the model
    model_1 = train_model(X_train, y_train, X_test, y_test, group_name)
    # evaluating the model
    eval_dict = test_on_heldout(X_test, y_test, model_1, eval_dict)
    # saving the  eval data to dictionary pickle
    joblib.dump(eval_dict, model_save_location + 'eval_dict_rf_' + group_name + '.pkl')


if __name__ == "__main__":
    train_data = '../data/train'
    model_location = '../model_saver/'
    algo = sys.argv[1]
    grp_name = sys.argv[2]
    main(train_data, model_location, algo, grp_name)