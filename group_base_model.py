import random
import sys
import os
import pandas as pd
import numpy as np
from utility import load_instances, load_labels, load_timestamps, convert_to_classlabels, write_results
from evaluation import performance_analysis
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from collections import OrderedDict, defaultdict
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GroupKFold

random.seed(100)


def p_features(X_in, type_="MinMax"):
    """
    :param X_in: feature matrix
    :param type_: type of normalizatoin
    :return: return normalized matrix
    """
    if type_ == 'MinMax':
        X_in = (X_in - np.min(X_in)) / (np.max(X_in) - np.min(X_in))
    return X_in


def convert_cwt(X_in):
    """

    :param X_in: feature matrix
    :return: CWT transform matrix
    """
    scale = np.arange(1, 53)
    dt = 1 / 4000  # Sampling rate
    frequencies = pywt.scale2frequency('morl', scale) / dt
    X_cwt = []
    for x in X_in:
        coeff, freq = pywt.cwt(x, frequencies, 'morl')
        X_cwt.append(np.abs(coeff))
    return X_cwt


def generate_features(instances, feature=None):
    """ generate features
        param instances: a list of Instance class objects
        return: a feature matrix
    """
    # This is an example of using raw accelerometer samples as features.
    # You are not required to use this feature set.
    if feature == 'touch':
        return np.array([list(instance.touch.values()) for instance in instances])
    if feature == 'accel':
        return np.array([instance.accel.astype(float) for instance in instances])
    else:
        X1 = np.array([instance.accel.astype(float) for instance in instances])
        X2 = np.array([list(instance.touch.values()) for instance in instances])
        X = np.concatenate((X1, X2), axis=1)
        return X


def get_group(train_instance, gr_name):
    try:
        if gr_name in train_instance[0].info.keys():
            return [x.info[gr_name] for x in train_instance]
    except KeyError:
        print("please enter the correct group name")


def train_model(X, y, algorithm=None):
    """ train a model (1 nearest neighbor)
        param X: a feature matrix
        param y: a vector contains labels
        return : trained model
        cross validation model with 10 fold cross validation
    """
    score = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
    if algorithm == 'logistic_regression':
        tuned_parameters = [{'penalty': ['l1', 'l2'], 'solver': ['saga', 'liblinear'],
                             "max_iter": [100, 200, 300, 400, 600], 'C': [1.0, 1e-1, 1e-2, 10, 100]},
                            ]
        model = GridSearchCV(LogisticRegression(), param_grid=tuned_parameters,
                             cv=cv, scoring=score, refit='accuracy', n_jobs=5)
        model.fit(X, y)

    elif algorithm == 'random_forest':
        tuned_parameters = [{'n_estimators': [10, 50, 100],
                             'criterion': ['gini', 'entropy'],
                             'min_samples_split': [2, 5, 10],
                             'max_features': ['auto', 'sqrt'],
                             'max_depth': [10, 20, 30, 50]}]
        model = GridSearchCV(RandomForestClassifier(), param_grid=tuned_parameters,
                             cv=cv, scoring=score, refit='accuracy', n_jobs=5)
    else:
        raise Exception("Enter the correct algorithm name")
    model.fit(X, y)
    return model


def test_on_heldout(X_test, y_test, model, eval_dict):
    """
    :param X_test: Held out test data
    :param y_test: Held out true labels
    :param model: trained model
    :param eval_dict: evaluation dict
    :return:
    """
    y_pred_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    eval_dict = performance_analysis(y_pred_prob, y_test, y_pred, eval_dict)

    return eval_dict


def main(*args):
    # prepare training data
    train_instances = load_instances(args[0])
    model_save_location = args[1]
    train_algo = args[2]
    group_name = args[3]
    eval_dict = defaultdict(list)
    X = generate_features(instances=train_instances, feature='touch')
    y = load_labels(train_instances)
    if group_name == 'surface':
        group = get_group(train_instances, group_name)
        group = np.array(group)
        train_indexes = np.where(group == 'table')
        test_indexes = np.where(group == 'hand')
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]
    elif group_name == 'general':
        X_train, X_test, y_train, y_test, = train_test_split(X, y, group, test_size=0.25, random_state=2019)
    else:
        group = get_group(train_instances, group_name)
        test_users = random.sample(group, k=5)
        bool_list = np.array([True if x in test_users else False for x in group])
        train_indexes = np.where(bool_list == False)
        test_indexes = np.where(bool_list == True)
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

    if train_algo == 'logistic_regression':
        model = train_model(X_train, y_train, algorithm='logistic_regression')
        eval_dict = test_on_heldout(X_test, y_test, model, eval_dict=eval_dict)
        joblib.dump(model, model_save_location + 'Logistic_Regression_' + group_name + '.pkl')
        joblib.dump(eval_dict, model_save_location + 'eval_dict_lr_' + group_name + '_.pkl')

    elif train_algo == 'random_forest':
        model = train_model(X_train, y_train, algorithm='random_forest')
        eval_dict = test_on_heldout(X_test, y_test, model, eval_dict=eval_dict)
        joblib.dump(model, model_save_location + 'Random_Forest_' + group_name + '.pkl')
        joblib.dump(eval_dict, model_save_location + 'eval_dict_rf_' + group_name + '.pkl')

    else:
        raise Exception("please enter the correct classifier names")


if __name__ == '__main__':
    train_data = '../data/train'
    model_location = '../model_saver/'
    algo = sys.argv[1]
    grp_name = sys.argv[2]
    main(train_data, model_location, algo, grp_name)
