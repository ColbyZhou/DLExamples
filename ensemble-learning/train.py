#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
This module provide service.

Author: 
Date:   2018/12/29 13:28:48
"""
import sys
import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from matplotlib import pyplot as plt
from IPython.display import Image

import pydotplus

import pandas as pd
import numpy as np

def train_gbm(train_X, train_Y):

    gbm = GradientBoostingClassifier(
            loss = "deviance",
            learning_rate = 0.1,
            n_estimators = 20,
            subsample = 0.8,
            max_depth = 10,
            min_samples_split = 600,
            min_samples_leaf = 100,
            min_weight_fraction_leaf = 0., # must [0, 0.5]
            max_features = None, # sqrt, log2, auto
            max_leaf_nodes = None,
            random_state = None,
            min_impurity_split = None,
            min_impurity_decrease=0.,
            presort = True,
            )

    gbm.fit(train_X, train_Y)

    return gbm

def tunning_gbm(train_X, train_Y):

    param_set = {"n_estimators": range(10, 101, 10), "max_depth": range(3, 14, 1), \
            "min_samples_split": range(10, 1000, 50), "min_samples_leaf": range(10, 300, 20)}

    gsearch = GridSearchCV(
            estimator = GradientBoostingClassifier(
                learning_rate = 0.1,
                #n_estimators = 20,
                #subsample = 0.8,
                #max_depth = 3,
                #min_samples_split = 610,
                #min_samples_leaf = 30,

                min_weight_fraction_leaf = 0.,
                max_features = None,
                max_leaf_nodes = None,
                #random_state = None,
                random_state = 10,
                min_impurity_split = None,
                min_impurity_decrease=0.,
                presort = True,
                ),
            param_grid = param_set,
            scoring = 'roc_auc',
            iid = False,
            cv = 5
            )
    gsearch.fit(train_X, train_Y)

    print gsearch.grid_scores_
    print gsearch.best_params_
    print gsearch.best_score_

    return gbm

def save_model(gbm, model_path):

    with open(model_path, 'w') as file:
        pickle.dump(gbm, file)

def load_model(model_path):
    with open(model_path, 'r') as file:
        gbm = pickle.load(file)

    return gbm

def read_data(path, fea_conf_path):

    data = pd.read_csv(path)
    target_name = 'label'

    fea_name_list = []
    with open(fea_conf_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line[0] == '#':
                continue
            fea_name_list.append(line)

    X = data[fea_name_list]
    Y = data[target_name]

    return X, Y, fea_name_list

def data_predict(gbm, X, Y, img_path):

    Y_pred = gbm.predict(X)
    Y_pred_prob = gbm.predict_proba(X)[:, 1]

    accuracy = metrics.accuracy_score(Y, Y_pred)
    auc = metrics.roc_auc_score(Y, Y_pred_prob)
    fpr, tpr, thresholds = metrics.roc_curve(Y, Y_pred_prob, pos_label = 1)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.savefig(img_path)

    return fpr, tpr, auc, accuracy

def plot_gbm(gbm, model_name, train_X, train_Y, test_X, test_Y, train_fea_name_list):

    img_dir = './data/' + model_name + '/'

    train_img_path = img_dir + 'roc_train_' + model_name + '.png'
    test_img_path = img_dir + 'roc_test_' + model_name + '.png'
    all_img_path = img_dir + 'roc_all_' + model_name + '.png'
    importance_img_path = img_dir + 'importance_' + model_name + '.png'

    train_fpr, train_tpr, train_auc, train_accuracy= data_predict(gbm, train_X, train_Y, train_img_path)
    test_fpr, test_tpr, test_auc, test_accuracy = data_predict(gbm, test_X, test_Y, test_img_path)

    print 'train data auc %.4f, test data auc %.4f' % (train_auc, test_auc)
    print 'train data accuracy %.4f, test data accuracy %.4f' % (train_accuracy, test_accuracy)
    print '**************** feature importance *****************'
    imp_items =  zip(train_fea_name_list, gbm.feature_importances_)
    sorted_imp_items = sorted(imp_items, key = lambda x:x[1], reverse = True)
    for name, imp in sorted_imp_items:
        print '%s: %.4f' % (name, imp)

    # *********** plot auc **************
    plt.figure()
    plt.plot(train_fpr, train_tpr, label = 'train_auc ' + \
            "%.2f" % (train_auc) + ', acc: ' + "%.2f" % (train_accuracy))
    plt.plot(test_fpr, test_tpr, label = 'test_auc ' + \
            "%.2f" % (test_auc) + ', acc: ' + "%.2f" % (test_accuracy))
    plt.legend()
    plt.savefig(all_img_path)

    # *************** plot importance ***************
    feature_importance = gbm.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure()
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(train_fea_name_list)[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

    plt.legend()
    plt.savefig(importance_img_path)

def main():

    if len(sys.argv) < 6:
        print 'usage: python %s train_path test_path fea_conf_path is_infer model_name' % (sys.argv[0])
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    fea_conf_path = sys.argv[3]
    is_infer = int(sys.argv[4])
    model_name = sys.argv[5]

    model_path = './data/' + model_name + '/model.pkl'

    train_X, train_Y, train_fea_name_list = read_data(train_path, fea_conf_path)
    print 'train_fea_name_list: ' + str(train_fea_name_list)
    test_X, test_Y, test_fea_name_list = read_data(test_path, fea_conf_path)
    print 'test_fea_name_list: ' + str(test_fea_name_list)

    if is_infer == 0:
        gbm = train_gbm(train_X, train_Y)
        save_model(gbm, model_path)
    else:
        gbm = load_model(model_path)

    plot_gbm(gbm, model_name, train_X, train_Y, test_X, test_Y, train_fea_name_list)

if __name__ == '__main__':
    main()

