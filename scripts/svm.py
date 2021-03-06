import sys

import numpy as np
from sklearn import svm
from sklearn import metrics
import time

from pandas import read_csv
import pathlib
import os
from sklearn.model_selection import train_test_split


def desc_data(y):
    count = {0: 0, 1: 0}
    for cls in y:
        count[cls] += 1
    for cl in count.keys():
        if cl == 0:
            print("Number of instances in class Liberal: " + str(count[cl]))
        else:
            print("Number of instances in class Conservative: " + str(count[cl]))
    return count


def run_svm(kernel, X_trn, Y_trn, X_tst, Y_tst, f_name):
    if kernel in ('rbf', 'poly'):
        model = svm.SVC(kernel=kernel, C=500)
    else:
        if f_name == 'tokenized_features.csv':
            model = svm.SVC(kernel=kernel, C=0.05)
        else:
            model = svm.SVC(kernel=kernel)

    model.fit(X_trn, np.ravel(Y_trn))

    Y_prd = model.predict(X_tst)

    print("\nDescribing training set...")
    data_count_train = desc_data(Y_trn)

    print('\nAccuracy for the training data: ' + str(
        round((model.score(X_trn, Y_trn) * 100), 2)) + ' %')

    print("\nDescribing testing set...")
    data_count_test = desc_data(Y_tst)

    acc_score = round((metrics.accuracy_score(Y_tst, Y_prd) * 100), 2)
    print('\nAccuracy for the test data: ' + str(acc_score) + ' %')

    cm = metrics.confusion_matrix(Y_tst, Y_prd)

    acc_score_liberal = round(((cm[0][0] / data_count_test[0]) * 100), 2)
    print("\nAccuracy for class liberal: " + str(acc_score_liberal) + " %")

    acc_score_cons = round(((cm[1][1] / data_count_test[1]) * 100), 2)
    print("Accuracy for class Conservative: " + str(acc_score_cons) + " %")

    return (acc_score, acc_score_liberal, acc_score_cons, cm)


def run_classifier(kernel, file_name = 'tokenized_features.csv'):
    if kernel not in ('linear', 'rbf', 'poly'):
        raise Exception('not a valid kernel.\nshould be one of (linear, rbf, poly)')
    project_root = pathlib.Path().resolve().parent
    inputF = open(os.path.join(project_root, 'data', 'generated', file_name), "r")

    df = read_csv(inputF)
    X = df.values[:, :-1]
    Y = df.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    start = time.time()
    (acc_score, acc_score_liberal, acc_score_cons, cm) = run_svm(kernel, x_train, y_train, x_test, y_test, file_name)
    end = time.time()
    print("\nTime taken: " + str(end - start))

    return (acc_score, acc_score_liberal, acc_score_cons, end-start, cm)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_classifier(sys.argv[2], file_name=sys.argv[1])
    else:
        raise Exception('not enough arguments to run the script.\n'
                        'the script should be provided with features file and kernel function to run.'
                        '\nrun info: python svm.py [tokenized_features.csv/w2vecscale.csv] [linear/rbf/poly]')
