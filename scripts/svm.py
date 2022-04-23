import sys

import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics

from pandas import read_csv
import pathlib
import os
from sklearn.model_selection import train_test_split

project_root = pathlib.Path().resolve().parent
inputF = open(os.path.join(project_root, 'data', 'generated', 'tokenized_features.csv'), "r")

df = read_csv(inputF)
X = df.values[:, :-1]
Y = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

def run_classifier(kernel, X_trn, Y_trn, X_tst, Y_tst, class1, class2):
    model = svm.SVC(kernel=kernel, C=0.05)
    model.fit(X_trn, np.ravel(Y_trn))

    Y_prd = model.predict(X_tst)

    print('Accuracy for the training data: ' + str(model.score(X_trn, Y_trn) * 100) + ' %')
    print('Accuracy for the test data: ' + str(metrics.accuracy_score(Y_tst, Y_prd) * 100) + ' %')

    cm = metrics.confusion_matrix(Y_tst, Y_prd)
    print(cm)

    '''
    if kernel == 'linear':
        b = model.intercept_[0]
        w1, w2 = model.coef_[0]

        c = -b/w2
        m = -w1/w2

        xmin, xmax = -2, 2
        xd = np.array([xmin, xmax])
        yd = m*xd + c

    fig, (ax1) = plt.subplots(1, 1)
    if kernel == 'linear':
        ax1.plot(xd, yd, 'k', lw=1, ls='--')

    for i in range(0, len(X_trn)):
        if (Y_trn[i][0] == class1):
            ax1.scatter(X_trn[i][0], X_trn[i][1], color='green', marker='o')
        else:
            ax1.scatter(X_trn[i][0], X_trn[i][1], color='green', marker='o')


    for i in range(0, len(X_tst)):
        if (Y_prd[i] == class1):
            ax1.scatter(X_tst[i][0], X_tst[i][1], color='red', marker='x')
        else:
            ax1.scatter(X_tst[i][0], X_tst[i][1], color='blue', marker='x')
    plt.show()
    '''

if (len(sys.argv) == 2):
    run_classifier(sys.argv[1], x_train, y_train, x_test, y_test, 0, 1)
else:
    raise Exception('not enough arguments to run the script.\n'
                    'the script should be provided kernel function to run.'
                    '\nrun info: python svm.py [linear/rbf/poly]')