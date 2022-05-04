import numpy as np
import sys
from matplotlib import pyplot as plt
import logisticregr as logreg
import svm
import NN

X = np.arange(5)

test_overall_accuracy = []
liberal_accuracy = []
conservative_accuracy = []
time_taken = []
recall = []
precision = []
f1_score = []

#test_overall_accuracy = [67.42, 66.37, 65.18, 64.62, 78.06]
#liberal_accuracy = [96.14, 99.04, 99.9, 99.95, 83.80]
#conservative_accuracy = [15.18, 6.93, 2.02, 0.35, 67.63]
#time_taken = [0.8770148754119873, 147.37696313858032, 7.723725318908691, 28.315014362335205, 40.93367600440979]

labels = ["logistic", "svm - linear", "svm - rbf", "svm - poly", "NN"]

if len(sys.argv) == 2:
    if sys.argv[1] in ('tokenized_features.csv'):
        file_name = sys.argv[1]
    else:
        raise Exception('Unsupported preprocessing file: ' + sys.argv[1])

    print("\nRunning logistic regression...")
    (acc, acc_lib, acc_cons, time, cm) = logreg.run_classifier(file_name)
    test_overall_accuracy.append(acc)
    liberal_accuracy.append(acc_lib)
    conservative_accuracy.append(acc_cons)
    time_taken.append(time)
    rec = cm[0][0] / (cm[0][0] + cm[1][0])
    prec = cm[0][0] / (cm[0][0] + cm[0][1])
    recall.append(rec)
    precision.append(prec)
    f1_score.append(2 * ((rec * prec) / (rec + prec)))

    print("\nRunning svm - linear kernel...")
    (acc, acc_lib, acc_cons, time, cm) = svm.run_classifier('linear', file_name)
    test_overall_accuracy.append(acc)
    liberal_accuracy.append(acc_lib)
    conservative_accuracy.append(acc_cons)
    time_taken.append(time)
    rec = cm[0][0] / (cm[0][0] + cm[1][0])
    prec = cm[0][0] / (cm[0][0] + cm[0][1])
    recall.append(rec)
    precision.append(prec)
    f1_score.append(2 * ((rec * prec) / (rec + prec)))

    print("\nRunning svm - rbf kernel...")
    (acc, acc_lib, acc_cons, time, cm) = svm.run_classifier('rbf', file_name)
    test_overall_accuracy.append(acc)
    liberal_accuracy.append(acc_lib)
    conservative_accuracy.append(acc_cons)
    time_taken.append(time)
    rec = cm[0][0] / (cm[0][0] + cm[1][0])
    prec = cm[0][0] / (cm[0][0] + cm[0][1])
    recall.append(rec)
    precision.append(prec)
    f1_score.append(2 * ((rec * prec) / (rec + prec)))

    print("\nRunning svm - poly kernel...")
    (acc, acc_lib, acc_cons, time, cm) = svm.run_classifier('poly', file_name)
    test_overall_accuracy.append(acc)
    liberal_accuracy.append(acc_lib)
    conservative_accuracy.append(acc_cons)
    time_taken.append(time)
    rec = cm[0][0] / (cm[0][0] + cm[1][0])
    prec = cm[0][0] / (cm[0][0] + cm[0][1])
    recall.append(rec)
    precision.append(prec)
    f1_score.append(2 * ((rec * prec) / (rec + prec)))

    print("\nRunning NN...")
    (acc, acc_lib, acc_cons, time, cm) = NN.run_classifier(2, [64, 32], file_name)
    test_overall_accuracy.append(acc)
    liberal_accuracy.append(acc_lib)
    conservative_accuracy.append(acc_cons)
    time_taken.append(time)
    rec = cm[0][0] / (cm[0][0] + cm[1][0])
    prec = cm[0][0] / (cm[0][0] + cm[0][1])
    recall.append(rec)
    precision.append(prec)
    f1_score.append(2 * ((rec * prec) / (rec + prec)))

    print(recall)
    print(precision)
    print(f1_score)

    fig, (ax) = plt.subplots(2, 2)

    ax[0][0].title.set_text("Overall accuracy")
    ax[0][0].scatter(X, test_overall_accuracy, color=['red', 'green', 'blue', 'orange', 'black'])
    ax[0][1].title.set_text("Accuracy - Liberal")
    ax[0][1].scatter(X, liberal_accuracy, color=['red', 'green', 'blue', 'orange', 'black'])
    ax[1][0].title.set_text("Accuracy - Conservative")
    ax[1][0].scatter(X, conservative_accuracy, color=['red', 'green', 'blue', 'orange', 'black'])
    ax[1][1].title.set_text("Time taken")
    ax[1][1].scatter(X, time_taken, color=['red', 'green', 'blue', 'orange', 'black'])

    for i, txt in enumerate(labels):
        ax[0][0].annotate(txt, (X[i]+0.05, test_overall_accuracy[i]+0.05))
        ax[0][1].annotate(txt, (X[i]+0.05, liberal_accuracy[i]+0.05))
        ax[1][0].annotate(txt, (X[i]+0.05, conservative_accuracy[i]+0.05))
        ax[1][1].annotate(txt, (X[i]+0.05, time_taken[i]+0.05))

    plt.show()
else:
    raise Exception('not enough arguments to run the script.\n'
                    'the script should be provided with the preprocessed csv file with featured.'
                    '\nrun info: python results.py [prepocessed_file]')