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

print("\nRunning logistic regression...")
(acc, acc_lib, acc_cons, time) = logreg.run_classifier()
test_overall_accuracy.append(acc)
liberal_accuracy.append(acc_lib)
conservative_accuracy.append(acc_cons)
time_taken.append(time)

print("\nRunning svm - linear kernel...")
(acc, acc_lib, acc_cons, time) = svm.run_classifier('linear')
test_overall_accuracy.append(acc)
liberal_accuracy.append(acc_lib)
conservative_accuracy.append(acc_cons)
time_taken.append(time)

print("\nRunning svm - rbf kernel...")
(acc, acc_lib, acc_cons, time) = svm.run_classifier('rbf')
test_overall_accuracy.append(acc)
liberal_accuracy.append(acc_lib)
conservative_accuracy.append(acc_cons)
time_taken.append(time)

print("\nRunning svm - poly kernel...")
(acc, acc_lib, acc_cons, time) = svm.run_classifier('poly')
test_overall_accuracy.append(acc)
liberal_accuracy.append(acc_lib)
conservative_accuracy.append(acc_cons)
time_taken.append(time)

print("\nRunning NN...")
(acc, acc_lib, acc_cons, time) = NN.run_classifier(2, [64, 32])
test_overall_accuracy.append(acc)
liberal_accuracy.append(acc_lib)
conservative_accuracy.append(acc_cons)
time_taken.append(time)

#test_overall_accuracy = [67.42, 66.37, 65.18, 64.62, 78.06]
#liberal_accuracy = [96.14, 99.04, 99.9, 99.95, 83.80]
#conservative_accuracy = [15.18, 6.93, 2.02, 0.35, 67.63]
#time_taken = [0.8770148754119873, 147.37696313858032, 7.723725318908691, 28.315014362335205, 40.93367600440979]

labels = ["logistic", "svm - linear", "svm - rbf", "svm - poly", "NN"]

if len(sys.argv) == 2:
    if sys.argv[1] == "overall":
        plt.scatter(X, test_overall_accuracy, color=['red', 'green', 'blue', 'orange', 'black'])
        for i, txt in enumerate(labels):
            plt.annotate(txt, (X[i]+0.05, test_overall_accuracy[i]+0.05))
    elif sys.argv[1] == "liberal":
        plt.scatter(X, liberal_accuracy, color=['red', 'green', 'blue', 'orange', 'black'])
        for i, txt in enumerate(labels):
            plt.annotate(txt, (X[i]+0.05, liberal_accuracy[i]+0.05))
    elif sys.argv[1] == "conservative":
        plt.scatter(X, conservative_accuracy, color=['red', 'green', 'blue', 'orange', 'black'])
        for i, txt in enumerate(labels):
            plt.annotate(txt, (X[i]+0.05, conservative_accuracy[i]+0.05))
    elif sys.argv[1] == "timing":
        plt.scatter(X, time_taken, color=['red', 'green', 'blue', 'orange', 'black'])
        for i, txt in enumerate(labels):
            plt.annotate(txt, (X[i]+0.05, time_taken[i]+0.05))
    else:
        raise Exception('Unsupported Option: ' + sys.argv[1])

    plt.show()
else:
    raise Exception('not enough arguments to run the script.\n'
                    'the script should be provided with the type of plot.'
                    '\nrun info: python results.py [overall/liberal/conservative/timing]')