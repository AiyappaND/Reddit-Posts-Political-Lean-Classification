import numpy as np
import sys
from matplotlib import pyplot as plt
import logisticregr as logreg
import svm
import NN
import seaborn as sns

X = np.arange(5)

test_overall_accuracy = []
liberal_accuracy = []
conservative_accuracy = []
time_taken = []
recall_lib = []
precision_lib = []
f1_score_lib = []
recall_cons = []
precision_cons = []
f1_score_cons = []

def record_results(accuracy, a_lib, a_cons, time, conf_m):
    test_overall_accuracy.append(accuracy)
    liberal_accuracy.append(a_lib)
    conservative_accuracy.append(a_cons)
    time_taken.append(time)
    rec_lib = conf_m[0][0] / (conf_m[0][0] + conf_m[1][0])
    prec_lib = conf_m[0][0] / (conf_m[0][0] + conf_m[0][1])
    rec_cons = conf_m[1][1] / (conf_m[0][1] + conf_m[1][1])
    prec_cons = conf_m[1][1] / (conf_m[1][1] + conf_m[1][0])
    recall_lib.append(rec_lib)
    precision_lib.append(prec_lib)
    f1_score_lib.append(2 * ((rec_lib * prec_lib) / (rec_lib + prec_lib)))
    recall_cons.append(rec_cons)
    precision_cons.append(prec_cons)
    f1_score_cons.append(2 * ((rec_cons * prec_cons) / (rec_cons + prec_cons)))


    ax = sns.heatmap(conf_m/np.sum(conf_m), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Liberal','Conservative'])
    ax.yaxis.set_ticklabels(['Liberal','Conservative'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


#test_overall_accuracy = [67.42, 66.37, 65.18, 64.62, 78.06]
#liberal_accuracy = [96.14, 99.04, 99.9, 99.95, 83.80]
#conservative_accuracy = [15.18, 6.93, 2.02, 0.35, 67.63]
#time_taken = [0.8770148754119873, 147.37696313858032, 7.723725318908691, 28.315014362335205, 40.93367600440979]

labels = ["logistic", "svm - linear", "svm - rbf", "svm - poly", "NN"]

if len(sys.argv) == 2:
    if sys.argv[1] in ('tokenized_features.csv', 'w2vecscale.csv'):
        file_name = sys.argv[1]
    else:
        raise Exception('Unsupported preprocessing file: ' + sys.argv[1])

    print("\nRunning logistic regression...")
    (acc, acc_lib, acc_cons, time, cm) = logreg.run_classifier(file_name)
    record_results(acc, acc_lib, acc_cons, time, cm)

    print("\nRunning svm - linear kernel...")
    (acc, acc_lib, acc_cons, time, cm) = svm.run_classifier('linear', file_name)
    record_results(acc, acc_lib, acc_cons, time, cm)

    print("\nRunning svm - rbf kernel...")
    (acc, acc_lib, acc_cons, time, cm) = svm.run_classifier('rbf', file_name)
    record_results(acc, acc_lib, acc_cons, time, cm)

    print("\nRunning svm - poly kernel...")
    (acc, acc_lib, acc_cons, time, cm) = svm.run_classifier('poly', file_name)
    record_results(acc, acc_lib, acc_cons, time, cm)

    print("\nRunning NN...")
    if file_name == 'w2vecscale.csv':
        hidden_l = 4
        nodes_hidden = [128, 128]
    elif file_name == 'tokenized_features.csv':
        hidden_l = 4
        nodes_hidden = [64, 32]
    else:
        raise Exception('configure NN for the feature file.')
    (acc, acc_lib, acc_cons, time, cm) = NN.run_classifier(hidden_l, nodes_hidden, file_name)
    record_results(acc, acc_lib, acc_cons, time, cm)

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

    fig, (ax) = plt.subplots(1, 2, sharey=True)

    ax[0].title.set_text("Precision - Liberal")
    ax[0].scatter(X, precision_lib, color=['red', 'green', 'blue', 'orange', 'black'])

    ax[1].title.set_text("Precision - Conservative")
    ax[1].scatter(X, precision_cons, color=['red', 'green', 'blue', 'orange', 'black'])

    for i, txt in enumerate(labels):
        ax[0].annotate(txt, (X[i]+0.05, precision_lib[i]+0.01))
        ax[1].annotate(txt, (X[i]+0.05, precision_cons[i]+0.01))

    plt.show()

    fig, (ax) = plt.subplots(1, 2, sharey=True)

    ax[0].title.set_text("Recall - Liberal")
    ax[0].scatter(X, recall_lib, color=['red', 'green', 'blue', 'orange', 'black'])

    ax[1].title.set_text("Recall - Conservative")
    ax[1].scatter(X, recall_cons, color=['red', 'green', 'blue', 'orange', 'black'])

    for i, txt in enumerate(labels):
        ax[0].annotate(txt, (X[i]+0.05, recall_lib[i]+0.01))
        ax[1].annotate(txt, (X[i]+0.05, recall_cons[i]+0.01))

    plt.show()

    fig, (ax) = plt.subplots(1, 2, sharey=True)

    ax[0].title.set_text("F1 score - Liberal")
    ax[0].scatter(X, f1_score_lib, color=['red', 'green', 'blue', 'orange', 'black'])

    ax[1].title.set_text("F1 score - Conservative")
    ax[1].scatter(X, f1_score_cons, color=['red', 'green', 'blue', 'orange', 'black'])

    for i, txt in enumerate(labels):
        ax[0].annotate(txt, (X[i]+0.05, f1_score_lib[i]+0.01))
        ax[1].annotate(txt, (X[i]+0.05, f1_score_cons[i]+0.01))

    print(test_overall_accuracy)
    print(liberal_accuracy)
    print(conservative_accuracy)
    print(recall_lib)
    print(precision_lib)
    print(f1_score_lib)
    print(recall_cons)
    print(precision_cons)
    print(f1_score_cons)
    plt.show()
else:
    raise Exception('not enough arguments to run the script.\n'
                    'the script should be provided with the preprocessed csv file with featured.'
                    '\nrun info: python results.py [tokenized_features.csv/w2vecscale.csv]')