from pandas import read_csv
import pathlib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time

def desc_data(y):
    count = {0:0, 1:0 }
    for cls in y:
        count[cls] += 1
    for cl in count.keys():
        if cl == 0:
            print("Number of instances in class Liberal: " + str(count[cl]))
        else:
            print("Number of instances in class Conservative: " + str(count[cl]))
    return count

def run_classifier(file_name = 'tokenized_features.csv'):
    project_root = pathlib.Path().resolve().parent
    inputF = open(os.path.join(project_root, 'data', 'generated', file_name), "r")

    df = read_csv(inputF)
    X = df.values[:, :-1]
    Y = df.values[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    print("\nDescribing training set...")
    desc_data(y_train)

    print("\nDescribing testing set...")
    data_count = desc_data(y_test)

    start = time.time()

    logisticRegr = LogisticRegression(max_iter=10000)
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test, y_test)

    end = time.time()

    acc = round((score * 100), 2)

    print("\nOverall accuracy: " + str(score) + " %")

    cm = metrics.confusion_matrix(y_test, predictions)

    acc_score_liberal = round(((cm[0][0] / data_count[0]) * 100), 2)
    print("\nAccuracy for class liberal: " + str(acc_score_liberal) + " %")

    acc_score_cons = round(((cm[1][1] / data_count[1]) * 100), 2)
    print("Accuracy for class Conservative: " + str(acc_score_cons) + " %")

    print("\nTime taken: " + str(end-start))

    #print(cm)
    return (acc, acc_score_liberal, acc_score_cons, end-start, cm)


if __name__ == "__main__":
    run_classifier()