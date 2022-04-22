from pandas import read_csv
import pathlib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



project_root = pathlib.Path().resolve().parent
inputF = open(os.path.join(project_root, 'data', 'generated', 'tokenized_features.csv'), "r")

df = read_csv(inputF)
X = df.values[:, :-1]
Y = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#print(type(y_train))

def desc_data(y):
    count = {0:0, 1:0 }
    for cls in y:
        count[cls] += 1
    for cl in count.keys():
        if cl == 0:
            print("Number of instances in class Liberal: " + str(count[cl]))
        else:
            print("Number of instances in class Conservative: " + str(count[cl]))


print("\nDescribing training set...")
desc_data(y_train)

print("\nDescribing testing set...")
desc_data(y_test)

logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
