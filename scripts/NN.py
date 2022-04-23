from torch.utils.data import Dataset
from pandas import read_csv
from collections import OrderedDict

import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, BCELoss
from torch import nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from sklearn.model_selection import train_test_split


class CustomTorchDataset(Dataset):
    def __init__(self, features, labels, target_transform=None):
        self.X = features
        self.Y = labels

        self.X = self.X.astype('float32')

        self.no_classes = len(set(self.Y))
        self.Y = self.Y.astype('float32')
        #self.Y = self.Y.reshape((len(self.Y), 1))
        self.target_transform = target_transform
        self.data_stat()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.Y[idx]
        #print(features)
        #features = torch.tensor(features)
        #label = torch.LongTensor(label)
        #if self.target_transform:
        #    label = self.target_transform(label)
        #print(features)
        #print("----")
        #print(label)
        return features, label

    def data_stat(self):
        print("\nNumber of classes in the dataset: " + str(self.no_classes))

        countDataInClass = {}

        for i in range(self.no_classes):
            countDataInClass[i] = {'count': 0}

        for y in self.Y:
            countDataInClass[y]["count"] += 1

        for cls in countDataInClass.keys():
            if cls == 0:
                print("Number of instances in class Liberal: " + str(countDataInClass[cls]['count']))
            else:
                print("Number of instances in class Conservative: " + str(countDataInClass[cls]['count']))




class ClassificationModel(Module):
    # define model elements
    def __init__(self, num_layers, nodes_hidden, n_inputs=37, n_outputs=1):
        super(ClassificationModel, self).__init__()
        # input to first hidden layer
        layers = OrderedDict()

        if num_layers == 2:
            layers["hidden1"] = Linear(n_inputs, n_outputs)
            layers["act1"] = Sigmoid()
        else:
            nodes_layers = [n_inputs]
            for val in nodes_hidden:
                nodes_layers.append(val)
            nodes_layers.append(n_outputs)

            for i in range(len(nodes_layers)-1):
                keyH = "hidden"+str(i+1)
                keyA = "act"+str(i+1)

                layers[keyH] = Linear(nodes_layers[i], nodes_layers[i+1])
                if i != (len(nodes_layers)-2):
                    layers[keyA] = ReLU()
                else:
                    layers[keyA] = Sigmoid()

        self.network = nn.Sequential(layers)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        return self.network(X)


def train_model(train_dataset, batch_size, lrn_rt, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=lrn_rt)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_val = 0
    # enumerate mini batches
    for i, (inputs, targets) in enumerate(train_dl):
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        output = model(inputs)
        # calculate loss
        loss = criterion(output, targets.unsqueeze(1))
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()

        loss_val = loss.item()

    print(f"loss: {loss_val:>7f}")
    return {"loss": loss_val}


def evaluate_model(dataset, model):
    dl = DataLoader(dataset, batch_size=100, shuffle=True)
    size = len(dl.dataset)
    num_batches = len(dl)
    test_loss, correct = 0, 0
    loss_fn = BCELoss()
    no_classes = 2

    prediction_accuracy = {}
    for i in range(no_classes):
        prediction_accuracy[i] = {'correct': 0, 'total': 0}

    with torch.no_grad():
        for X, y in dl:
            pred = model(X)
            out = y.numpy()
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            #print(type(pred))
            predicted_class = np.round(pred.detach())

            for i in range(len(predicted_class)):
                prediction_accuracy[out[i]]['total'] += 1
                if predicted_class[i] == out[i]:
                    prediction_accuracy[out[i]]['correct'] += 1
                    correct += 1

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(prediction_accuracy)
    for cls in prediction_accuracy.keys():
        if cls == 0:
            print(f"accuracy for class Liberal: {(100 * (prediction_accuracy[cls]['correct'] / prediction_accuracy[cls]['total'])):>0.1f}%")
        else:
            print(f"accuracy for class Conservative: {(100 * (prediction_accuracy[cls]['correct'] / prediction_accuracy[cls]['total'])):>0.1f}%")
        #print(f"accuracy for class {dataset.le.inverse_transform([cls])}: {(100 * (prediction_accuracy[cls]['correct'] / prediction_accuracy[cls]['total'])):>0.1f}%")

    return {"accuracy": (100 * correct), "perclass": prediction_accuracy, "loss": test_loss}

project_root = pathlib.Path().resolve().parent
dataDir = 'data'
dataSubDir = 'generated'
file = 'tokenized_features.csv'
#data_set = CustomTorchDataset(os.path.join(project_root, dataDir, dataSubDir, file))

df = read_csv(os.path.join(project_root, dataDir, dataSubDir, file))
X = df.values[:, :-1]
Y = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

train_dataset = CustomTorchDataset(x_train, y_train)
test_dataset = CustomTorchDataset(x_test, y_test)
#val_dataset = CustomTorchDataset(os.path.join(project_root, dataDir, dataSubDir, 'validation.csv'))

number_layers = int(input("\nEnter the number of layers for the neural network including the input and output layer (i.e, value > 2): "))
if number_layers < 2:
    raise Exception("number of layers cannot be less than 2, exiting...")
nodes_hidden = []
for i in range(number_layers-2):
    nodes = int(input("Enter the number of nodes at "+ str(i+1) + " hidden layer (layer " + str(i+2) + "): "))
    if nodes < 1:
        raise Exception("number of nodes at a layer cannot be less than 1, exiting...")
    nodes_hidden.append(nodes)

model = ClassificationModel(number_layers, nodes_hidden)

hyper_parameters = {"epoch": [30, 25, 25], "batch_size": [5, 10, 15], "learning_rate": [0.05, 0.01, 0.05]}
#hyper_parameters = {"epoch": [30], "batch_size": [5], "learning_rate": [0.05]}

min_loss = 99999999
best_at_epoch = -1
best_batch_size = -1
best_lrn_rt = -1
best_epochs = -1

fig, ax = plt.subplots(len(hyper_parameters["epoch"]), 1, sharey=True)

for it in range(len(hyper_parameters["epoch"])):
    epochs = hyper_parameters["epoch"][it]
    batch_size = hyper_parameters["batch_size"][it]
    lrn_rt = hyper_parameters["learning_rate"][it]

    print("\nRunning training with following hyper parameters:")
    print("Number of epochs: " + str(epochs))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(lrn_rt))

    plot_X = np.arange(0, epochs, 1)
    train_loss = []
    #val_loss = []

    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        print("\nTraining...")
        temp_loss = train_model(train_dataset, batch_size, lrn_rt, model)["loss"]
        train_loss.append(temp_loss)
        #print("\nValidation")
        #val_accuracy = evaluate_model(val_dataset, model)
        #val_loss.append(val_accuracy["loss"])
        if temp_loss < min_loss:
        #    max_accuracy = val_accuracy["accuracy"]
            min_loss = temp_loss
            best_lrn_rt = lrn_rt
            best_batch_size = batch_size
            best_at_epoch = t
            best_epochs = epochs
            torch.save(model.state_dict(), 'weights_only.pth')
    if len(hyper_parameters["epoch"]) == 1:
        ax.plot(plot_X, train_loss, color='red', label="train_loss: ep="+str(epochs)+", b_size="+str(batch_size)+", lrn_rt=" + str(lrn_rt))
        #ax.plot(plot_X, val_loss, color='green', label="val_loss: ep="+str(epochs)+", b_size="+str(batch_size)+", lrn_rt=" + str(lrn_rt))
        ax.legend(loc='upper right')
    else:
        ax[it].plot(plot_X, train_loss, color='red', label="train_loss: ep="+str(epochs)+", b_size="+str(batch_size)+", lrn_rt=" + str(lrn_rt))
        #ax[it].plot(plot_X, val_loss, color='green', label="val_loss: ep="+str(epochs)+", b_size="+str(batch_size)+", lrn_rt=" + str(lrn_rt))
        ax[it].legend(loc='upper right')

plt.show()
print("\nHyper parameters after tuning:")
print("Number of epochs:" + str(best_epochs))
print("Learning rate:" + str(best_lrn_rt))
print("Batch size:" + str(best_batch_size))

# loading the state_dict
model_new = ClassificationModel(number_layers, nodes_hidden)
model_new.load_state_dict(torch.load('weights_only.pth'))

print("\nTesting on the best model...")
print("\nAccuracy and loss for train dataset:")
evaluate_model(train_dataset, model_new)
#print("\nAccuracy and loss for validation dataset:")
#evaluate_model(val_dataset, model_new)
print("\nAccuracy and loss for test dataset:")
evaluate_model(test_dataset, model_new)
print("Done!")