import numpy as np
from sklearn.datasets import fetch_openml
def load_data()-> tuple:
    mnist = fetch_openml("mnist_784")
    data, labels = mnist["data"].astype(np.float32), mnist["target"].astype(np.float32)
    test = data[60000:]
    test_labels = labels[60000:]
    data = data[:60000]
    labels = labels[:60000]
    f_train_data = []
    f_train_labels = []
    f_test_data = []
    f_test_labels = []
    for i in range(len(labels)):
        if labels[i] < 5:
            f_train_data.append(data[i])
            f_train_labels.append(labels[i])
    for i in range(len(test_labels)):
        if test_labels[i] < 5:
            f_test_data.append(test[i])
            f_test_labels.append(test_labels[i])
    return np.array(f_train_data), np.array(f_train_labels), np.array(f_test_data), np.array(f_test_labels)


def convert_to_one_hot(train: list)-> tuple:
    one_hot_train = []
    for i in range(len(train)):
        temp = np.zeros(5)
        temp[int(train[i])] = 1
        one_hot_train.append(temp)
    return np.array(one_hot_train)


def softmax(prediction)-> np.ndarray:
    e_x = np.exp(prediction - np.max(prediction))
    return e_x / e_x.sum()


def logistic_regression(data: list, labels: list, max_iter: int, learning_rate: float)-> list:
    # print("data", data.shape)
    weights = np.zeros((784, 5))
    # print("weights", weights.shape)
    # print("prediction", np.dot(data, weights).shape)
    for i in range(max_iter):
        prediction = np.dot(data, weights)
        prediction = softmax(prediction)
        error = labels - prediction
        gradient = np.dot(data.T, error)
        weights += learning_rate * gradient
    return weights


def calc_metrics(test_data, test_label, weights)-> tuple:
    acc = 0
    predictions = []
    confusion_matrix = np.zeros((5, 5))
    # accuracy
    for i in range(len(test_data)):
        prediction = np.dot(test_data[i], weights)
        prediction = softmax(prediction)
        predictions.append(prediction)
        if test_label[i] == np.argmax(prediction):
            acc += 1
        confusion_matrix[np.argmax(prediction)][int(test_label[i])] += 1
    # precision
    precision = 0
    for i in range(5):
        precision += confusion_matrix[i][i] / sum(confusion_matrix[:,i])
    precision /= 5
    # recall
    recall = 0
    for i in range(5):
        recall += confusion_matrix[i][i] / sum(confusion_matrix[i])
    recall /= 5
    return acc, predictions, confusion_matrix, precision, recall
