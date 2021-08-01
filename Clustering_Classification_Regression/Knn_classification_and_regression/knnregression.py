import operator
import numpy as np
def getDistance(i1, i2):
    """
    :parameter i1: Float array
    :parameter i2: Float array
    :return: Euclidean distance between i1 and i2 (using l2-norm vector)
    """
    return np.linalg.norm(np.asarray(i1) - np.asarray(i2))


def getNeighbours(train, testInstance, k):
    """
    :param training: training input array.
    :param testInstance: test input array.
    :param k: k no. of neighbours
    :return: k nearest neighbours to testInstance.
    """
    distances = []
    for instance in train:
        dist = getDistance(testInstance, instance)
        distances.append((instance, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for instance in range(k):
        neighbours.append(distances[instance][0])
    return neighbours


def calculateValue(neighbours, k):
    """
    :param neighbours: k nearest neighbours to testInstance
    :return: average of top k neighbours
    """
    sum = 0
    for neighbour in neighbours:
        sum += neighbour[-1]
    return sum/k


def getAccuracy(X, test, k):
    """
    :parameter X: Input training data. Array of array of floats
    :parameter test: Input test data. Array of array of floats
    :parameter k: k hyperparameter
    """
    predictions = []
    for testInstance in test:
        neighbours = getNeighbours(X, testInstance, k)
        prediction = calculateValue(neighbours, k)
        predictions.append(prediction)
    correct = 0
    for i in range(len(test)):
        # print('predicted: ' + repr(predictions[i]) + ' | actual: ' + repr(test[i][-1]))
        if (predictions[i] - 0.15) <= test[i][-1] <= (predictions[i] + 0.15):
            correct += 1
    print('Accuracy: ' + repr((correct / float(len(test))) * 100.0))
    return (correct / float(len(test))) * 100.0


def myknnregress(X, test, k):
    """
    :parameter X: Training dataset
    :parameter test: Test dataset
    :parameter k: k no. of neigbours
    :return predicted label for test data
    """
    neighbours = getNeighbours(X, test, k)
    prediction = calculateValue(neighbours, k)
    return prediction
