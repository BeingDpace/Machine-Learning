import operator
import numpy as np
def getDistance(i1, i2):
    """
    :parameter i1: Float array
    :parameter i2: Float array
    :return: Euclidean distance between i1 and i2 (using l2-norm of vectors)
    """
    return np.linalg.norm(np.asarray(i1[:len(i1) - 1]) - np.asarray(i2[:len(i2) - 1]))


def getNeighbours(train, testInstance, k):
    """
    :parameter train: training input array.
    :parameter testInstance: test input array.
    :parameter k: k no. of k neighbours
    :return: k nearest neighbours to testInstance
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


def predictClass(neighbours):
    """
    :parameter neighbours: k nearest neighbours to testInstance
    :return: max likely class of test instance
    """
    classes = {}
    for neighbour in neighbours:
        currClass = neighbour[-1]
        if currClass in classes:
            classes[currClass] += 1
        else:
            classes[currClass] = 1
    return max(classes.items(), key=operator.itemgetter(1))[0]


def getAccuracy(X, test, k):
    """
    :parameter X: Input training data
    :parameter test: Input test data
    :parameter k: k no. of neighbours
    """
    predictions = []
    for testInstance in test:
        neighbours = getNeighbours(X, testInstance, k)
        prediction = predictClass(neighbours)
        predictions.append(prediction)
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1
    print('Accuracy: ' + repr((correct / float(len(test))) * 100.0))
    return (correct / float(len(test))) * 100.0


def myknnclassify(X, test, k):
    """
    :param X: Input training data
    :param test: Input test instance
    :param k: k no. of neighbours
    :return: predicted label for test data
    """
    neighbours = getNeighbours(X, test, k)
    prediction = predictClass(neighbours)
    return prediction
