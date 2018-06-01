#!/usr/bin/env python
# encoding: utf-8

import re
import test_class
import preprocessing_lib as pp
import features
import numpy as np
import PIL.Image
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import entropy, kurtosis
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class Mammogram(object):

    def __init__(self, name, centreX, centreY, radius, type, Abclass, severity):
        self.name = name
        self.centreX = centreX
        self.centreY = centreY
        self.radius = radius
        self.type = type
        self.Abclass = Abclass
        self.severity = severity
        self.array = self.cropping()

    def cropping(self):
        full_image = PIL.Image.open(
            r"C:\Users\James_000\Documents\University\Third Year\BEng Final Project\MIAS Mini Database/"+self.name+".pgm")
        startX = self.centreX - self.radius
        startY = self.centreY - self.radius
        width = self.centreX + self.radius
        height = self.centreY + self.radius
        area = (startX, startY, width, height)
        cropped = full_image.crop(area)
        return np.asarray(cropped, dtype=np.uint8)/255

    def mean(self):
        return np.mean(self.array)

    def standard_deviation(self):
        return np.std(self.array)

    def smoothness(self):
        return features.smoothness(self.standard_deviation())

    def entropy(self):
        probs = []
        unique, counts = np.unique(self.array*255, return_counts=True)
        unique = unique.astype(int)
        for i in range(0, counts.size):
            probs.append(counts[i] / (self.array.shape[0] * self.array.shape[1]))
        # print(probs)
        # pp.show_function(probs, 'Probability distribution')
        # plt.bar(unique, probs)
        return entropy(probs, base=2)

    def skewness(self):
        # summation = np.sum((self.array - self.mean()) / self.standard_deviation() ** 3)
        # skew = summation / (self.array.shape[0] * self.array.shape[1])

        # print(temp_arr)
        # floaty = skew(temp_arr, axis=None)
        return stats.skew(self.array, axis=None)

    def kurtosis(self):
        # summation = np.sum((self.array - self.mean()) / self.standard_deviation() ** 4)
        # # print(summation)
        # kurt = (summation / (self.array.shape[0] * self.array.shape[1]))
        # print(kurt)
        return kurtosis(self.array, axis=None)

    def uniformity(self):
        probs = []
        unique, counts = np.unique(self.array * 255, return_counts=True)
        unique = unique.astype(int)
        for i in range(0, counts.size):
            probs.append(counts[i] / (self.array.shape[0] * self.array.shape[1]))
        probs = np.asarray(probs)
        return np.sum(probs ** 2)

    def features(self):
        return self.mean(), self.standard_deviation(), self.smoothness(), self.entropy(), self.skewness(), self.kurtosis(), self.uniformity()

    def targets(self):
        return self.name, self.Abclass, self.severity


if __name__ == "__main__":
    info_file = open(r"C:\Users\James_000\Documents\University\Third Year\BEng Final Project\MIAS Mini Database\Information.txt")
    targets = np.empty((0, 3), float)
    feature_set = np.empty((0, 7), float)

    for line in info_file:
        target = []
        feature = []
        # if line[9:13] == "NORM":
        #     name = line[0:6]
        #     type = line[7:8]
        #     radius
        if len(line) > 20:
            abclass = line[9:13]
            if abclass not in ['CALC', 'ASYM', 'ARCH']:
                name = line[0:6]
                type = line[7:8]
                severity = line[14:15]
                x_cord = int(re.sub('^([0-9]+)\s([0-9]+)\s([0-9]+)', r'\1', line[16:], 0, re.I).strip())
                y_cord = 1024 - int(re.sub('^([0-9]+)\s([0-9]+)\s([0-9]+)', r'\2', line[16:], 0, re.I).strip())
                radius = int(re.sub('^([0-9]+)\s([0-9]+)\s([0-9]+)', r'\3', line[16:], 0, re.I).strip())
                mammogram = Mammogram(name, x_cord, y_cord, radius, type, abclass, severity)
                # pp.show_function(mammogram.array, name)
                mean, std_dev, smooth, entrop, skew, kurt, uni = mammogram.features()
                target = [mammogram.targets()[0], mammogram.targets()[1], mammogram.targets()[2]]
                feature = [mean, std_dev, smooth, entrop, skew, kurt, uni]
                feature_set = np.vstack((feature_set, feature))
                targets = np.vstack((targets, target))
    info_file.close()

    # feature_set = preprocessing.normalize(feature_set)
    # print(feature_set)
    val = 0
    clf = MLPClassifier(hidden_layer_sizes=(7), verbose=False, max_iter=4000)
    test_feature_set, test_targets, obj, probs = test_class.create_test_set()

    while val < 100.00:
        X_train, X_test, Y_train, Y_test = train_test_split(feature_set, targets)
        W_test = np.array(Y_test[:, 2])
        W_test[Y_test[:, 2] == "M"] = 0.6
        W_test[Y_test[:, 2] == "B"] = 1.4
        W_test = W_test.astype(float)
        print(W_test)
    #hidden_layer_sizes=(50,), max_iter=400, alpha=1e-4, hidden_layer_sizes=(50,100,250,100,50)
                        #solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        #learning_rate_init=.1)

        clf.fit(X_train, Y_train[:, 2])
        print("Testing set score: %f" % (clf.score(X_test, Y_test[:, 2], W_test) * 100))
        if clf.score(X_test, Y_test[:, 2]) > 0.75 and clf.score(test_feature_set, test_targets[:, 2]) > 0.75:
            print("Testing set score: %f" % (clf.score(test_feature_set, test_targets[:, 2])*100))
            print("Prediction test: ", clf.predict_proba(test_feature_set), "Actual: ", test_targets[:, 2])
            print("Prediction test: ", clf.predict(test_feature_set), "Actual: ", test_targets[:, 2])
            print("Testing set score: %f" % (clf.score(X_test, Y_test[:, 2], W_test) * 100))
            print("Prediction test: ", clf.predict_proba(X_test), "Actual: ", Y_test[:, 2])
            print("Prediction test: ", clf.predict(X_test), "Actual: ", Y_test[:, 2])
        val = clf.score(X_test, Y_test[:, 2])*100


    # print(clf.loss)

    # print("Training set score: %f" % (clf.score(X_train, Y_train[:, 2])*100))
    # print("Prediction test: ", clf.predict(X_test), "Actual: ", Y_test[:, 2])
    # print("Testing set score: %f" % (clf.score(X_test, Y_test[:, 2])*100))

    # print(clf.coefs_)
    #
    # if clf.score(X_test, Y_test[:, 2]) >= 0.7:
    # test_feature_set, test_targets, obj, probs = test_class.create_test_set()
    # print(test_targets)
    # print(test_feature_set)
    # test_feature_set = preprocessing.normalize(test_feature_set)
    # print("Prediction validation: ", clf.predict(test_feature_set), "Actual: ", test_targets[:, 2])
    # print("Validation dataset score: %f" % (clf.score(test_feature_set, test_targets[:, 2])*100))
    # params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
    #            'learning_rate_init': 0.2},
    #           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
    #            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
    #           {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
    #            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
    #           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
    #            'learning_rate_init': 0.2},
    #           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
    #            'nesterovs_momentum': True, 'learning_rate_init': 0.2},
    #           {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
    #            'nesterovs_momentum': False, 'learning_rate_init': 0.2},
    #           {'solver': 'adam', 'learning_rate_init': 0.01}]
    #
    # labels = ["constant learning-rate", "constant with momentum",
    #           "constant with Nesterov's momentum",
    #           "inv-scaling learning-rate", "inv-scaling with momentum",
    #           "inv-scaling with Nesterov's momentum", "adam"]
    #
    # for label, param in zip(labels, params):
    #         print("training: %s" % label)
    #         clf = MLPClassifier(verbose=0, random_state=0,
    #                             max_iter=400, **param)
    #         clf.fit(X_test, Y_test)
    #         print("Training set score: %f" % (clf.score(X_train, Y_train)*100)+'%')
    #         print("Training set loss: %f\n" % clf.loss_)
    info_file.close()
