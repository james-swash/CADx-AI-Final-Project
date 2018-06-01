#!/usr/bin/env python
# encoding: utf-8

import re
import os
import skimage
import preprocessing_lib as pp
import features
import numpy as np
import PIL.Image
from scipy import stats
from scipy.stats import entropy, kurtosis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class Mammogram(object):

    def __init__(self, name, centreX, centreY, radius, type, Abclass, severity, array):
        self.name = name
        self.centreX = centreX
        self.centreY = centreY
        self.radius = radius
        self.type = type
        self.Abclass = Abclass
        self.severity = severity
        self.array = array/255

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
        return entropy(probs, base=2)

    def skewness(self):
        # summation = np.sum((self.array - self.mean()) / self.standard_deviation() ** 3)
        # skew = summation / (self.array.shape[0] * self.array.shape[1])
        return stats.skew(self.array, axis=None)

    def kurtosis(self):
        # summation = np.sum((self.array - self.mean()) / self.standard_deviation() ** 4)
        # # print(summation)
        # kurt = (summation / (self.array.shape[0] * self.array.shape[1]))
        # # print(kurt)
        return kurtosis(self.array, axis=None)

    def uniformity(self):
        probs = []
        unique, counts = np.unique(self.array * 255, return_counts=True)
        unique = unique.astype(int)
        for i in range(0, counts.size):
            probs.append(counts[i] / (self.array.shape[0] * self.array.shape[1]))
        probs = np.asarray(probs)
        return np.sum(probs ** 2)

    def probability(self):
        probs = []
        unique, counts = np.unique(self.array * 255, return_counts=True)
        unique = unique.astype(int)
        for i in range(0, counts.size):
            probs.append(counts[i] / (self.array.shape[0] * self.array.shape[1]))
        probs = np.asarray(probs)
        return probs

    def features(self):
        return self.mean(), self.standard_deviation(), self.smoothness(), self.entropy(), self.skewness(), self.kurtosis(), self.uniformity()

    def targets(self):
        return self.name, self.Abclass, self.severity


def create_test_set():
    old_file = "Cam006 - L - CC"
    test_targets = np.empty((0, 3), float)
    test_feature_set = np.empty((0, 7), float)
    f = []
    object_list = []
    probabilities = []
    images_per_patient = 0
    for root, dirs, files in os.walk(
            r'C:\Users\James_000\Documents\University\Third Year\BEng Final Project\Mammograms'):
        f.extend(files)
        for item in f:
            if item[-4:] == '.bmp':
                file_loc = root + "\\" + item
                file = os.path.join(skimage.data_dir, file_loc)
                array = pp.image_converter(file)

                # plt.hist(array, bins=20, color='c')
                patient_image = re.sub('(^[A-Z]+[0-9]+[-][R|L][-]CC|MLO)(.*$)', r'\1', item, 0, re.I)
                if old_file == patient_image:
                    images_per_patient += 1
                else:
                    old_file = patient_image
                    images_per_patient = 0
                for file in f:
                    if file[:len(patient_image)] == patient_image and file[-17:] == '.bmp_FCanavan.txt':
                        image = re.sub('(^[A-Z]+[0-9]+[-][R|L][-]CC|MLO)(.*$)', r'\1', file, 0, re.I)
                        if images_per_patient > 0:
                            infile = open(root + '\\' + file, 'r')
                            line_no = 0
                            for line in infile:
                                line_no += 1
                                if line_no == images_per_patient + 1:
                                    type = re.sub('^[0-9]+\s[0-9]+\s[0-9]+\s[0-9]+\s(\w+)\s(\d).*$', r'\1',
                                                  line, 0, re.I).strip()
                                    severity = re.sub('^[0-9]+\s[0-9]+\s[0-9]+\s[0-9]+\s(\w+)\s(\d).*$', r'\2',
                                                      line, 0, re.I).strip()
                            infile.close()
                        else:
                            infile = open(root + '\\' + file, 'r')
                            line_no = 0
                            for line in infile:
                                line_no += 1
                                if line_no == 1:
                                    type = re.sub('^[0-9]+\s[0-9]+\s[0-9]+\s[0-9]+\s(\w+)\s(\d).*$', r'\1',
                                                  line, 0, re.I).strip()
                                    severity = re.sub('^[0-9]+\s[0-9]+\s[0-9]+\s[0-9]+\s(\w+)\s(\d).*$', r'\2',
                                                      line, 0, re.I).strip()
                            infile.close()
                        if type == "Mass":
                            abclass = "SPIC"
                        else:
                            abclass = "CALC"
                        if int(severity) >= 5:
                            severity = 'M'
                        else:
                            severity = 'B'
                        x_dist = array.shape[0]
                        y_dist = array.shape[0]
                        x_cord = int(x_dist/2)
                        y_cord = int(y_dist/2)
                        if y_dist < x_dist:
                            radius = int(y_dist/2) - int(y_dist*0.1)
                        else:
                            radius = int(x_dist/2) - int(x_dist*0.1)
                        if abclass == "SPIC":
                            mammogram = Mammogram(image, x_cord, y_cord, radius, type, abclass, severity, array)
                            object_list.append(mammogram)
                            probabilities.append(mammogram.probability())
                            mean, std_dev, smooth, entrop, skew, kurt, uni = mammogram.features()
                            target = [mammogram.targets()[0], mammogram.targets()[1], mammogram.targets()[2]]
                            feature = [mean, std_dev, smooth, entrop, skew, kurt, uni]
                            test_feature_set = np.vstack((test_feature_set, feature))
                            test_targets = np.vstack((test_targets, target))
    # print(test_feature_set)
    return test_feature_set, test_targets, object_list, probabilities

create_test_set()