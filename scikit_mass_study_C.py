#!/usr/bin/env python
# encoding: utf-8

import pylab as pl
import re
import numpy as np
import PIL.Image
import preprocessing_lib as pp
from math import isclose, sqrt
from scipy.stats import entropy
from numpy import linalg as LA
from PIL import Image, ImageDraw


def polyarea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def com(segmented_image):
    (X, Y) = segmented_image.shape
    m = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            m[x, y] = segmented_image[(x, y)] != 1.0
    m = m / np.sum(np.sum(m))
    # marginal distributions
    dx = np.sum(m, 1)
    dy = np.sum(m, 0)
    # expected values
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))
    return cx, cy


class Mammogram(object):

    perimeter = 0
    area = 0
    xlc = 0
    ylc = 0
    contour = []
    mask = []
    rk = []

    def __init__(self, name, centrex, centrey, radius, type, Abclass, severity):
        self.name = name
        self.centrex = centrex
        self.centrey = centrey
        self.radius = radius
        self.type = type
        self.Abclass = Abclass
        self.severity = severity
        self.array = self.cropping_image()
        self.full_image = self.full_image()

    def full_image(self):
        full_image = PIL.Image.open(
            r"C:\Users\James_000\Documents\University\Third Year\BEng Final Project\MIAS Mini Database/"+self.name+".pgm")
        return np.asarray(full_image, dtype=np.uint8)

    def cropping_image(self):
        full_image = PIL.Image.open(
            r"C:\Users\James_000\Documents\University\Third Year\BEng Final Project\MIAS Mini Database/"+self.name+".pgm")
        startx = self.centrex - (self.radius+(self.radius*0.2))
        starty = self.centrey - (self.radius+(self.radius*0.2))
        width = self.centrex + (self.radius+(self.radius*0.2))
        height = self.centrey + (self.radius+(self.radius*0.2))
        area = (startx, starty, width, height)
        cropped = full_image.crop(area)
        return np.asarray(cropped, dtype=np.uint8)

    def cropping(self, x, y, r):
        full_image = PIL.Image.open(
            r"C:\Users\James_000\Documents\University\Third Year\BEng Final Project\MIAS Mini Database/"+self.name+".pgm")
        startx = x - r
        starty = y - r
        width = x + r
        height = y + r
        area = (startx, starty, width, height)
        cropped = full_image.crop(area)
        return np.asarray(cropped, dtype=np.uint8)

    def segmentation(self):
        r_min = self.radius*0.8
        a_max = np.pi*r_min**2
        global_max = np.max(self.array)
        xy = np.argwhere(self.array == global_max)
        Mammogram.xlc = xy[0][0]
        Mammogram.ylc = xy[0][1]
        thresh = int(global_max/2)
        edge_cords = pp.locate_contours(self.array, thresh, self.name)
        edge_x = edge_cords[:, 1]
        edge_y = edge_cords[:, 0]
        area = polyarea(edge_x, edge_y)
        new_thresh = thresh
        old_thresh = thresh + 100
        while not (area/a_max < 1.05 and a_max/area < 1.05) or \
                (old_thresh/new_thresh < 1.1 and new_thresh/old_thresh < 1.1):
            old_thresh = thresh
            if area > a_max or edge_cords[0].all() != edge_cords[-1].all():
                thresh = int(thresh/2 + thresh)
                edge_cords = pp.locate_contours(self.array, thresh, self.name)
                edge_x = edge_cords[:, 1]
                edge_y = edge_cords[:, 0]
                area = polyarea(edge_x, edge_y)
            elif area < a_max and edge_cords[0].all() == edge_cords[-1].all():
                thresh = int(thresh - thresh/2)
                edge_cords = pp.locate_contours(self.array, thresh, self.name)
                edge_x = edge_cords[:, 1]
                edge_y = edge_cords[:, 0]
                area = polyarea(edge_x, edge_y)
            new_thresh = thresh
        Mammogram.perimeter = len(edge_cords)
        Mammogram.area = area
        edge_cords[:, [0, 1]] = edge_cords[:, [1, 0]]
        Mammogram.contour = edge_cords
        polygon = mammogram.contour
        polygon = np.asarray(polygon, dtype=int)
        polygon = list(map(tuple, polygon))
        img = Image.new('L', (mammogram.array.shape), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)
        Mammogram.mask = mask
        copy = self.cropping_image()
        copy.setflags(write=True)
        copy[mask == 0] = 0.0
        self.array = copy
        pp.show_function(self.array, self.name)
        return xy

    def circularity(self):
        return Mammogram.perimeter**2/Mammogram.area

    def radk(self):
        rk = np.array([], dtype=np.int64)
        for x, y in Mammogram.contour:
            rk = np.append(rk, sqrt((self.centrex - x) ** 2 + (self.centrey - y) ** 2))
        Mammogram.rk = rk/np.max(rk)

    def radial_mean(self):
        return np.mean(Mammogram.rk)

    def radial_std(self):
        return np.std(Mammogram.rk)

    def intensity_entropy(self):
        probs = []
        unique, counts = np.unique(self.array, return_counts=True)
        unique = unique.astype(int)
        for i in range(0, counts.size):
            probs.append(counts[i] / (self.array.shape[0] * self.array.shape[1]))
        return entropy(probs, base=2)

    def fractal_index(self):
        return pp.fractal_dimension(Mammogram.mask)

    def eccentricity(self):
        matrix = np.empty((0, 2), float)
        a11 = np.sum((Mammogram.contour[:, 1] - self.centrex)**2)
        a12 = np.sum(np.dot((Mammogram.contour[:, 1] - self.centrex), (Mammogram.contour[:, 0] - self.centrey)))
        a22 = np.sum((Mammogram.contour[:, 0] - self.centrey)**2)
        row = [a11, a12]
        matrix = np.vstack((matrix, row))
        row = [a12, a22]
        matrix = np.vstack((matrix, row))
        vals = LA.eigvalsh(matrix)
        lambda1 = vals[0]
        lambda2 = vals[1]
        s1 = sqrt(abs(lambda1/2))
        s2 = sqrt(abs(lambda2/2))
        return s1/s2

    def luminosity(self):
        outerx_nom = np.array([], dtype=np.int64)
        outery_nom = np.array([], dtype=np.int64)
        outer_denom = np.array([], dtype=np.int64)
        for i in range(0, self.array.shape[1]):
            innerx_nom = []
            innery_nom = []
            inner_denom = []
            for j in  range(0, self.array.shape[0]):
                x = j
                y = i
                if Mammogram.mask[x, y] == 0:
                    resultx_nom = 0
                    resulty_nom = 0
                    result_denom = 0
                else:
                    ixy = self.array[x, y]
                    resultx_nom = (ixy*x)
                    resulty_nom = (ixy*y)
                    result_denom = ixy
                innerx_nom.append(resultx_nom)
                innery_nom.append(resulty_nom)
                inner_denom.append(result_denom)
            outerx_nom = np.append(outerx_nom, sum(innerx_nom))
            outery_nom = np.append(outery_nom, sum(innery_nom))
            outer_denom = np.append(outer_denom, sum(inner_denom))
        Mammogram.xlc = int(sum(outerx_nom)/sum(outer_denom))
        Mammogram.ylc = int(sum(outery_nom)/sum(outer_denom))

    def inertial_momentum(self):
        result = np.array([], dtype=np.int64)
        for i in range(0, self.array.shape[1]):
            for j in  range(0, self.array.shape[0]):
                distance = sqrt((Mammogram.xlc - j) ** 2 + (Mammogram.ylc - i) ** 2)
                val = self.array[j, i]
                result = np.append(result, val*(distance**2))
        return np.sum(result)


    def contour_entropy(self):
        directions = []
        for x, y in Mammogram.contour:
            values = {}
            x = int(x)
            y = int(y)
            black = int(self.array[x, y])
            values[1] = abs(black - int(self.array[x-1, y-1]))
            values[2] = abs(black - int(self.array[x, y-1]))
            values[3] = abs(black - int(self.array[x+1, y-1]))
            values[4] = abs(black - int(self.array[x+1, y]))
            values[5] = abs(black - int(self.array[x+1, y-1]))
            values[6] = abs(black - int(self.array[x, y-1]))
            values[7] = abs(black - int(self.array[x-1, y-1]))
            values[8] = abs(black - int(self.array[x-1, y]))
            directions.append(max(values, key=values.get))
        unique, counts = np.unique(directions, return_counts=True)
        probs = counts/sum(counts)
        return entropy(probs, base=2)

    def anisotropy(self):
        geo_x, geo_y = com(self.array)
        return sqrt((Mammogram.xlc - geo_x) ** 2 + (Mammogram.ylc - geo_y) ** 2)

    def mean(self):
        return np.mean(self.array)

    def std(self):
        return np.std(self.array)

    def segment_area(self):
        return Mammogram.area

    def features(self):
        self.segmentation()
        self.luminosity()
        self.radk()
        feature_list = [self.radial_mean(), self.radial_std(), self.intensity_entropy(),
                        self.fractal_index(), self.eccentricity(), self.inertial_momentum(),
                        self.contour_entropy(), self.anisotropy(), self.mean(), self.std(),
                        self.segment_area()]
        return feature_list




info_file = open(r"C:\Users\James_000\Documents\University\Third Year\BEng Final Project\MIAS Mini Database\Information.txt")
for line in info_file:
    # if line[9:13] == "NORM":
    #     name = line[0:6]
    #     type = line[7:8]
    #     radius
    if len(line) > 20:
        abclass = line[9:13]
        if abclass == 'CIRC':#not in ['CALC', 'ASYM', 'ARCH']:
            name = line[0:6]
            type = line[7:8]
            severity = line[14:15]
            x_cord = int(re.sub('^([0-9]+)\s([0-9]+)\s([0-9]+)', r'\1', line[16:], 0, re.I).strip())
            # x_cord = 1024/2
            y_cord = 1024 - int(re.sub('^([0-9]+)\s([0-9]+)\s([0-9]+)', r'\2', line[16:], 0, re.I).strip())
            # y_cord = 1024/2
            radius = int(re.sub('^([0-9]+)\s([0-9]+)\s([0-9]+)', r'\3', line[16:], 0, re.I).strip())
            radius = int(radius + (radius*0.2))
            mammogram = Mammogram(name, x_cord, y_cord, radius, type, abclass, severity)
            print(mammogram.features())

