import skimage
import time
from scipy import stats
import preprocessing_lib as pp
import numpy as np
import os
import matplotlib.pyplot as plt
from os import walk
import re
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pylab as pl

class Mammogram(object):
    no_mammograms = 0

    def __init__(self, image, severity, type, array=[]):
        self.image = image
        self.type = type
        self.type_number = severity
        self.area = array.shape
        if re.sub('(^[A-Z]+[0-9]+[-])([A-Z])(.*$)', r'\2', image, 0, re.I) == 'L':
            self.side = 'Left Breast'
        elif re.sub('(^[A-Z]+[0-9]+[-])([A-Z])(.*$)', r'\2', image, 0, re.I) == 'R':
            self.side = 'Right Breast'
        # self.side = re.sub('(^[A-Z]+[0-9]+[-])([A-Z])(.*$)', r'\2', image, 0, re.I)
        self.patient = re.sub('(^[A-Z]+[0-9]+)(.*$)', r'\1', image, 0, re.I)
        self.array = array
        Mammogram.no_mammograms += 1


def feature_extraction(show=False):
    start = time.time()
    global features
    targets = np.empty((0, 2), float)
    old_file = "Cam006 - L - CC"
    f = []
    images_per_patient = 0
    features_selection = np.empty((0, 22), float)
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
                        mamgram = Mammogram(str(image), severity, type, array)

                if mamgram.type != "MicroCalc":
                    if show:
                        print('\n'+item)
                        print(mamgram.image)
                        print(mamgram.type)
                        print(mamgram.severity)
                        print(mamgram.area)
                        print(mamgram.side)
                        print(mamgram.patient)
                        print(str(Mammogram.no_mammograms))
                    target_features = [mamgram.patient, mamgram.type_number]
                    targets = np.vstack((target_features, targets))
                    # avg = np.mean(array)
                    # intensity = avg / 255.00
                    a = np.array(pp.image_statistics(array))
                    # a = np.append(a, avg)
                    b = a[2:]
                    # a = np.append(a, intensity)
                    PATCH_SIZE = 21;
                    ycord = int(round(a[0]))
                    xcord = int(round(a[1]))
                    features = []
                    patches = array[ycord:ycord + PATCH_SIZE, xcord:xcord + PATCH_SIZE]
                    # patches = pp.glcm_prep(array, 21)
                    cs = []
                    ds = []
                    hs = []
                    es = []
                    # for patch in patches:

                    # pl.hist(array.flatten(), 128)
                    # pl.title(str(image))
                    # # pl.savefig(str(image)+'_hist.png')
                    # fig = pl.gcf()
                    # fig.canvas.set_window_title(str(image)+'-'+severity+'_hist.png')
                    # pl.show()
                    mean = np.mean(array)
                    pp.locate_contours(array, mean, str(image)+' -- '+severity+' -- '+str(int(round(mean))))
                    # print(np.min(patches))
                    # print(np.mean(patches))
                    glcm = greycomatrix(patches, [1, 5, 10], [0, np.pi / 4, np.pi / 2, (3 * np.pi) / 4], 256,
                                        symmetric=True, normed=True)
                    ds.append(greycoprops(glcm, 'dissimilarity'))#[0, 0])
                    cs.append(greycoprops(glcm, 'contrast'))#[0, 0])
                    hs.append(greycoprops(glcm, 'homogeneity'))#[0, 0])
                    es.append(greycoprops(glcm, 'energy'))#[0, 0])
                    # for i in ds:
                    #     for j in i:
                    #         for k in j:
                    #             b = np.append(b, k)
                    # for i in cs:
                    #     for j in i:
                    #         for k in j:
                    #             b = np.append(b, k)
                    # for i in hs:
                    #     for j in i:
                    #         for k in j:
                    #             b = np.append(b, k)
                    # for i in es:
                    #     for j in i:
                    #         for k in j:
                    #             b = np.append(b, k)
                    # print(np.mean(es, axis=0))
                    b = np.append(b, np.std(ds, axis=1))
                    b = np.append(b, np.std(cs, axis=1))
                    b = np.append(b, np.std(hs, axis=1))
                    b = np.append(b, np.std(es, axis=1))
                    for feature in b:
                        features.append(feature)
                    # features.append(avg)
                    # features.append(intensity)
                    features_selection = np.vstack((features, features_selection))

    # print(time.time() - start)
    return features_selection, targets


X, Y = feature_extraction()
print(X)
print(Y)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y[:, 1], test_size=0.3)
# clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.1)
# clf.fit(X_train, Y_train)
#
# print("Training set score: %f" % clf.score(X_train, Y_train))
# print("Test set score: %f" % clf.score(X_test, Y_test))

params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

for label, param in zip(labels, params):
        print("training: %s" % label)
        clf = MLPClassifier(verbose=0, random_state=0,
                            max_iter=400, **param)
        clf.fit(X_test, Y_test)
        print("Training set score: %f" % (clf.score(X_train, Y_train)*100)+'%')
        print("Training set loss: %f\n" % clf.loss_)
#