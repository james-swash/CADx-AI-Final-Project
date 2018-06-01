import preprocessing_lib as pp
import numpy as np
import pylab as pl
from skimage.feature import greycomatrix, greycoprops

import matplotlib.pyplot as plt

from scipy.stats import entropy

img = pp.image_converter(r'C:\Users\James_000\Documents\University\Third Year\BEng Final Project\MIAS Mini Database\mdb001.pgm')
img = img/255
# shape = img.shape[0]*img.shape[1]
# summation = np.sum((img - np.mean(img))/np.std(img)**3)
# value = summation/shape
# print(value)
# print(img)
probs = []
unique, counts = np.unique(img*255, return_counts=True)
unique = unique.astype(int)
print(unique.size)
# counts = np.unique(img, return_counts=True)
for i in range(0, counts.size):
    # print(value)
    # print(counts[value])
    probs.append(counts[i] / (img.shape[0] * img.shape[1]))
# print(entropy(probs, base=2))
print(probs)
probs = np.asarray(probs)
print(probs)
print(np.sum(probs ** 2))

# pixels = []
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         if image[i, j] > 0:
#             pixels.append((i, j))
#
# Lx = image.shape[1]
# Ly = image.shape[0]
# print(Lx, Ly)
# pixels = pl.array(pixels)
# # pl.plot(pixels[:,1], pixels[:,0], '.', ms=0.01)
# # pl.show()
# print(pixels.shape)
#
# # computing the fractal dimension
# # considering only scales in a logarithmic list
# scales = np.logspace(0.01, 10, num=10, endpoint=False, base=2)
# Ns = []
# # looping over several scales
# for scale in scales:
#     print("======= Scale :", scale)
#     # computing the histogram
#     try:
#         H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
#     except ValueError:  # raised if `y` is empty.
#         pass
#     Ns.append(np.sum(H > 0))
#
# # linear fit, polynomial of degree 1
# coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
#
# pl.plot(np.log(scales), np.log(Ns), 'o', mfc='none')
# pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
# pl.xlabel('log $\epsilon$')
# pl.ylabel('log N')
# # pl.savefig('sierpinski_dimension.pdf')
#
# print("The Hausdorff dimension is", -coeffs[0])  # the fractal dimension is the OPPOSITE of the fitting coefficient


# img = pp.contrast_stretching(img)

# img = pp.histogram_equalization(img)

# img = pp.adaptive_equalization(img)

# img = pp.global_equalization(img)

# img = pp.local_equalization(img)

# fig, (ax1) = plt.subplots(nrows=2, ncols=3, figsize=(8, 3),
#                               sharex=True, sharey=True)
# image = img_as_float(img)
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title(title, fontsize=20)
# fig.tight_layout()
# plt.show()
#
# for x in np.nditer(img):
# print(x)
# img[img >= 200] = 225
# img[(img < 200) & (img >= 150)] = 175
# img[150 > img >= 100] = 125
# img[100 > img >= 50] = 75
# img[(50 > img) & (img > 0)] = 25
#
# blah = pp.locate_contours(img, mean, "pgm test")
# print(blah)
# print(img.shape)
# PATCH_SIZE = 40
#
# tissue_location = [(0, 0)]
# tissue_patches = []
#
# list = pp.split_array(img, 41, 77)
# blah = list[14]
# print(blah)
#
# glcm = greycomatrix(img, [20], [0], 256)
#
# print(glcm)

# var = len(list)
# for i in range(0, var):
#     array = list[i]
#     for loc in tissue_location:
#         tissue_patches.append(array[loc[0]:loc[0] + PATCH_SIZE,
#                               loc[1]:loc[1] + PATCH_SIZE])
#     print(tissue_patches)
#     xs = []
#     ys = []
#     for patch in tissue_patches:
#         glcm = greycomatrix(patch, [20], [0], 256, symmetric=True, normed=True)
#         xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
#         ys.append(greycoprops(glcm, 'correlation')[0, 0])
#     fig = plt.figure(figsize=(8, 8))
#
#     ax = fig.add_subplot(3, 2, 1)
#     ax.imshow(array, cmap=plt.cm.gray, interpolation='nearest',
#               vmin=0, vmax=255)
#     for (y, x) in tissue_location:
#         ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
#     ax.set_xlabel('Original Image')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.axis('image')
#
#     ax = fig.add_subplot(3, 2, 2)
#     ax.plot(xs[:len(tissue_patches)], ys[:len(tissue_patches)], 'go',
#             label='Tissue')
#     ax.set_xlabel('GLCM Dissimilarity')
#     ax.set_ylabel('GLCM Correlation')
#     ax.legend()
#     fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
# plt.show()
    # print(array.shape)
# pp.show_function(img, 'Original')
