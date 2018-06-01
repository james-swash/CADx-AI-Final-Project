#!/usr/bin/python3
# Filename: preprocessing_lib.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, io, exposure, measure
from skimage.morphology import disk
from skimage.filters import rank

matplotlib.rcParams['font.size'] = 8


def image_converter(file):
    img = io.imread(
        file)  # r'C:\Users\James_000\Documents\University\Third Year\BEng Final Project\Mammograms/' + file)
    return img


def contrast_stretching(img):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale


def histogram_equalization(img):
    img_eq = exposure.equalize_hist(img)
    return img_eq


def adaptive_equalization(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


def global_equalization(img):
    global_img = exposure.equalize_hist(img)
    return global_img


def local_equalization(img):
    selem = disk(30)
    local = rank.equalize(img, selem=selem)
    return local


def locate_contours(img, pixel_value, title):
    try:
        contours = measure.find_contours(img, pixel_value)
        max_len = max([len(i) for i in contours])
        # fig, ax = plt.subplots()
        # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            if len(contour) == max_len:
                edge = contour
                # print(contour[:, 1], contour[:, 0])
                # ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        # ax.axis('image')
        # ax.set_title(title)
        # fig.canvas.set_window_title(title)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.show()
        return edge
    except:
        return []



def show_function(img, title):
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 3),
                              sharex=True, sharey=True)
    image = img_as_float(img)
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title(title, fontsize=20)
    fig.tight_layout()
    plt.show()


def split_array(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape

    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def fractal_dimension(Z, threshold=0.9):
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def image_statistics(Z):
    # Input: Z, a 2D array, hopefully containing some sort of peak
    # Output: cx,cy,sx,sy,skx,sky,kx,ky
    # cx and cy are the coordinates of the centroid
    # sx and sy are the stardard deviation in the x and y directions
    # skx and sky are the skewness in the x and y directions
    # kx and ky are the Kurtosis in the x and y directions
    # Note: this is not the excess kurtosis. For a normal distribution
    # you expect the kurtosis will be 3.0. Just subtract 3 to get the
    # excess kurtosis.

    h, w = np.shape(Z)

    x = range(w)
    y = range(h)

    # calculate projections along the x and y axes
    yp = np.sum(Z, axis=1)
    xp = np.sum(Z, axis=0)

    # centroid
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)

    # standard deviation
    x2 = (x - cx) ** 2
    y2 = (y - cy) ** 2

    sx = np.sqrt(np.sum(x2 * xp) / np.sum(xp))
    sy = np.sqrt(np.sum(y2 * yp) / np.sum(yp))

    # skewness
    x3 = (x - cx) ** 3
    y3 = (y - cy) ** 3

    skx = np.sum(xp * x3) / (np.sum(xp) * sx ** 3)
    sky = np.sum(yp * y3) / (np.sum(yp) * sy ** 3)

    # Kurtosis
    x4 = (x - cx) ** 4
    y4 = (y - cy) ** 4
    kx = np.sum(xp * x4) / (np.sum(xp) * sx ** 4)
    ky = np.sum(yp * y4) / (np.sum(yp) * sy ** 4)

    return cx, cy, sx, sy, skx, sky, kx, ky


# We can check that the result is the same if we use the full 2D data array
def image_statistics_2D(Z):
    h, w = np.shape(Z)

    x = range(w)
    y = range(h)

    X, Y = np.meshgrid(x, y)

    # Centroid (mean)
    cx = np.sum(Z * X) / np.sum(Z)
    cy = np.sum(Z * Y) / np.sum(Z)

    ###Standard deviation
    x2 = (range(w) - cx) ** 2
    y2 = (range(h) - cy) ** 2

    X2, Y2 = np.meshgrid(x2, y2)

    # Find the variance
    vx = np.sum(Z * X2) / np.sum(Z)
    vy = np.sum(Z * Y2) / np.sum(Z)

    # SD is the sqrt of the variance
    sx, sy = np.sqrt(vx), np.sqrt(vy)

    ###Skewness
    x3 = (range(w) - cx) ** 3
    y3 = (range(h) - cy) ** 3

    X3, Y3 = np.meshgrid(x3, y3)

    # Find the thid central moment
    m3x = np.sum(Z * X3) / np.sum(Z)
    m3y = np.sum(Z * Y3) / np.sum(Z)

    # Skewness is the third central moment divided by SD cubed
    skx = m3x / sx ** 3
    sky = m3y / sy ** 3

    ###Kurtosis
    x4 = (range(w) - cx) ** 4
    y4 = (range(h) - cy) ** 4

    X4, Y4 = np.meshgrid(x4, y4)

    # Find the fourth central moment
    m4x = np.sum(Z * X4) / np.sum(Z)
    m4y = np.sum(Z * Y4) / np.sum(Z)

    # Kurtosis is the fourth central moment divided by SD to the fourth power
    kx = m4x / sx ** 4
    ky = m4y / sy ** 4

    return cx, cy, sx, sy, skx, sky, kx, ky


def glcm_prep(array, patchsize=5):
    """
    Function to split image up into sub-arrays and
    return a list of those sub-arrays for glcm processing
    :param array: A 2D ndarray of the original image
    :param patchsize: The n x n size of the sub-arrays
    :return: all n x n sub-arrays created in a list form
    """
    PATCH_SIZE = patchsize
    xcord = []
    ycord = []
    locations = []
    patches = []
    h, w = array.shape
    for x in range(0, h + 1, PATCH_SIZE):
        ycord.append(x)
    for y in range(0, w + 1, PATCH_SIZE):
        xcord.append(y)
    for c in ycord:
        for v in xcord:
            tup = (c, v)
            locations.append(tup)
    for loc in locations:
        sub_array = array[loc[0]:loc[0] + PATCH_SIZE,
                    loc[1]:loc[1] + PATCH_SIZE]
        if sub_array.shape == (PATCH_SIZE, PATCH_SIZE):
            patches.append(sub_array)

    return patches
