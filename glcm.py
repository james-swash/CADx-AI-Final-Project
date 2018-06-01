import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import greycomatrix, greycoprops

import preprocessing_lib as pp


PATCH_SIZE = 21

# open the camera image
image = pp.image_converter(r'C:\Users\James_000\Documents\University\Third Year\BEng Final '
                           r'Project\Mammograms\Cam006-L-CC.bmp_FCanavan_MarkedRegion_1_Output.bmp')

# image = pp.contrast_stretching(image)

print(image.shape)

# select some patches from grassy areas of the image
tissue_locations = [(10, 280), (342, 254), (278, 14), (10, 34)]
tissue_patches = []
for loc in tissue_locations:
    tissue_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
cancer_locations = [(168, 120), (182, 193), (223, 130), (100, 123)]
cancer_patches = []
for loc in cancer_locations:
    cancer_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])
print(cancer_patches)

# compute some GLCM properties each patch
xs = []
ys = []

# print(tissue_patches)
# print(cancer_patches)
#
# print(tissue_patches + cancer_patches)

for patch in (tissue_patches + cancer_patches):
   # print(len(patch))
   glcm = greycomatrix(patch, [1,2,3,4,5,6,7,8,9,10], [0, np.pi/4, np.pi/2, (3*np.pi)/4], 256, symmetric=True, normed=True)
   print(greycoprops(glcm, 'contrast'), greycoprops(glcm, 'dissimilarity'))
   xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
   ys.append(greycoprops(glcm, 'contrast')[0, 0])

 # create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in tissue_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in cancer_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(tissue_patches)], ys[:len(tissue_patches)], 'go',
        label='Tissue')
ax.plot(xs[len(cancer_patches):], ys[len(cancer_patches):], 'bo',
        label='Cancer')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Contrast')
ax.legend()

# display the image patches
for i, patch in enumerate(tissue_patches):
    ax = fig.add_subplot(3, len(tissue_patches), len(tissue_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Tissue %d' % (i + 1))

for i, patch in enumerate(cancer_patches):
    ax = fig.add_subplot(3, len(cancer_patches), len(cancer_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Cancer %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
