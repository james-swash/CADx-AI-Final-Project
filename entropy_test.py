import matplotlib.pyplot as plt
import numpy as np
import preprocessing_lib as pp
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

# First example: object detection.

noise_mask = 28 * np.ones((128, 128), dtype=np.uint8)
noise_mask[32:-32, 32:-32] = 30

noise = (noise_mask * np.random.random(noise_mask.shape) - 0.5 *
         noise_mask).astype(np.uint8)
img = noise + 128

entr_img = entropy(img, disk(10))

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

ax0.imshow(noise_mask, cmap='gray')
ax0.set_xlabel("Noise mask")
ax1.imshow(img, cmap='gray')
ax1.set_xlabel("Noisy image")
ax2.imshow(entr_img, cmap='viridis')
ax2.set_xlabel("Local entropy")

fig.tight_layout()

# Second example: texture detection.

image = pp.image_converter(r'C:\Users\James_000\Documents\University\Third Year\BEng Final '
                           r'Project\Mammograms\Cam006-L-CC.bmp_FCanavan_MarkedRegion_1_Output.bmp')

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                               sharex=True, sharey=True)

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title("Image")
ax0.axis("off")
fig.colorbar(img0, ax=ax0)

print(np.std(entropy(image, disk(5))))
print(np.std(entropy(image, disk(5)), axis=0))
print(np.std(entropy(image, disk(5)), axis=1))
print(np.mean(entropy(image, disk(5))))
print(np.mean(entropy(image, disk(5)), axis=0))
print(np.mean(entropy(image, disk(5)), axis=1))

img1 = ax1.imshow(entropy(image, disk(5)), cmap='gray')
ax1.set_title("Entropy")
ax1.axis("off")
fig.colorbar(img1, ax=ax1)

fig.tight_layout()

plt.show()