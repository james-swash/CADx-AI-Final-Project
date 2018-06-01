import preprocessing_lib as pp

import numpy as np
from scipy.fftpack import fft, ifft
from scipy import stats
import matplotlib.pyplot as plt
array = pp.image_converter(r'C:\Users\James_000\Documents\University\Third Year\BEng Final '
                           r'Project\Mammograms\Cam006-L-CC.bmp_FCanavan_MarkedRegion_1_Output.bmp')
# print(stats.describe(array))
features = []
print(pp.image_statistics(array)[2:])
a = np.array(pp.image_statistics(array)[2:])
for item in a:
    features.append(item)

print(features)