# libraries
import time
import biosppy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import warnings
from tqdm import tqdm
from scipy import signal


#sliding window
from sliding.ecg_slider import ECGSlider
#statistic wasserstein distance
from statistic.wasserstein_distance import WassersteinDistance, WassersteinDistanceDeviation
from Bootstrap.wasserstein_bootstrap import wasserstein_computation
#transformation
from transform.indexed_transform import IndexedTransformer
from transform.interpolate import SplineInterpolate
from transform.pca import PCATransformer
from transform.scale import ScaleTransform
from transform.series_to_curve import CurveProjection, IndicesWindow
from transform.transformer import SequentialTransformer
from transform.triangle_pattern import TrianglePattern
#artificial data
from Data_loader.data_loader import dataloader
from Data_loader.Artificial_data import artificial_data


art = artificial_data(1,1,noise=0)
X = art.create_rr(150,1,noise=0,arryhtmia=False)

a=1
func= 2*np.sin(3*X)+a


smooth_transform = SequentialTransformer(
    ScaleTransform(0, 1),
    SplineInterpolate(0.02)
)
smooth_func=smooth_transform(func)
smooth_signal=smooth_transform(X)

plt.figure(figsize=(18,6))
plt.plot(smooth_func[:1000])
plt.plot(smooth_signal[:1000])
plt.show()

const=1.2
n = smooth_signal.shape[0]
new_indices = [int(i * const) for i in range(n)]
new_indices = list(filter(lambda x: x < n, new_indices))

func_new = smooth_func[new_indices]

plt.figure(figsize=(18,6))
plt.plot(smooth_signal[:1000])
plt.plot(func_new[:1000])
plt.show()

joined_data = np.concatenate(
     (smooth_signal[..., np.newaxis][:n//2], func_new[..., np.newaxis][0:n//2]),
     axis=1
)

projection_step=1
window_size = 50
curve_transform = SequentialTransformer(
        CurveProjection(
            window=IndicesWindow.range(size=window_size, step=1),
            step=projection_step
        ),
        PCATransformer(2)
)


cloud = curve_transform(joined_data)

print(cloud.shape)

plt.scatter(cloud[:, 0], cloud[:, 1])
plt.show()
