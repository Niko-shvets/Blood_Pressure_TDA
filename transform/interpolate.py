import numpy as np

from transform.transformer import TimeSeriesTransformer
import scipy.interpolate as inter


class SplineInterpolate(TimeSeriesTransformer):

    def __init__(self, smooth: float,bound=2000,wind=1000):
        self.smooth = smooth
        self.bound=bound
        self.wind=wind
    def interpolate(self, data: np.ndarray):
        x = np.arange(data.shape[0])
        spl = inter.UnivariateSpline(x, data, k=3)
        spl.set_smoothing_factor(self.smooth)
        return spl(x)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        #bound length of 4 PQRST cycles, with frequancy = 450, about 2000
        #wind is length of 2 PQRST cycles, as a result approx. about 2 cycle in each part of window
        assert data.ndim == 1
        n_split = 1 if data.shape[0] <= self.bound else int(np.floor(data.shape[0] / self.wind))
        return np.concatenate([
          self.interpolate(part) for part in np.array_split(data, n_split)
        ])
