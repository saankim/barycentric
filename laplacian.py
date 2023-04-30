import numpy as np
from sklearn.gaussian_process import kernels


class LaplacianKernel(
    kernels.StationaryKernelMixin, kernels.NormalizedKernelMixin, kernels.Kernel
):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    # @property
    # def hyperparameter_length_scale(self):
    #     return kernels.Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def _l1_distance(self, x, y):
        return np.sum(np.abs(x - y))

    def _kernel_value(self, x, y):
        return np.exp(-self._l1_distance(x, y) / self.length_scale)

    def _kernel_gradient(self, x, y):
        return (
            -np.exp(-self._l1_distance(x, y) / self.length_scale)
            * np.sign(x - y)
            / self.length_scale
        )

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        if eval_gradient:
            return (
                np.array([[self._kernel_value(x, y) for y in Y] for x in X]),
                np.array([[[self._kernel_gradient(x, y)] for y in Y] for x in X]),
            )
        else:
            return np.array([[self._kernel_value(x, y) for y in Y] for x in X])

    def gradient_x(self, x, X_train):
        return np.array([self._kernel_gradient(x, x2) for x2 in X_train])
