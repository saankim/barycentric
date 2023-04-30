import numpy as np
import sklearn.gaussian_process.kernels as kernels


# Example usage: perpendicular_vectors(3) returns [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
def basis(n):
    # Initialize the array of basis vectors
    b = np.zeros((n, n))
    # Set each basis vector to a unit vector along one dimension
    for i in range(n):
        b[i, i] = 1.0
    # Compute the Gram-Schmidt process to orthonormalize the basis vectors
    for i in range(n):
        for j in range(i):
            b[i] -= np.dot(b[i], b[j]) * b[j]
        b[i] /= np.linalg.norm(b[i])
    return b


def lin_space(dimensions=3, density=1000):
    ans = []
    X = np.linspace(0, 1, density)

    def helper(d, arr):
        if d == 0:
            z = 1 - sum(arr, 0)
            if z < 0:
                return
            ans.append(arr + [z])
            return
        for x in X:
            helper(d - 1, arr + [x])

    helper(dimensions - 1, [])
    return np.array(ans, dtype=np.float64)


def side_of_simplex(basis: np.ndarray) -> np.ndarray:
    def dist(a, b):
        return np.linalg.norm(a - b)

    M = np.zeros((len(basis), len(basis)))
    for i in range(len(basis)):
        for j in range(len(basis)):
            M[i, j] = dist(basis[i], basis[j])
    return M


def distance(x=[0.33, 0.33, 0.34], y=[], b=basis(3)):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    diff = x - y
    M = side_of_simplex(b)
    B = M - np.sum(M, axis=0) - np.sum(M, axis=1)
    rtn = -0.5 * np.dot(diff, np.dot(B, diff))
    if rtn < 0:
        raise ValueError(f"distance is negative, {rtn}, x={x}, y={y}, b={b}")
    return rtn


# You can add length_scale and length_scale_bounds to the constructor if you want to use this as a kernel with automated hyperparameter optimization in scikit-learn
class BarycentricKernel(
    kernels.StationaryKernelMixin, kernels.NormalizedKernelMixin, kernels.Kernel
):
    """exporenticial kernel for barycentric coordinates"""

    def __init__(self, simplex=basis(3)):
        """exponential is a function recieving a scalar distance and returning a scalar"""
        if simplex is int:
            simplex = basis(simplex)
        # length scale is not yet implemented for this kernel
        # you can make a pull request if you want to add it
        # self.length_scale = length_scale
        # self.length_scale_bounds = length_scale_bounds
        self.simplex = simplex
        self.exponential = np.exp

    def _f(self, xx1, xx2):
        """
        kernel value between a pair of coordinates
        """
        dist = distance(xx1, xx2, b=self.simplex)  # / self.length_scale
        return self.exponential(-dist)

    def _g(self, xx1, xx2):
        """
        kernel derivative between a pair of barycentric coordinates
        """
        dist = distance(xx1, xx2, b=self.simplex)
        return self.exponential(-dist) * -dist  # / self.length_scale

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        if eval_gradient:
            return (
                np.array([[self._f(x, y) for y in Y] for x in X]),
                np.array([[[self._g(x, y)] for y in Y] for x in X]),
            )
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def gradient_x(self, x, X_train):
        return np.array(
            list(map(lambda x2: np.ones(len(self.simplex)) * self._g(x, x2), X_train))
        )
