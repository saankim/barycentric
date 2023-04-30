import numpy as np
from olympus import Emulator


class RandomNoise:
    def __init__(self, func, noise_level=0.0):
        self.noise_level = noise_level
        self.func = func

    def __call__(self, xx):
        if self.func is None:
            raise NotImplementedError
        return self.func(xx) + np.random.normal(0, self.noise_level)


#
class Hartmann3D:
    name = "Hartmann3D"
    dims = 3

    def __call__(self, xx):
        """
        This function evaluates the Hartmann 3D function at x.
        The Hartmann 3D function is commonly used to test optimization algorithms.
        It has four local minima, where the global minimum is at x = [0.114614, 0.555649, 0.852547].
        The global minimum value is -3.86278.
        https://www.sfu.ca/~ssurjano/hart3.html
        """
        alpha = np.asfarray([1.0, 1.2, 3.0, 3.2])
        A = np.asfarray([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = 1e-4 * np.asfarray(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )
        xxmat = np.tile(xx, 4).reshape(4, 3)
        inner = np.sum(A * (xxmat - P) ** 2, axis=1)
        outer = np.sum(alpha.T * np.exp(-inner))
        return -outer


#
class Hartmann4D:
    name = "Hartmann4D"
    dims = 4

    def __call__(self, xx):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.asfarray(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        xxmat = np.tile(xx, (4, 1))
        inner = np.sum(A[:, 1:5] * (xxmat - P[:, 1:5]) ** 2, axis=1)
        outer = np.sum(alpha.T * np.exp(-inner))
        y = (1.1 - outer) / 0.839
        return y


#
class Hartmann6D:
    name = "Hartmann6D"
    dims = 6

    def __call__(self, xx):
        alpha = np.asfarray([1.0, 1.2, 3.0, 3.2])
        A = np.asfarray(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.asfarray(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        xxmat = np.reshape(np.repeat(xx, 4), (4, 6), order="F")
        inner = np.sum(A[:, 0:6] * (xxmat - P[:, 0:6]) ** 2, axis=1)
        outer = np.sum(alpha.T * np.exp(-inner))
        y = -outer
        return y


#
_emul_cn9 = Emulator(dataset="colors_n9", model="NeuralNet")


def EmulCN9(x):
    rtn = _emul_cn9.run(np.array(x).flatten())
    return np.array(rtn).flatten()[0]


_emul_pce10 = Emulator(dataset="photo_pce10", model="NeuralNet")


def EmulPCE10(x):
    rtn = _emul_pce10.run(np.array(x).flatten())
    return np.array(rtn).flatten()[0]


_emul_bob = Emulator(dataset="colors_bob", model="NeuralNet")


def EmulBOB(x):
    rtn = _emul_bob.run(np.array(x).flatten())
    return np.array(rtn).flatten()[0]
