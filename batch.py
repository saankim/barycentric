import exp
import itertools
import util
from baryc import BarycentricKernel, basis
import skopt.learning.gaussian_process.kernels as ks
import os
import ground as g
import exp
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings
import laplacian

warnings.filterwarnings("ignore")


def single_wrapper(kernel_name, ground_name, seed, max_iter=30):
    kernel = ""
    ground = ""
    dims = 0
    is_simplex = False
    match ground_name:
        case "Hartmann3D":
            ground = g.Hartmann3D()
            dims = 3
        case "Hartmann4D":
            ground = g.Hartmann4D()
            dims = 4
        case "Hartmann6D":
            ground = g.Hartmann6D()
            dims = 6
        case "EmulCN9":
            ground = g.EmulCN9
            dims = 3
            is_simplex = True
        case "EmulPCE10":
            ground = g.EmulPCE10
            dims = 4
            is_simplex = True
        case "EmulBOB":
            ground = g.EmulBOB
            dims = 5
            is_simplex = True
    match kernel_name:
        case "BarycentricBasis":
            kernel = BarycentricKernel(simplex=basis(dims))
        case "RadialBasis":
            kernel = ks.RBF()
        case "Matern":
            kernel = ks.Matern()
        case "Constant":
            kernel = ks.ConstantKernel()
        case "RationalQuadratic":
            kernel = ks.RationalQuadratic()
        case "WhiteKernel":
            kernel = ks.WhiteKernel()
        case "DotProduct":
            kernel = ks.DotProduct()
        case "Laplacian":
            kernel = laplacian.LaplacianKernel()
    is_simplex = True  # for simplex experiments
    match is_simplex:
        case False:
            result = exp.single(
                kernel=kernel, ground=ground, dims=dims, seed=seed, max_iter=max_iter
            )
            result = dict(
                ground=ground_name,
                kernel=kernel_name,
                dims=dims,
                result=exp.skopt_result_to_dataframe(result),
            )
            util.Dump(result, f"{ground_name}_{kernel_name}_{seed}")
        case True:
            result = dict(
                ground=ground_name,
                kernel=kernel_name,
                dims=dims,
                result=exp.single_simplex(
                    kernel=kernel,
                    ground=ground,
                    dims=dims,
                    seed=seed,
                    max_iter=max_iter,
                ),
            )
            util.Dump(result, f"simplex_{ground_name}_{kernel_name}_{seed}")
    return


#
kernels = [
    "BarycentricBasis",  # My kernel
    "RadialBasis",  # Popular
    "Matern",  # Popular
    "Constant",  # Baseline
    "WhiteKernel",  # Baseline
    "DotProduct",  # Basic kernel
    "Laplacian",  # Basic kernel
]
grounds = [
    "Hartmann3D",  # Benchmark function, well known
    "Hartmann4D",  # Benchmark function, well known
    "Hartmann6D",  # Benchmark function, well known
    "EmulCN9",  # Chemical reaction dataset
    "EmulPCE10",  # Chemical reaction dataset
    "EmulBOB",  # Chemical reaction dataset
]
seeds = list(range(40))
workers = os.cpu_count() - 2


# single_wrapper was wrapped by batch to be used in multiprocessing or further features such as callback for every single experiment
def batch(arg):
    single_wrapper(*arg)


# To run a single experiment
# single_wrapper(
#     kernel_name="Laplacian",
#     ground_name="Hartmann3D",
#     seed=13,
# )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        args = list(itertools.product(kernels, grounds, seeds))
        results = list(executor.map(batch, args))
