from skopt import gp_minimize
from skopt.utils import cook_estimator
from skopt.learning import GaussianProcessRegressor
import itertools
import pandas as pd
import numpy as np
import baryc
import scipy.optimize
from scipy.stats import norm


def skopt_result_to_dataframe(result):
    # Extract function values, parameters, and iteration numbers from the result object
    function_values = result.func_vals
    parameters = result.x_iters
    iterations = list(range(1, len(function_values) + 1))

    # Create a pandas DataFrame
    df = pd.DataFrame(
        parameters, columns=[f"param_{i}" for i in range(len(parameters[0]))]
    )
    df["function_value"] = function_values
    df["iteration"] = iterations

    return df


def single(kernel, ground, dims, seed=0, max_iter=20):
    simplex = [(0.0, 1.0)] * dims
    return gp_minimize(
        ground,
        simplex,
        base_estimator=cook_estimator(
            base_estimator=GaussianProcessRegressor(kernel=kernel), space=simplex
        ),
        acq_func="EI",
        # n_jobs=-1,
        n_initial_points=1,
        verbose=True,
        random_state=seed,
        n_calls=max_iter,
        # callback=skopt.callbacks.DeltaYStopper(0.01),
    )


from scipy.stats import norm


def expected_improvement(gpr, best):
    def ei(x):  # x is a parameter. Like x = [1.0, 2.0, 3.0]
        x = np.array(x).reshape(1, -1)
        y_mean, y_std = gpr.predict(x, return_std=True)
        y_std = y_std.reshape(-1, 1)
        if y_std == 0:
            return 0.0

        z = (best - y_mean) / y_std
        ei_value = (best - y_mean) * norm.cdf(z) + y_std * norm.pdf(z)
        return ei_value[0, 0]

    return ei


def single_simplex(kernel, ground, dims, seed=0, max_iter=20):
    bnds = [(0.0, 1.0)] * dims
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

    np.random.seed(seed)

    search_space = baryc.lin_space(dims, 10 - dims)  # auto density: dims in [3, 6]

    X = [np.random.uniform(low=0.0, high=1.0, size=(dims,))]  # start point
    Y = [ground(X[0])]
    gp = GaussianProcessRegressor(kernel=kernel, random_state=seed)

    for i in range(max_iter):
        gp.fit(X, Y)

        newX = None
        newY = None
        # find best point
        for x in search_space:
            ei = expected_improvement(gp, np.min(Y))
            res = scipy.optimize.minimize(
                lambda x: -ei(x),  # minimize -ei
                x,
                bounds=bnds,
                constraints=cons,
                method="SLSQP",
            )
            if res.success:
                if newX is None:
                    newX = res.x
                    newY = res.fun
                elif newY > res.fun:
                    newX = res.x
                    newY = res.fun
        newY = ground(newX)
        if newX is None or newY is None:
            raise Exception("No feasible point found")
        X.append(newX)
        Y.append(newY)
        print(f"{str(ground)} {str(kernel)} iteration {i}: {newX} -> {newY}")

    history = pd.DataFrame(
        columns=[f"param_{i}" for i in range(dims)]
        + [
            "function_value",
            "iteration",
        ]
    )
    history[[f"param_{i}" for i in range(dims)]] = np.array(X)
    history["function_value"] = Y
    history["iteration"] = list(range(len(Y)))
    return history


#
def min_and_max(benchmark_function, n_dims, step_size=0.1):
    # Create the search space
    search_space = [np.arange(0, 1 + step_size, step_size) for _ in range(n_dims)]

    # Perform coarse grid search
    coarse_min_value = float("inf")
    coarse_max_value = float("-inf")
    coarse_min_solution = None
    coarse_max_solution = None

    for point in itertools.product(*search_space):
        value = benchmark_function(point)

        if value < coarse_min_value:
            coarse_min_value = value
            coarse_min_solution = point

        if value > coarse_max_value:
            coarse_max_value = value
            coarse_max_solution = point

    # Use scipy.optimize.minimize to refine the search around the coarse grid points
    min_result = scipy.optimize.minimize(
        benchmark_function, coarse_min_solution, bounds=[(0, 1)] * n_dims
    )
    max_result = scipy.optimize.minimize(
        lambda x: -benchmark_function(x), coarse_max_solution, bounds=[(0, 1)] * n_dims
    )

    # Get refined minimum and maximum values and their corresponding points
    min_value = min_result.fun
    min_solution = min_result.x
    max_value = -max_result.fun
    max_solution = max_result.x

    return min_value, min_solution, max_value, max_solution


def min_and_max_simplex(benchmark_function, n_dims, density=10):
    # Create the search space
    search_space = baryc.lin_space(n_dims, density)

    # Perform coarse grid search
    coarse_min_value = float("inf")
    coarse_max_value = float("-inf")
    coarse_min_solution = None
    coarse_max_solution = None

    for point in search_space:
        value = benchmark_function(point)

        if value < coarse_min_value:
            coarse_min_value = value
            coarse_min_solution = point

        if value > coarse_max_value:
            coarse_max_value = value
            coarse_max_solution = point

    # Use scipy.optimize.minimize to refine the search around the coarse grid points
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    min_result = scipy.optimize.minimize(
        benchmark_function,
        coarse_min_solution,
        bounds=[(0, 1)] * n_dims,
        constraints=cons,
        method="SLSQP",
    )
    max_result = scipy.optimize.minimize(
        lambda x: -benchmark_function(x),
        coarse_max_solution,
        bounds=[(0, 1)] * n_dims,
        constraints=cons,
        method="SLSQP",
    )

    # Get refined minimum and maximum values and their corresponding points
    min_value = min_result.fun
    min_solution = min_result.x
    max_value = -max_result.fun
    max_solution = max_result.x

    return min_value, min_solution, max_value, max_solution
