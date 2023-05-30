# Barycentric Kernel for Bayesian Optimization of Chemical Mixture

This repository implements the Barycentric kernel. Also, this repository includes some Python scripts to conduct comparison and emulated experiments with several kernels including radial basis function kernel, Matern kernel and Laplacian kernel.

You can cite this paper with the Bibtex below.

```bibtex
@article{kim2023barycentric,
  title={Barycentric Kernel for Bayesian Optimization of Chemical Mixture},
  author={Kim, San and Kim, Jaekwang},
  journal={Electronics},
  volume={12},
  number={9},
  pages={2076},
  year={2023},
  publisher={MDPI}
}
```

## Python scripts

-   `baryc.py`
    -  `np.linspace`-like linear space filler for a simplex.
    -  Barycentric metric
    -  Barycentric kernel
- `batch.py`
    - Experimental hyperparameters and runnable Python script
- `exp.py`
    - Functions to help to conduct experiments
    - Functions to wrap the experimental details up
-   `ground.py`
    -   ground truth functions
    - Hartmaan 3D, 4D, 6D benchmark functions
    - Three chemical experiment emulators
- `laplacian.py`
    - Implements laplacian kernel in the Scikit-learn manner
- `util.py`
    - Utility functions to store and load data dumps

## Data pickles

The result data is stored in the `/data/dump` directory with this rule.

`{simplex_}{ground}_{kernel}_{seed}/{timestamp}.pkl`
- `simplex`: This is added if the experiment conducted on a simplex
- `ground`: The name of the ground truth function.
- `kernel`: The name of the kernel function.
- `seed`: The number indicates the random seed.
- `timestamp`: The timestamp indicates when the experiment was conducted.
