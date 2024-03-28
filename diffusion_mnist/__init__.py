"""
diffusion_mnist
===============

Provides:
    1. model_ddm.py: The model architecture for the Defading Diffusion Model (DDM).
    2. model_ddpm.py: The model architecture for the Denoising Diffusion Probabilistic Model (DDPM).
    3. engine.py: The training engines for DDPM and DDM.
    4. evaluate.py: Evaluation functions for the DDM and DDPM on MNIST dataset.

Dependencies:
-------------
- numpy
    The fundamental package for scientific computing with Python.
    - URL: https://numpy.org/
- torch
    An open source machine learning framework that accelerates the path from research prototyping to production deployment.
    - URL: https://pytorch.org/
- torchvision
    The image and video datasets and models for PyTorch.
    - URL: https://pytorch.org/vision/stable/index.html
- torchmetrics
    Metrics for PyTorch.
    - URL: https://torchmetrics.readthedocs.io/en/latest/
- tqdm
    A fast, extensible progress bar for loops and CLI.
    - URL: https://tqdm.github.io/
- typing
    Support for type hints.
    - URL: https://docs.python.org/3/library/typing.html
- scipy
    A Python-based ecosystem of open-source software for mathematics, science, and engineering.
    - URL: https://www.scipy.org/
- matplotlib
    A comprehensive library for creating static, animated, and interactive visualizations in Python.
    - URL: https://matplotlib.org/
- configparser
    The configuration file parser library.
    - URL: https://docs.python.org/3/library/configparser.html
- os
    The OS module in Python provides a way of using operating system dependent functionality.
    - URL: https://docs.python.org/3/library/os.html

Example:
--------
Usage:
    Import functions from submodules as follows:
    >>> from diffusion_mnist.model_ddpm import DDPM
    >>> from diffusion_mnist.engine import train_ddpm
    >>> from diffusion_mnist.evaluate import compute_fid

    Create an instance of the DDPM model:
    >>> model = DDPM()

    Train the DDPM model:
    >>> train_ddpm(model, train_loader, test_loader, device)

    Compute the Frechet Inception Distance (FID) for the generated samples:
    >>> fid = compute_fid(generated_samples, true_samples, device)

    For more information, please refer to the documentation of each submodule.

Note:
-----
This module was build for the coursework of "Application of Machine Learning."
For help, please use help() function in Python.
"""