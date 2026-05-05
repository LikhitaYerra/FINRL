"""
Spinning Up — vendored PyTorch/MPI fork for FinRL-DeepSeek.

Original Spinning Up install pins TF1, Gym 0.15, and Torch 1.3; this fork only
needs the PyTorch + MPI stack used by ``train_cppo_multi_signal.py``.
TensorFlow is optional (``spinup.utils.logx`` tolerates missing TF).
"""
from os.path import join
from setuptools import find_packages, setup
import sys

if sys.version_info < (3, 9):
    raise SystemExit("This vendored spinup requires Python 3.9 or newer.")

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name="spinup",
    version=__version__,
    packages=find_packages(include=["spinup", "spinup.*"]),
    description="Spinning Up in Deep RL — PyTorch/MPI fork for FinRL-DeepSeek.",
    author="Joshua Achiam (original OpenAI Spinning Up)",
    python_requires=">=3.9",
    install_requires=[
        "cloudpickle>=1.6",
        "gymnasium>=0.29.1",
        "joblib>=1.3",
        "matplotlib>=3.7",
        "mpi4py>=3.1",
        "numpy>=1.26,<2",
        "pandas>=2.0",
        "psutil>=5.9",
        "scipy>=1.10",
        "torch>=2.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "viz": ["seaborn>=0.12"],  # spinup.utils.plot
        "dev": ["pytest>=7.0", "ipython>=8.0"],
    },
)
