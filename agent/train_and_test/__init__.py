"""
The train test submodul contain functions to train, test and perform episodes for an agent on en environment.
"""
from .perform import perform
from .train import train
from .test import test
__all__ = ["perform", "train", "test"]