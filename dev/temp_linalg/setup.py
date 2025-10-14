from setuptools import setup
from Cython.Build import cythonize
import os

setup(
    name="linalg",
    packages=["linalg"],
    ext_modules=cythonize(
        "linalg/linalg.pyx",
        annotate=True,        # generates linalg.html for Cython analysis
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
