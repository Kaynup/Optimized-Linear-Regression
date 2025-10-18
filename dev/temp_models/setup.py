from setuptools import setup
from Cython.Build import cythonize
import os

setup(
    name="models",
    packages=["models"],
    ext_modules=cythonize(
        "linalg/linalg.pyx",
        annotate=True,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)