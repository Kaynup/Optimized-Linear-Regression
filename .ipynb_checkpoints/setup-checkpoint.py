from setuptools import setup, Extension
from Cython.Build import cythonize

module = Extension(
    name="OptLinearRegress.linear_regressor", 
    sources=["OptLinearRegress/linear_regressor.pyx"], 
    language="c++",
    extra_compile_args=["-std=c++17"],
)

setup(
    name="OptLinearRegress",
    version="1.0.0",
    packages=["OptLinearRegress"],
    ext_modules=cythonize(module, compiler_directives={"language_level": 3})
)