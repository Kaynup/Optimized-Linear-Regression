from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os

def find_pyx_files(package_dir):
    pyx_files = []
    for root, _, files in os.walk(package_dir):
        if ".ipynb_checkpoints" in root:
            continue
        for file in files:
            if file.endswith(".pyx"):
                full_path = os.path.join(root, file)
                module_name = full_path.replace(os.path.sep, ".").replace(".pyx", "")
                pyx_files.append((module_name, full_path))
    return pyx_files

extensions = [
    Extension(
        name=mod_name,
        sources=[source],
        language="c++",
        extra_compile_args=["-std=c++17"]
    )
    for mod_name, source in find_pyx_files("OptLinearRegress")
]

setup(
    name="OptLinearRegress",
    version="2.0.0",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions, 
        compiler_directives={"language_level": 3}
    ),
)
