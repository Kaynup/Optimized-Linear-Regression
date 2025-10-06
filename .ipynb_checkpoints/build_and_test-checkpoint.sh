#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail

echo "=== Step 1: Cleaning previous builds ==="
rm -rf build/ dist/ *.egg-info

echo "=== Step 2: Building Cython extensions ==="
python setup.py build_ext --inplace

echo "=== Step 3: Installing package locally ==="
pip install -e .

echo "=== Step 4: Running tests ==="
pytest tests --maxfail=1 --disable-warnings -v

echo "=== All done! ==="
