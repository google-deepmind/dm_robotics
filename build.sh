#!/bin/bash

# Fail on any error.
set -e

root=`pwd`

cmake_binary=${CMAKE_EXE:-cmake}
echo "Using cmake command '$cmake_binary'"

tox_binary=${TOX_EXE:-tox}
echo "Using tox command '$tox_binary'"

python_binary=${PYTHON_EXE:-python3}
echo "Using python command '$python_binary'"

# Determine what version of Python $python_binary is.
# The extra && and || mean this will not stop the script on failure.
# python_version will be numbers and dots, e.g. 3.8.2
python_version="$($python_binary --version | grep --only-matching '[0-9.]*' 2>&1)" && exit_status=$? || exit_status=$?

# Allow the python version to be overridden.
python_version=${PYTHON_VERION:-$python_version}

# Finally default python_version, but this should not be needed.
python_version=${python_version:-3.8}
echo "Using python version '$python_version'"

# Install tox, which we use to build the packages.
# The packages themselves do not depend on tox.
python3 -m pip install tox

echo "Recreating $root/cpp/build directory"
rm -rf "$root/cpp/build"
mkdir "$root/cpp/build"

# echo "Running cmake in $root/cpp/build"
# $cmake_binary .. "-DDMR_PYTHON_VERSION=$python_version"
# make -j 4

# Build the dm_robotics.controllers package wheel.
echo "Building controllers package (setup.py) from $root/cpp"
cd "$root/cpp"
$python_binary setup.py bdist_wheel  # Uses the CMAKE_EXE environment variable.
ls "$root/cpp/dist"/dm_robotics_controllers*.whl  # Check that the wheel was built.

if [[ -n "$DM_ROBOTICS_VERSION_SCRIPT" ]]; then
  # If this is set, we're building wheels for distribution, so convert to
  # manylinux wheel (from an x86_64 wheel).

  cd "$root/cpp/dist"
  python3 -m auditwheel repair --plat manylinux_2_27_x86_64 dm_robotics_controllers*.whl

  # Remove the x86_64 wheel, and replace with the manylinux version.
  rm dm_robotics_controllers*.whl
  mv wheelhouse/* .
  rm -rf wheelhouse
  cd "$root/cpp"
fi

# Copy the wheel to the tox distshare directory.
echo "Copying controllers package wheel file to Tox distshare folder"
rm -rf "$root/py/dist"
mkdir "$root/py/dist"
cp "$root/cpp/dist"/dm_robotics_controllers*.whl "$root/py/dist"

echo "Building python transformations package"
cd "$root/py/transformations"
$tox_binary

echo "Building python geometry package"
cd "$root/py/geometry"
$tox_binary

echo "Building python agentflow package"
cd "$root/py/agentflow"
$tox_binary

echo "Building python moma package"
cd "$root/py/moma"
$tox_binary

echo "Building python manipulation package"
cd "$root/py/manipulation"
$tox_binary
