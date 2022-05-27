#!/bin/bash

# Fail on any error.
set -e

root=`pwd`

if [[ -z "$MJLIB_PATH" ]]; then
  # If MJLIB_PATH was not set, attempt to locate mujoco.so.
  # This grep aims to avoid nogl versions of the MuJoCo libraru.
  MJLIB_PATH=$(find $HOME/.mujoco/ -xtype f -name "*mujoco*.so" | grep "libmujoco[[:digit:]]*.so")
  if [[ ! $? ]]; then
    echo "Failed to find mujoco shared library (.so file)."
    echo "Please set MJLIB_PATH to the location of the mujoco .so file."
    exit 1
  fi
fi

if [[ ! -r "$MJLIB_PATH" ]]; then
  echo "Cannot read the mujoco library at ${MJLIB_PATH}"
  echo "Set the MJLIB_PATH env var to change this location"
  exit  1
fi

echo "MJLIB_PATH: ${MJLIB_PATH}"
export MJLIB_PATH

cmake_binary=${CMAKE_EXE:-cmake}
echo "Using cmake command '$cmake_binary'"

python_binary=${PYTHON_EXE:-python3}
echo "Using python command '$python_binary'"

tox_binary=${TOX_EXE:-$python_binary -m tox}
echo "Using tox command '$tox_binary'"

# Determine what version of Python $python_binary is.
# The extra && and || mean this will not stop the script on failure.
# python_version will be numbers and dots, e.g. 3.8.2
python_version="$($python_binary --version | grep --only-matching '[0-9.]*' 2>&1)" && exit_status=$? || exit_status=$?

# Allow the python version to be overridden.
python_version=${PYTHON_VERION:-$python_version}

# Finally default python_version, but this should not be needed.
python_version=${pytagentflowhon_version:-3.8}
echo "Using python version '$python_version'"

# Install tox, which we use to build the packages.
# The packages themselves do not depend on tox.
if ! [[ -x $tox_binary ]]; then
  $PYTHON_EXE -m pip install tox
fi

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

if which auditwheel; then
  pushd "$root/cpp/dist"
  echo "Tagging wheels as manylinux_2_27_x86_64"
  for file in $(ls dm_robotics_controllers*.whl); do
    auditwheel repair --plat manylinux_2_27_x86_64 "$file"
  done

  # Remove the x86_64 wheel, and replace with the manylinux version.
  rm dm_robotics_controllers*.whl
  mv wheelhouse/* .
  rm -rf wheelhouse
  popd
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

echo "Running integration tests"
cd "$root/py/integration_test"
$tox_binary
