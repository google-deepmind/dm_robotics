#!/bin/bash

# Copyright 2020 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script will:
# 1. Optionally, create a virtualenv,
#   a. Whether or not a virtualenv is used is user overridable.
#   b. The python version used in the virtualenv is overridable.
# 2. Install packages dm_transformations depends on - this is overridable,
# 3. Run the dm_transformations tests - also overridable.


# User controlled flags:
create_virtualenv=1  # Set by --virtualenv, --novirtualenv
install_dependencies=1  # Set by --dependencies, --nodependencies
run_tests=1  # Set by --tests, --notests
virtualenv_python_version=""  # Set by --python_version <version>

# Get the directory this file is in:
MYDIR="$(dirname "$(realpath "$0")")"
SRC_DIR="$(dirname "$MYDIR")"  # some/path/dm_robotics/transformations
PACKAGE_DIR="$(dirname "$SRC_DIR")"  # some/path/dm_robotics
VENV_DIR="venv"

function main() {
  cd "$PACKAGE_DIR" || die "Could not cd to $PACKAGE_DIR"
  parse_flags "$@"
  ensure_python_is_executable
  install_cleanup
  maybe_create_virtualenv  # depending on flags.
  maybe_install_dependencies  # depending on flags.
  run_setup
  install
  maybe_run_tests  # depending on flags.
}

function parse_flags() {
  while (( "$#" )); do
    case "$1" in
      -h|-help|--help) print_usage; exit 0 ;;
      --dependencies) install_dependencies=1; shift ;;
      --nodependencies) install_dependencies=0; shift ;;
      --virtualenv) create_virtualenv=1; shift ;;
      --novirtualenv) create_virtualenv=0; shift ;;
      --tests) run_tests=1; shift ;;
      --notests) run_tests=0; shift ;;
      --python_version)
        if ! virtualenv_python_version="$(parse_arg "$1" "$2")"; then
          die "Failed to parse $1"
        else
          shift 2
        fi
         ;;
      -*|--*=) die "Error: Unknown flag $1" ;;
      *) die "Unexpected argument: $1" ;;
    esac
  done
}

function print_usage() {
  usage="Builds; installs; and runs the tests for; dm_transformations.

  Arguments:
    --[no]dependencies:  Default: on.
       Controls the installation of packages that dm_transformations depends on.

    --[no]virtualenv:  Default: on.
       Controls whether package installation is in a virtualenv or not,
       including dependencies.

    --[no]tests:  Default: on.
       Controls whether unit tests are run after installation.

    --python_version <version-name>:
        determines the python version for virtualenv.
        See virtualenv documentation for -p.  Example legal value: 'cpython3.6'
        Default is determined by which version of python is installed.
"
  echo "$usage"
}

function parse_arg() {
  if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
    echo "$2"
  else
    exit 1
  fi
}

function ensure_python_is_executable() {
  local python_bin=`which python3`
  if [ ! -x "$python_bin" ]; then
    die "python ($python_bin) not found, or not executable"
  fi
}

function install_cleanup() {
  trap cleanup EXIT
}

function cleanup() {
  if (( $create_virtualenv )); then
    info "Deactivating virtualenv"
    deactivate
    rm -rf "$PACKAGE_DIR/$VENV_DIR"
  fi

  # Remove by-products of the build process.
  rm -rf "build"
  rm -rf "dist"
  rm -rf "dm_transformations.egg-info"
}

function maybe_create_virtualenv() {
  if (( ! $create_virtualenv )); then
    return
  fi

  get_python_version_for_virtualenv  # sets $virtualenv_python_version

  if ! virtualenv -p "$virtualenv_python_version" "$VENV_DIR"; then
    die "virtualenv -p $virtualenv_python_version failed"
  fi

  info "Activating virtual environment"
  source "$VENV_DIR/bin/activate"
}

function get_python_version_for_virtualenv() {
  if [ "$virtualenv_python_version" == "" ]; then
    local python_version="$(get_python_version)"
    case "$python_version" in
      3.4|3.5|3.6|3.7)
        # These are the versions we can work with.
        virtualenv_python_version="cpython${python_version}}"
        ;;
      3.[0-3])
        # These versions are too old, optimistically try 3.4.
        warning "Python version $python_version is not supported"
        warning "Attempting to use version 3.4"
        # TODO(123456789): Python version override
        virtualenv_python_version="cpython3.4"
        ;;
      3.*)
        # These versions are too recent, optimistically try 3.7.
        warning "Python version $python_version is not supported"
        warning "Attempting to use version 3.7"
        # TODO(123456789): Python version override
        virtualenv_python_version="cpython3.7"
        ;;
      *)
        # For anything else, give up.
        die "Python $python_version not supported"
        ;;
    esac
  fi

  info "Using $virtualenv_python_version for vitrtualenv"
}

# Returns python3 version as a string "<major>.<minor>"
# E.g. python version 3.5.2 is reported as "3.5"
# E.g. python version 3.6.11 is reported as "3.6"
function get_python_version() {
  python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'
}

function maybe_install_dependencies() {
  if (( $install_dependencies != 1 )); then
    return
  fi

  if ! pip3 install --upgrade -r transformations/pip_dependencies.txt; then
    die "Installation of dependencies failed."
  fi
}

function run_setup() {
  (cd transformations && python setup.py bdist_wheel) || die "Package build failed"
}

function install() {
  (cd transformations && python setup.py install) || die "Package installation failed"
}

function maybe_run_tests() {
  if (( ! $run_tests )); then
    return
  fi
  python transformations/transformations_test.py || die "Tests failed"
}

function info() {
  echo -e "\033[32m${1}\e[0m"  # Green
}

function warning() {
  echo -e "\033[33m${1}\e[0m"  # Yellow
}

function die() {
  echo -e "\033[31m${1}\e[0m"  # Red
  exit 1
}


main "$@"
