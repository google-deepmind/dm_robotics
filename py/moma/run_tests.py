# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A manual test runner that runs each test file in a separate process.

This is needed because there is global state in spec_utils used to switch off
validation after some time (for performance on real robots).  This automatic
switch-off causes the validation test to fail because validation is switched
off when it comes to run, unless each test starts in a new process.
"""


import os
import subprocess
import sys


_MODULE = "dm_robotics.moma"
_EXCLUDED_PATHS = ["build", "./build", ".tox", "./.tox", "venv", "./venv"]


def test_file_paths(top_dir):
  """Yields the path to the test files in the given directory."""

  def excluded_path(name):
    return any(name.startswith(path) for path in _EXCLUDED_PATHS)

  for dirpath, dirnames, filenames in os.walk(top_dir):
    # do not search tox or other hidden directories:
    remove_indexes = [
        i for i, name in enumerate(dirnames) if excluded_path(name)
    ]
    for index in reversed(remove_indexes):
      del dirnames[index]

    for filename in filenames:
      if filename.endswith("test.py"):
        yield os.path.join(dirpath, filename)


def module_name_from_file_path(pathname):
  # dirname will be like: "./file.py", "./dir/file.py" or "./dir1/dir2/file.py"
  # convert this to a module.name:
  submodule_name = pathname.replace("./", "").replace("/", ".")[0:-3]
  return _MODULE + "." + submodule_name


def run_test(test_module_name):
  return subprocess.call([sys.executable, "-m", test_module_name]) == 0


if __name__ == "__main__":
  dir_to_search = sys.argv[1]
  success = True
  for test_path in test_file_paths(dir_to_search):
    module_name = module_name_from_file_path(test_path)
    success &= run_test(module_name)
  sys.exit(0 if success else 1)
