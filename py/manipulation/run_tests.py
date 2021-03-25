# Copyright 2020 DeepMind Technologies Limited.
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


def test_file_paths(top_dir):
  """Yields the path to the test files in the given directory."""

  for dirpath, dirnames, filenames in os.walk(top_dir):
    # do not search tox or other hidden directories:
    remove_indexes = [
        i for i, name in enumerate(dirnames)
        if "tox" in name or name.startswith(".")
    ]
    for index in reversed(remove_indexes):
      del dirnames[index]

    for filename in filenames:
      if filename.endswith("test.py"):
        yield os.path.join(dirpath, filename)


def run_test(file_path):
  return subprocess.call([sys.executable, file_path]) == 0


if __name__ == "__main__":
  dir_to_search = sys.argv[1]
  success = True
  for test_path in test_file_paths(dir_to_search):
    success &= run_test(test_path)
  sys.exit(0 if success else 1)
