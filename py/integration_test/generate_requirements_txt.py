# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate a requirements.txt file with the artifacts in ../dist/.

This ensures tox/pip will test with these, rather than with some from
pypi. This currently assumes that ../dist only contains one version of
each dm_robotics library.
"""

import glob
import os
import pathlib


_CURRENT_FILE_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


if __name__ == '__main__':
  with open(_CURRENT_FILE_DIR / 'requirements.txt', 'w') as f:
    for artifact in glob.glob('../dist/*'):
      f.write(artifact + os.linesep)
