# Copyright 2020 DeepMind Technologies Limited.
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

"""An empty space."""

import os
from typing import Optional
from dm_control import composer
from dm_control import mjcf

RESOURCES_ROOT_DIR = os.path.dirname(__file__)  # arenas
_EMPTY_CELL_XML_PATH = os.path.join(RESOURCES_ROOT_DIR,
                                    'empty_assets/arena.xml')


class Arena(composer.Arena):
  """An empty arena with a ground plane and camera."""

  def _build(self, name: Optional[str] = None):
    """Initializes this arena.

    Args:
      name: (optional) A string, the name of this arena. If `None`, use the
        model name defined in the MJCF file.
    """
    super()._build(name)
    self._mjcf_root.include_copy(
        mjcf.from_path(_EMPTY_CELL_XML_PATH), override_attributes=True)
    self._ground = self._mjcf_root.find('geom', 'ground')

  @property
  def ground(self):
    """The ground plane mjcf element."""
    return self._ground

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to this arena."""
    return self._mjcf_root
