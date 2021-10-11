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

r"""A class that defines an rgb_objects as an instance of parametric_object.

An rgb_object is a specific instance of a parametric object. An rgb_object
has 9 parameters named: 'sds', 'shr', 'scx', 'scy', 'scz', 'shx', 'shy', 'hlw',
'drf'.
The meaning of these parameters is the following:

  sds: Number of sides, 2 circle
  shr: Shrink the 2D shape before sxtruding
  drf: Draft pyramidizing [deg]
  hlw: % Hollow: 0 solid
  shx: Shear in X [deg]
  shy: Shear in Y [deg]
  scx: Scale in X
  scy: Scale in Y
  scz: Scale in Z

These parameters are constrained as follows and this guarantees that the
mapping from parameters to shape is one-to-one. For RGB-objects versions
1.0 the constraints are:

/   2 <= sds <= 10
|   0 <= shr <= 90
|   0 <= drf <= 45
|   0 <= hlw <= 90
<   0 <= shx <= shy
| shx <= shy <= 45
|  10 <= scx <= scy
| scx <= scy <= scz
\ scy <= scz <= 150

"""
import enum
from dm_robotics.manipulation.props.parametric_object import parametric_object


@enum.unique
class RgbVersion(enum.Enum):
  v1_0 = '1.0'
  v1_3 = '1.3'

# pylint: disable=bad-whitespace
_RGB_SHAPE_BOUNDS = {}
_RGB_SHAPE_BOUNDS[RgbVersion.v1_0] = {
    'sds': [[    2,    10]],
    'shr': [[    0,    90]],
    'drf': [[    0,    45]],
    'hlw': [[    0,    90]],
    'shx': [[    0, 'shy']],
    'shy': [['shx',    45]],
    'scx': [[   10, 'scy']],
    'scy': [['scx', 'scz']],
    'scz': [['scy',   150]]}

_RGB_SHAPE_BOUNDS[RgbVersion.v1_3] = {
    'sds': [[    2,    4 ],[ 2,   4  ],[ 2,    4 ],[ 2,   4  ],[ 5,   10 ]],
    'shr': [[    0,   90 ],[ 1,   90 ],[ 0,    90],[ 0,   90 ],[ 0,   90 ]],
    'drf': [[    0,   60 ],[ 0,   60 ],[ 1,    60],[ 0,   60 ],[ 0,   60 ]],
    'hlw': [[    0,   90 ],[ 0,   90 ],[ 0,    90],[ 0,   90 ],[ 0,   90 ]],
    'shx': [[    0, 'shy'],[ 0, 'shy'],[ 0, 'shy'],[ 0, 'shy'],[ 0, 'shy']],
    'shy': [['shx',   60 ],[ 0,   60 ],[ 0,    60],[ 1,   60 ],[ 0,   60 ]],
    'scx': [[   10, 'scy'],[10,   150],[10,   150],[10,   150],[10,   150]],
    'scy': [['scx', 'scz'],[10,   150],[10,   150],[10,   150],[10,   150]],
    'scz': [['scy',   150],[10,   150],[10,   150],[10,   150],[10,   150]]}

# pylint: enable=bad-whitespace

_RGB_SHAPE_NAMES_TYPES = {
    'sds': parametric_object.ParametersTypes.INTEGER,
    'shr': parametric_object.ParametersTypes.INTEGER,
    'drf': parametric_object.ParametersTypes.INTEGER,
    'hlw': parametric_object.ParametersTypes.INTEGER,
    'shx': parametric_object.ParametersTypes.INTEGER,
    'shy': parametric_object.ParametersTypes.INTEGER,
    'scx': parametric_object.ParametersTypes.INTEGER,
    'scy': parametric_object.ParametersTypes.INTEGER,
    'scz': parametric_object.ParametersTypes.INTEGER}

_RGB_TEXTURE_BOUNDS = {
    'r': [[0, 255]],
    'g': [[0, 255]],
    'b': [[0, 255]]}

_RGB_TEXTURE_NAMES_TYPES = {
    'r': parametric_object.ParametersTypes.INTEGER,
    'g': parametric_object.ParametersTypes.INTEGER,
    'b': parametric_object.ParametersTypes.INTEGER}


class RgbObject(parametric_object.ParametricObject):
  """A class to parametrically describe an RGB-object.

    Args:
      version: a string describing the RGB version to be used.
  """

  def __init__(self, version: RgbVersion = RgbVersion.v1_0) -> None:
    shape_names = tuple(_RGB_SHAPE_NAMES_TYPES.keys())
    try:
      self._shape_bounds = parametric_object.ParametricMinMaxBounds(
          _RGB_SHAPE_BOUNDS[RgbVersion(version)],
          _RGB_SHAPE_NAMES_TYPES)
    except KeyError:
      raise ValueError('Invalid `version` for RGB-objects \n'
                       f'RgbObject initialized with version: {version} \n'
                       f'Available verions are: {[v for v in RgbVersion]} \n')
    shape = parametric_object.ParametricProperties(
        shape_names, self._shape_bounds)

    texture_names = tuple(_RGB_TEXTURE_NAMES_TYPES.keys())
    texture_bounds = parametric_object.ParametricMinMaxBounds(
        _RGB_TEXTURE_BOUNDS, _RGB_TEXTURE_NAMES_TYPES)
    texture = parametric_object.ParametricProperties(
        texture_names, texture_bounds)

    super().__init__(shape, texture)

  @property
  def shape_bounds(self):
    return self._shape_bounds
