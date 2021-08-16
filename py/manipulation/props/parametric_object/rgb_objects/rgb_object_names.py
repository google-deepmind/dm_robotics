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

"""Define some common object shortcuts for RGB objects.

We can configure generative objects somewhat freely in parameters, but in
current experiments, we use a discrete set of nicknamed objects. These objects
are defined with a per-object constant set of parameters. For easier use, these
are specified here.

The initial version of RGB-objects (named RGB30) was created manually and is not
a part of the current parametric object pipeline.

There is a visualization of the objects and more information can be found here:
https://sites.google.com/corp/google.com/rgb--stacking#h.p_Hbvm_ijsde_K
"""
import collections
import copy
import itertools

from typing import Dict, Tuple
from dm_robotics.manipulation.props.parametric_object import parametric_object
from dm_robotics.manipulation.props.parametric_object.rgb_objects import parametric_rgb_object


# RGB-objects v1.0 are created with 3 deformations of a seed object (a cube with
# a 50mm side): G minor deformation, B average deformation, R major
# deformation. Deformations are chosen by sampling independently 5 parameters of
# the RGB-shapes. We have chosen: 1 - hollowness; 2 - number of sides;
# 3 - shrinking; 4 - not used; 5 - shear; 6 - form factor.

ParametersDict = parametric_object.ParametersDict
RgbVersion = parametric_rgb_object.RgbVersion

# Breaking these over > 100 lines does not help visibility so:
# pylint: disable=line-too-long, bad-whitespace
# pyformat: disable

_OBJECTS_V1_0 = collections.OrderedDict({
    "g1": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw": 15, "shx": 0, "shy":  0, "scx": 50, "scy": 50, "scz":  50 }),
    "b1": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw": 20, "shx": 0, "shy":  0, "scx": 50, "scy": 50, "scz":  50 }),
    "r1": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw": 35, "shx": 0, "shy":  0, "scx": 50, "scy": 50, "scz":  50 }),
    "g2": ParametersDict({ "sds":  6, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 46, "scy": 46, "scz":  50 }),
    "b2": ParametersDict({ "sds":  8, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 45, "scy": 45, "scz":  50 }),
    "r2": ParametersDict({ "sds": 10, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 45, "scy": 45, "scz":  50 }),
    "g3": ParametersDict({ "sds":  4, "shr": 25, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 51, "scy": 51, "scz":  60 }),
    "b3": ParametersDict({ "sds":  4, "shr": 48, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 46, "scy": 49, "scz":  63 }),
    "r3": ParametersDict({ "sds":  4, "shr": 75, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 41, "scy": 49, "scz":  71 }),
    "g5": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy": 20, "scx": 50, "scy": 50, "scz":  50 }),
    "b5": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy": 31, "scx": 50, "scy": 50, "scz":  50 }),
    "r5": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy": 42, "scx": 50, "scy": 50, "scz":  50 }),
    "g6": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 40, "scy": 56, "scz":  80 }),
    "b6": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 32, "scy": 48, "scz":  96 }),
    "r6": ParametersDict({ "sds":  4, "shr":  0, "drf": 0, "hlw":  0, "shx": 0, "shy":  0, "scx": 29, "scy": 29, "scz": 150 })})
# pylint: enable=line-too-long, bad-whitespace
# pyformat: enable


# RGB-objects v1.3 adds two deformation axes to v1.1 (axis 7 and axis 8). The
# axis 6 in v1.1 is the deformation of the form factor by increasing the size
# along the z-axis. With v1.3 we introduce deformations which correspond to
# scaling along the x-axis (axis 7) and along the y-axis (axis 8). Moreover,
# we generate all objects as interpolations from a seed object s and 7 objects
# r2, r3, r5, r6, r7, r8 which are the maximum deformations of the seed
# object along the available axes 2, 3, 5, 6, 7, 8. Finally also, v1.3
# defines the objects Omn with Omn = (Om + On)//2.

# pylint: disable=line-too-long, bad-whitespace
# pyformat: disable
_OBJECTS_V1_3 = collections.OrderedDict({
    "s":  ParametersDict({ "sds": 4, "shr": 0, "drf": 0, "hlw": 0, "shx": 0, "shy": 0, "scx": 50, "scy": 50, "scz": 50}),
    "r2": ParametersDict({ "sds": 10, "shr": 0, "drf": 0, "hlw": 0, "shx": 0, "shy": 0, "scx": 45, "scy": 45, "scz": 50}),
    "r3": ParametersDict({ "sds": 4, "shr": 75, "drf": 0, "hlw": 0, "shx": 0, "shy": 0, "scx": 41, "scy": 49, "scz": 71}),
    "r5": ParametersDict({ "sds": 4, "shr": 0, "drf": 0, "hlw": 0, "shx": 0, "shy": 42, "scx": 50, "scy": 50, "scz": 50}),
    "r6": ParametersDict({ "sds": 4, "shr": 0, "drf": 0, "hlw": 0, "shx": 0, "shy": 0, "scx": 29, "scy": 29, "scz": 150}),
    "r7": ParametersDict({ "sds": 4, "shr": 0, "drf": 0, "hlw": 0, "shx": 0, "shy": 0, "scx": 29, "scy": 150, "scz": 29}),
    "r8": ParametersDict({ "sds": 4, "shr": 0, "drf": 0, "hlw": 0, "shx": 0, "shy": 0, "scx": 150, "scy": 29, "scz": 29})})
_OBJECTS_V1_3_NON_GRASPABLE = ("l36", "m36", "y36", "r36", "r67", "v36")
_OBJECTS_V1_3_NON_UNIQUE = ("f25", "h27", "l27", "m27", "r27", "r68", "u27", "v27", "x27", "y27", "h28", "l28", "m28", "r28", "u28", "v28", "x28", "y28", "r78")

# pylint: enable=line-too-long, bad-whitespace
# pyformat: enable


def parameters_interpolations(
    params_dict1: ParametersDict,
    params_dict2: ParametersDict,
    interpolation_length: int = 1,
    interpolation_keys: Tuple[str, ...] = ()) -> collections.OrderedDict:
  """Function to interpolate in between two parametersDict.

  This function can be used to interpolate in between parametersDicts. The
  function takes as input two parameterDicts. The function interpolates in
  between the parametersDicts generating a given number of equally-spaced
  samples. By default, only one sample is added, corresponding to an element
  in between the two provided parameterDicts (e.g. m = (s+e)/2). Generated
  parameterDicts are returned in a collection. By default, associated labels are
  m1, m2, ... or otherwise specified by the user through a tuple.

  Args:
    params_dict1: the first parameterDict.
    params_dict2: the second parameterDict.
    interpolation_length: the numnber of interpolation samples.
    interpolation_keys: the keys used in the resulting collection.
  Returns:
    the collection of combinations
  """

  result_dictionary = collections.OrderedDict({})
  # Creating intermediate objects from two adjucent ones.
  if not interpolation_keys:
    for i in range(0, interpolation_length):
      interpolation_length = (*interpolation_length, "m" + str(i))
  for i in range(1, interpolation_length+1):
    obj_nickname = interpolation_keys[i - 1]
    step = i / (interpolation_length + 1)
    obj_values = params_dict1 + (params_dict2 - params_dict1) * step
    result_dictionary.update({obj_nickname: obj_values})
  return result_dictionary


def parameters_numeric_combinations(
    params_dict_collection: collections.OrderedDict,
    labels_alphabetic_keys: Tuple[str, ...],
    labels_numeric_keys: Tuple[str, ...],
    combination_length: int = 2) -> collections.OrderedDict:
  """Function to combine collections of parametersDict with alphanumeric keys.

  This function can be used to create combinations of parametersDict. The
  function takes as input a collection of parameterDicts each labelled with an
  alphanumeric string (e.g. e1, e2, e3, g1, g2, g3). The function combines the
  parametersDicts taking the set of alpahbetic keys (e.g. {e, g}) and the set of
  numeric keys (e.g. {1, 2, 3}). By default, for each alphabetic key all
  2-combinations of numeric keys are created using the parameterDicts algebra.
  In the example above we have: e12 = (e1 + e2) // 2, e13 = (e1 + e3) // 2,
  e23 = (e2 + e3) // 2, g12 = (g1 + g2) // 2, g13 = (g1 + g3) // 2,
  g23 = (g2 + g3) // 2. Otherwise, a specific combination length can be
  specified. If 3-combination is specified then the following parameterDicts
  are created: e123 = (e1 + e2 + e3) // 3 and g123 = (g1 + g2 + g3) // 3.

  Args:
    params_dict_collection: a collection of parametersDict. The keys associated
      to each parametersDict should be alphanumeric.
    labels_alphabetic_keys: the alphabetic part of the key labels.
    labels_numeric_keys: the numeric part of the key labels.
    combination_length: the length of cretated combinations.
  Returns:
    the collection of combinations
  """

  result_dictionary = collections.OrderedDict({})
  # Creating intermediate objects from two adjacent ones.
  for alpha in labels_alphabetic_keys:
    for num in itertools.combinations(labels_numeric_keys, combination_length):
      obj_nickname = alpha
      obj_nickname = obj_nickname + num[0]
      obj_values = params_dict_collection[alpha + num[0]]
      for i in range(1, combination_length):
        obj_nickname = obj_nickname + num[i]
        obj_values = obj_values + params_dict_collection[alpha + num[i]]
      obj_values = obj_values // combination_length
      result_dictionary.update({obj_nickname: obj_values})
  return result_dictionary


def parameters_equispaced_combinations(
    params_dict_collection: collections.OrderedDict,
    coefficients: Tuple[int, ...]) -> collections.OrderedDict:
  """Function to create equispaced combinations.

  This function can be used to create equispaced distributed combinations of
  parametersDict. The function takes as input a collection of alphabetically
  tagged parameterDicts (e.g. a, .. z). The function combines the given
  parametersDicts to create new parametersDicts constructed as a*ca + ..
  + z*cz with ca + .. + cz = 1. The number of generated parametersDicts is
  controlled by fixing the valid values for the coefficients cn. The resulting
  objects are named aca_..._zcz.

  Args:
    params_dict_collection: a collection of parametersDict.
    coefficients: the valid coefficients (tuple of int) expressed as integer
      percentage, (0, 25, 50, 75, 100) corresponds to (0, 0.25, 0.5, 0.75, 1).
  Returns:
    the collection of combinations
  """

  result_dictionary = collections.OrderedDict({})
  n = len(params_dict_collection)

  # Creating valid combinations
  valid_combinations = [
      s for s in itertools.product(coefficients, repeat=n) if sum(s) == 100
  ]

  # Creating convex combinations of objects
  result_dictionary = collections.OrderedDict({})
  for valid_combination in valid_combinations:
    obj_nickname = ""
    obj_values = None
    p = params_dict_collection
    for kn, vn, cn in zip(p.keys(), p.values(), valid_combination):
      if obj_values is None:
        obj_values = vn * cn
        if cn != 0:
          obj_nickname = str(coefficients.index(cn)) + kn  # pytype: disable=attribute-error
      else:
        obj_values = obj_values + vn * cn
        if cn != 0:
          obj_nickname = obj_nickname + str(coefficients.index(cn)) + kn
    result_dictionary.update({obj_nickname: obj_values//100})
  return result_dictionary


class RgbObjectsNames:
  """A class to define the RGB-objects names according to different versions.

    Args:
      version: string to describe the RGB-objects version.
  """

  def __init__(self, version: RgbVersion = RgbVersion.v1_0):
    self.__version__ = version
    self._nicknames = collections.OrderedDict({})

    if version == RgbVersion.v1_0:
      self._nicknames.update(_OBJECTS_V1_0)
    if version == RgbVersion.v1_3:
      self._nicknames = collections.OrderedDict(copy.deepcopy(_OBJECTS_V1_3))
      # Adding dn, fn, en, hn, xn, ln, bn, mn, yn, un
      # linearly interpolating 10 objects in between "s" and "tn"
      for n in ("2", "3", "5", "6", "7", "8"):
        self._nicknames.update(parameters_interpolations(
            _OBJECTS_V1_3["s"],
            _OBJECTS_V1_3["r" + n],
            10, ("d"+n, "f"+n, "e"+n, "u"+n, "h"+n,
                 "x"+n, "l"+n, "v"+n, "m"+n, "y"+n)))
      # Removing the seed object
      self._nicknames.pop("s")

      # Creating intermediate Omn = (Om + On)//2.
      self._nicknames.update(parameters_numeric_combinations(
          self._nicknames,
          ("d", "f", "e", "h", "x", "l", "m", "y", "r", "u", "v"),
          ("2", "3", "5", "6", "7", "8"),
          2))

      # Remove non-graspable and non-unique
      for o in _OBJECTS_V1_3_NON_GRASPABLE + _OBJECTS_V1_3_NON_UNIQUE:
        self._nicknames.pop(o, None)

      # Add RGB v1.0 objects, except for the hollow ones.
      self._nicknames.update(_OBJECTS_V1_0)
      for o in ["r1", "b1", "g1"]:
        self._nicknames.pop(o, None)

    # This is necessary to guarantee one-to-one mapping: parameters <-> shapes
    for v in self._nicknames.values():
      if (v["shr"], v["drf"], v["hlw"], v["shx"], v["shy"]) == (0,)*5:
        ordered_scale = sorted((v["scx"], v["scy"], v["scz"]))
        v["scx"] = ordered_scale[0]
        v["scy"] = ordered_scale[1]
        v["scz"] = ordered_scale[2]

    # Look for duplicates, print log-info if found and raise an error.
    my_rgb = parametric_rgb_object.RgbObject(version)
    uniques, duplicates = set(), set()
    self._duplicates_groups = dict()
    uniques_dict = {}
    for obj_nick, obj_dict in self.nicknames.items():
      obj_name = my_rgb.shape.get_name(obj_dict)
      if obj_name in uniques:
        duplicates.add(obj_nick)
        self._duplicates_groups[
            obj_name] = self._duplicates_groups[obj_name] + (obj_nick,)
      else:
        uniques.add(obj_name)
        self._duplicates_groups.update({obj_name: (obj_nick,)})
        uniques_dict.update({obj_nick: obj_name})
    if duplicates:
      for o in duplicates:
        self._nicknames.pop(o, None)

  @property
  def nicknames(self) -> Dict[str, ParametersDict]:
    # Dictionary of creation parameters sorted by object names.
    return collections.OrderedDict(sorted(self._nicknames.items()))

  @property
  def duplicates(self) -> Dict[str, Tuple[str]]:
    # Dictionary of object names and associated nicknames.
    return self._duplicates_groups
