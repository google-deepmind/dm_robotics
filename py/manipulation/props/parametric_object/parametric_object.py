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

"""A class that defines parametric objects.

The class ParametricObject is used to collect the object shape (ParametricShape)
and texture (ParametricTexture). Both shape and texture are represented as a
collection of parameters. Parameters live in a space which is the union of
bounding box conditions (e.g. m <= param_1 <= M); bounding boxes can also
be specified in terms of other paramers (e.g. m <= param_1 <= param_2).
"""

import collections
import enum
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


@enum.unique
class ParametersUnits(enum.Enum):
  MILLIMETER = 'mm'
  CENTIMETER = 'cm'
  METER = 'm'
  DEGREE = 'deg'
  RADIAN = 'rad'
  ADIMENSIONAL = ''


@enum.unique
class ParametersTypes(enum.Enum):
  INTEGER = int
  FLOAT = float


class ParametersDict(collections.UserDict):
  """A dictionary that allows some arithmetic on dictionaries.

  This class inherirs from a `UserDict` and overloads the operators `+`,
  `-`, `*`, `//` to allow some arithmetic operations on dictionaries that
  describe `ParametricProperties`. Additions and subtractions operate on
  two dictionaries (`A` and `B`) key by key adding/subtracting the
  corresponding values (e.g. `A-B`). Multiplications and divisions operate
  with a scalar float or int (`s`) on a dictionary (`A`) from the right-hand
  side (e.g. `A*s` or `A//s`) and return a dictionary where all values have
  been multiplied or divided by the scalar. With multiplications and
  divisions, parameters are casted according to their type as specified in
  the constructor (default types are assumed int).
  """

  def __init__(self,
               other: Optional[Mapping[str, Any]], *,
               param_types: Optional[Tuple[ParametersTypes]] = None):
    super().__init__()
    if param_types is None:
      self._param_types = (ParametersTypes.INTEGER,) * len(other)
    else:
      self._param_types = param_types
    self.update(**other)

  def __add__(self, other):
    r = dict()
    if isinstance(other, self.__class__):
      if self.param_types != other.param_types:
        ValueError('The added ParametersDict have non-matching param_types.')
      for self_key, other_key in zip(self, other):
        if self_key != other_key:
          raise ValueError('The added ParametersDict have non-matching keys.')
        r[self_key] = self[self_key] + other[other_key]
      return ParametersDict(r, param_types=self.param_types)
    else:
      raise TypeError(f'unsupported __add__ operand type(s) '
                      f'for +: {self.__class__} and {type(other)}')

  def __sub__(self, other):
    r = dict()
    if isinstance(other, self.__class__):
      if self.param_types != other.param_types:
        ValueError('Subtracted ParametersDict have non-matching param_types.')
      for self_key, other_key in zip(self, other):
        if self_key != other_key:
          raise ValueError('The added ParametersDict have non-matching keys.')
        r[self_key] = self[self_key] - other[other_key]
      return ParametersDict(r, param_types=self.param_types)
    else:
      raise TypeError(f'unsupported __sub__ operand type(s) '
                      f'for -: {self.__class__} and {type(other)}')

  def __mul__(self, scale):
    r = dict()
    if isinstance(scale, float) or isinstance(scale, int):
      for self_key, type_key in zip(self, self._param_types):
        r[self_key] = type_key.value(self[self_key] * scale)
      return ParametersDict(r, param_types=self.param_types)
    else:
      raise TypeError(f'unsupported __mul__ operand type(s) '
                      f'for *: {self.__class__} and {type(scale)} ')

  def __floordiv__(self, scale):
    r = dict()
    if isinstance(scale, float) or isinstance(scale, int):
      for self_key in self:
        r[self_key] = self[self_key] // scale
      return ParametersDict(r, param_types=self.param_types)
    else:
      raise TypeError(f'unsupported __floordiv__ operand type(s) '
                      f'for //: {self.__class__} and {type(scale)} ')

  def __truediv__(self, scale):
    r = dict()
    if isinstance(scale, float) or isinstance(scale, int):
      for self_key, type_key in zip(self, self._param_types):
        r[self_key] = type_key.value(self[self_key]/scale)
      return ParametersDict(r, param_types=self.param_types)
    else:
      raise TypeError(f'unsupported __truediv__ operand type(s) '
                      f'for /: {self.__class__} and {type(scale)} ')

  def distance(self, other):
    r = 0
    if isinstance(other, self.__class__):
      for self_key, other_key in zip(self, other):
        if self_key != other_key:
          raise ValueError('The added ParametersDict have non-matching keys.')
        r = r + (self[self_key] - other[other_key]) ** 2
      return r
    else:
      raise TypeError(f'unsupported operand type(s) '
                      f'for `distance`: {self.__class__} and {type(other)}')

  @property
  def param_types(self) -> Tuple[ParametersTypes]:
    """Returns the tuple that contains the types of the parameters.

    Returns:
      tuple with the types of the parameters.
    """
    return self._param_types


class ParametricMinMaxBounds():
  """A class to parametrically describe the parametric properties of an object.

  Each parameter is specified with three quantities which correspond
  to the name of the parameters, their minimum and their maximum value. The
  name of each parameter is a string, the minimum and maximum can be either a
  integer or a string, the latter option if they refer to the value of another
  parameter. The minimum and maximum values are specified as tuple of lists
  with minimum and maximum values. Each list specifies a subset for min and max
  values of each parameter, the range of values for parameters is union of
  all the subsets. Parameters can be either integers or floats. For each
  parameter we have a string `si` (alphabetic only), a K-tuple of min (`mi1`,
  ..., `miK`) and a K-tuple for max (`Mi1`, ..., `MiK`) values. The string is
  used to specify the name of the parameters (`s1`, ..., `sN`). The K-tuples
  contain the mins ((`m11`, ..., `mN1`), ..., (`m1K`, ..., `mNK`)) and the max
  ((`M11`, ...,`MN1`), ..., (`M1K`, ..., `MNK`)). Both min and max can be
  integers or strings; if a string then it has to coincide with the name of
  another parameter which will be used as the min or max value. The valid set of
  parameters is the union of K-subsets, each subset being (`m1k <= s1 <= M1k`,
  ..., `mNk <= sN <= MNk`) with `k` = `1`, ..., `K`.
  """

  def __init__(
      self,
      param_names_bounds: Dict[str, Sequence[List[Union[int, str]]]],
      param_dict_types: Optional[Dict[str, ParametersTypes]] = None) -> None:
    """The ParametricShape class initialiazer.

    Args:
      param_names_bounds: a dictionary with keys corresponding to the parameters
      names. Values needs to be to a list of lists. These lists have exactly
      two elements. The first element is the min value for the given parameter.
      The second element is the max value for the given parameter.
      param_dict_types: (optional) a dictionary with keys corresponding to the
      parameters names and values corresponding to the type of paramesters
      (currently only int and flot supported).
    """
    param_names = tuple(param_names_bounds.keys())
    param_bounds = tuple(param_names_bounds.values())

    # Check the same number of min-max is provided
    if param_bounds:
      result = all(len(elem) == len(param_bounds[0]) for elem in param_bounds)
      if not result:
        raise ValueError('The provided bounds do noy have the same '
                         'dimension for all provided parameters')
    else:
      raise ValueError('The provided sequences of bounds are empty.')

    # Check all min-max values are present
    for n in param_names:
      for b in param_names_bounds[n]:
        if len(b) != 2:
          raise ValueError(f'The provided bounds for {n} have wrong size. '
                           f'Expecting size 2 but I have found size {len(b)}')

    number_bounds = len(param_bounds[0])
    param_mins, param_maxs = [], []
    for j in range(0, number_bounds):
      param_mins.append(tuple([d[j][0] for d in param_bounds]))
      param_maxs.append(tuple([d[j][1] for d in param_bounds]))
    param_mins, param_maxs = tuple(param_mins), tuple(param_maxs)

    for param_min, param_max in zip(param_mins, param_maxs):
      if len(param_names) != len(param_min) or len(param_min) != len(
          param_max):
        raise ValueError('ParametricProperties '
                         'initialized with different sizes.\n'
                         f'param_names length is: {len(param_names)}\n'
                         f'param_min length is: {len(param_min)}\n'
                         f'param_max length is: {len(param_max)}\n')
    for n in param_names:
      if not n.isalpha():
        raise ValueError('Property names should be alphabetic only. ' +
                         (f'Found a non-alpahbetic character: {n}. ') +
                         'Numbers are reserved for parameters value. '
                         'See the `get_name` mathod of this class.')
    self._param_mins = param_mins
    self._param_maxs = param_maxs
    self._param_names = param_names
    if param_dict_types is None:
      self._param_types = tuple([])
    else:
      self._param_types = tuple(param_dict_types.values())

  def __call__(self, values: ParametersDict) -> bool:
    """A function to check the validity of the provided parameters.

    The function requires a dictionary which contain all parameter values.
    The dictionary is structured as follows:
    {'par1': values[0], ..., 'parN': values[N-1]}

    Args:
      values: a dictionary ['params': value] for property.

    Returns:
      true/false depending on the instanced values being within linear bounds.
    """
    check_belong_subset_union = False
    check_belong_subset_error = ''
    for param_min, param_max in zip(self._param_mins, self._param_maxs):
      # Variable to check parameters belong to at least one subset
      check_belong_subset = True

      # Check that provided dictionary has all properties names.
      if not all(name in values for name in self._param_names):
        raise ValueError('The provided dictionary misses some parameters. '
                         f'Class instantianted with: {self._param_names}. '
                         f'Parameter provided: {values.keys()}.')

      # Check that provided dictionary are compatible with max and min values
      for p, lb, ub in zip(self._param_names, param_min, param_max):
        value = values[p]

        # Define the lowerbound, conditional on `lb` being a `str` or an `int`
        if isinstance(lb, int):
          lower = lb
        elif isinstance(lb, str):
          lower = values[lb]
        # Define the upperbound, conditional on `ub` being a `str` or an `int`
        if isinstance(ub, int):
          upper = ub
        elif isinstance(ub, str):
          upper = values[ub]

        if not lower <= value <= upper:
          check_belong_subset = False
          check_belong_subset_error = check_belong_subset_error + (
              f'{p} : '
              f'{lower} <= {value} <= {upper}\n')
      if not check_belong_subset:
        check_belong_subset_error = check_belong_subset_error +(
            f'Checking min values: {param_min} \n' +
            f'and max values: {param_max} \n' +
            f'for the object parameters: {values} \n')

      check_belong_subset_union = check_belong_subset_union or check_belong_subset

    if not check_belong_subset_union:
      logging.info('Wrong object configuration:\n%s', check_belong_subset_error)
      return False

    # Testing the provided values are of the right type
    if self._param_types:
      for p, t in zip(self._param_names, self._param_types):
        if not isinstance(values[p], t.value):
          logging.info('Types for parameters %s are wrong: value %s not %s',
                       self._param_names, values[p], t)
          logging.info('Parameter types are: %s. ', self._param_types)
          logging.info('Parameter instance is: %s. ', values)
          return False

    return True


class ParametricProperties:
  """A class to parametrically describe the parametric properties of an object.

  ParametricProperties class manipulates parametric properties. A parametric
  property is the collection of two things: (1) the parameters names in the form
  of a N-tuple of string (e.g. ('param1_name', ..., 'paramN_name')); (2) a
  function which takes a dict (e.g. {'param1_name': val1, ...,
  'paramN_name': valN}) and checks if the given values are valid. Both the
  param names and the function are specified in the constructor. A simple
  example of a parametric object can be a cube which has only one parameter
  (the length of the cube edge); the validity check is for this parameter to be
  grather than zero.
  """

  def __init__(
      self,
      param_names: Tuple[str],
      param_bounds_function: ParametricMinMaxBounds,
      param_units: Optional[Tuple[ParametersUnits]] = None,
      param_types: Optional[Tuple[ParametersTypes]] = None) -> None:
    """The ParametricShape class initialiazer.

    Args:
      param_names: a tuple containing the parameter names
      param_bounds_function: a function that returns a bool given a dictionary
        which defines an instance of a parametric object. This function returns
        true if the given instance is valid (i.e. inside the bounds), false
        otherwise. The second element is the max value for the given parameter.
      param_units: a tuple containing the unit of measure for the provided
        parameters. If the parameter is a length accepted units are
        millimeter, centimeter and meter (i.e. 'mm', 'cm', 'm'). If the
        parameter is an angle accepted units are degree and radian (i.e. 'deg',
        'rad'). If the parameter is adimensional then an empty string should be
        provided. If no param_units is provided all parameters are
        considered adimensional.
      param_types: a tuple containing the types (int/float) for the provided
        parameters. If no param_types is provided all parameters are
        considered integers.
    """
    self._param_bounds_function = param_bounds_function
    self._param_names = param_names
    if param_units is None:
      self._param_units = (ParametersUnits.ADIMENSIONAL,) * len(
          self._param_names)
    else:
      self._param_units = param_units

    if param_types is None:
      self._param_types = (ParametersTypes.INTEGER,) * len(self._param_names)
    else:
      self._param_types = param_types

  def check_instance(self, values: ParametersDict) -> bool:
    """A function to check the validity of the provided parameters.

    The function requires a dictionary which contain all parameter values.
    The dictionary is structured as follows:
    {'par1': values[0], ..., 'parN': values[N-1]}

    Args:
      values: a dictionary ['params': value] for property.

    Returns:
      true/false depending on the instanaced values being valid or not.
    """
    return self._param_bounds_function(values)

  def get_name(self, property_dict: ParametersDict) -> str:
    """Creates a string which can be used as name for a parametric property.

    The input dictionary should have the following structure:
                {'par1': values[0], ..., 'parN': values[N-1]}
    Args:
      property_dict: a dictionary containing ['params': value].

    Returns:
      string with the names of the parameters separated by '_'.
    """
    self.check_instance(property_dict)
    return '_'.join('{}{}'.format(k, v) for k, v in property_dict.items())

  @property
  def param_names(self) -> Tuple[str, ...]:
    """Returns the tuple that contains the names of the parameters.

    Returns:
      tuple with the names of the parameters.
    """
    return self._param_names

  @property
  def param_types(self) -> Tuple[ParametersTypes]:
    """Returns the tuple that contains the types of the parameters.

    Returns:
      tuple with the types of the parameters.
    """
    return self._param_types

  @property
  def param_units(self) -> Tuple[ParametersUnits]:
    """Returns the tuple that contains the units of the parameters.

    Returns:
      tuple with the units of the parameters.
    """
    return self._param_units

  def get_dict(self, param_string: str) -> ParametersDict:
    """Creates a dict given the string that describes the parametric properties.

    The input string should have the following structure:
                'par1values[0]_ ... _parNvalues[N-1]'
    The output dict will have the following structure:
                {'par1': values[0], ..., 'parN': values[N-1]}
    Numeric values (i.e. values[0], ..., values[N-1]) can be `int` or `float`.
    If `float` we assume the number is represented with a string of the form x.y
    (i.e. `first0.01_second3_third12.25` is converted to {'first': 0.01,
    'second': 3, 'third': 12.25}).
    Args:
      param_string: a srting that describes the parametric property.

    Returns:
      a dict that describes the parametric property.
    """
    array_names = re.findall('[a-zA-Z]+', param_string)
    array_values = re.findall(r'(\d+(?:\.\d+)?)', param_string)
    array = param_string.split('_')

    if len(array_names) != len(array_values):
      raise ValueError(f'Something wrong converting: {param_string}.')
    if len(array_names) != len(array):
      raise ValueError(f'Something wrong with delimiters: {param_string}.')

    shape_dict = ParametersDict({})
    iter_names = iter(self.param_names)
    for value, name in zip(array_values, array_names):
      if name != next(iter_names):
        raise ValueError(f'Wrong parameters in given string: {param_string}.')
      try:
        shape_dict[name] = int(value)
      except ValueError:
        try:
          shape_dict[name] = float(value)
        except ValueError:
          raise ValueError(f'Cannot convert {value} to int or float.')
    try:
      iter_names.__next__()
    except StopIteration:
      return shape_dict
    raise ValueError('Error: not all parameters were provided.')


class ParametricObject:
  """A class to parametrically describe an object.

  The object class is a collection of a parametric shape and a parametric
  texture.
  """

  def __init__(self,
               object_shape: ParametricProperties,
               object_texture: ParametricProperties):
    """The ParametricObject class initialiazer.

    Args:
      object_shape: the ParametricProperties to describe the object shape.
      object_texture: the ParametricProperties to describe the object texture.
    """

    self.shape = object_shape
    self.texture = object_texture

  def check_instance(self,
                     shape_dict: ParametersDict,
                     texture_dict: ParametersDict) -> bool:
    """A function to check the validity of the provided parameters.


    Args:
      shape_dict: a dictionary containing ['params': value] for shape.
      texture_dict: a dictionary containing ['params': value] for texture.

    Returns:
      true if and only if both shape_dict and texture_dict are valid
    """
    check_shape = self.shape.check_instance(shape_dict)
    check_texture = self.texture.check_instance(texture_dict)
    return check_shape and check_texture

  def get_name(
      self,
      shape_dict: ParametersDict,
      texture_dict: ParametersDict) -> str:
    """Creates a string which can be used as name of the parametric object.

    The input dictionary should have the following structure:
                {'par1': values[0], ..., 'parN': values[N-1]}
    The output string will have the following structure:
                'par1values[0]_ ... _parNvalues[N-1]'
    Args:
      shape_dict: a dictionary describing the parametric shape.
      texture_dict: a dictionary describing the parametric texture.

    Returns:
      a string which is the concatenation of shape and texture.
    """
    shape_name = self.shape.get_name(shape_dict)
    texture_name = self.texture.get_name(texture_dict)

    return shape_name + '_' + texture_name
