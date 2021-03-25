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

# python3
"""Decorators for adding dynamic tuning of object properties.

```python
# Decorate with `register_class_properties` to allow registering tunable
# properties.
@register_class_properties
class Foo(object)

  def __init__(self, ...):
    # Call `register_dynamic_object` during init to tell registry that this
    # object has tunables.
    register_dynamic_object(self)

  @property
  @register_property  Label this property as a tunable getter.
  def some_property(self):
    return self._some_property

  @some_property.setter
  @register_property  Label this property as a tunable setter.
  def some_property(self, val):
    self._some_property = val
```

Follow this pattern for all objects you wish to expose for tuning.
----

To tune properties on all decorated classes add the following to your app main:
```python
gui = property_editor.PropertyEditor(poll_freq=1.)
gui.run()
```
"""

import inspect
from typing import Any, Dict, List, Text
import weakref

registered_properties = {}  # type: Dict[type, Dict[Text, 'PropertyType']]
registered_objects = {}  # type: Dict[type, 'DynamicObjects']


def overrides(interface):
  """Overrides decorator to annotate method overrides parent's."""

  def overrider(method):
    if not hasattr(interface, method.__name__):
      raise Exception(
          'method %s declared to be @override is not defined in %s' %
          (method.__name__, interface.__name__))
    return method

  return overrider


class PropertyType(object):
  """Whether a property has a getter and setter."""

  def __init__(self,
               getter: bool = False,
               setter: bool = False):
    self.getter = getter
    self.setter = setter

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return 'PropertyType(setter={}, getter={})'.format(
        self.setter, self.getter)


class DynamicObjects(object):
  """A container for a list of objects.

  DynamicObjects stores weak-references to objects to allow the cache to reflect
  only the currently existing objects whose __init__ was decorated with
  `regregister_dynamic_object`.
  """

  def __init__(self):
    self._object_refs = []

  def add_object(self, obj):
    self._object_refs.append(weakref.ref(obj))

  def get_objects(self) -> List[Any]:
    all_objs = [obj_ref() for obj_ref in self._object_refs]
    return [obj for obj in all_objs if obj is not None]

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return 'DynamicObjects(objects={})'.format(self.get_objects())


def register_class_properties(leaf_cls):
  """Decorate a class with this in order to use @register_property.

  This decorator will look for properties (setter or getter) on the provided
  class and any of its parent classes that are decorated with
  `register_property` and register them under this class.

  Args:
    leaf_cls: A class of some type.

  Returns:
    The class.
  """

  classes_to_add = (leaf_cls,) + leaf_cls.__bases__
  for cls in classes_to_add:
    for name, method in cls.__dict__.items():
      # Add property to registry as getter.
      if (hasattr(method, 'fget') and
          hasattr(method.fget, 'register_getter')) or hasattr(
              method, 'register_getter'):
        registered_properties.setdefault(leaf_cls, {})
        registered_properties.get(leaf_cls).setdefault(name, PropertyType())  # pytype: disable=attribute-error
        registered_properties.get(leaf_cls).get(name).getter = True  # pytype: disable=attribute-error

      # Add property to registry as setter.
      if (hasattr(method, 'fset') and
          hasattr(method.fset, 'register_setter')) or hasattr(
              method, 'register_setter'):
        registered_properties.setdefault(leaf_cls, {})
        registered_properties.get(leaf_cls).setdefault(name, PropertyType())  # pytype: disable=attribute-error
        registered_properties.get(leaf_cls).get(name).setter = True  # pytype: disable=attribute-error

  return leaf_cls


def register_property(func):
  """Adds a property to registered_properties. must appear after @property."""
  if isinstance(func, property):
    raise AssertionError('@register_property must be after @property')

  argspec = inspect.getfullargspec(func)  # pytype: disable=wrong-arg-types
  if len(argspec.args) == 1:
    func.register_getter = True
  elif len(argspec.args) == 2:
    func.register_setter = True
  return func


def register_dynamic_object(obj):
  """Stores the provided object in the registry of dynamic objects.

  This function should be called during __init__ on all objects that utilize
  `register_property` to expose tunable properties.

  Args:
    obj: The `self` argument to the object to register.
  """
  # Add obj reference to registered objects.
  cls = obj.__class__
  registered_objects.setdefault(cls, DynamicObjects())
  registered_objs = registered_objects.get(cls)
  assert registered_objs is not None
  registered_objs.add_object(obj)
