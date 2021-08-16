# Objects for robotic manipulation.

This folder provides various physical objects and software utilities used to
build simulated environments for robotic manipulation. In particular, it
provides a way to generate parametric objects with OnShape's Python API.

## An object in simulation.
It's worth pointing out that in the context of this codebase when we refer
to "objects" we don't have in mind a software object but rather a physical
object with attributes such as mass, shape and texture. The codebase helps
with generating objects representations (e.g. .STL files) with the
[OnShape][OnShapeAPI] python API client. OnShape is cloud-based CAD software
which helps creating three-dimensional shapes and their parameterizations.

## Parametric objects

An important component of this codebase is the `ParametricObject` class which
can be used to represent a parametric object and check its consitency. A
`ParametricObject` is a collection of two `ParametricProperties`,
one representing the shape and the other representing the texture.

#### [Parametric objects documentation](props/parametric_object/README.md)


### Parametric RGB-objects &#128721;&#129001;&#128311;

An example of `ParametricObject` are the [RGB-objects][RgbDocumentation].
The RGB-objects are a set of parametric objects designed in OnShape.
Their shape has been designed to offer a multitude of different affordances
for a parallel gripper grapsing and stacking. The basic idea behind
RGB-objects' shapes is to start from a 2D shape and to extrude this
shape along an axis. Both the 2D shape and the extrusion are parameterized
by a few parameters, described in the documentation page.

#### [RGB-objects &#128721;&#129001;&#128311; documentation](props/parametric_object/README.md)


## Usage example

RGB-objects are located in `props/rgb_objects` and can be initialized as following.


```python
from dm_robotics.manipulation.props.rgb_objects import rgb_object

 color_set = [
      [1, 0, 0, 1],  # RED
      [0, 1, 0, 1],  # GREEN
      [0, 0, 1, 1],  # BLUE
  ]

# Only a name of the object and its color are required. Optional parameters
# include scaling, mass etc.
prop_1 = rgb_object.RgbObjectProp(obj_id='r3', color=color_set[0])
prop_2 = rgb_object.RgbObjectProp(obj_id='g2', color=color_set[1])
prop_3 = rgb_object.RgbObjectProp(obj_id='b1', color=color_set[2])

# When a list of all available objects could be accessed with
ids_list =  rgb_object.ids_list
```

Objects could be created individually as in the above snippet.
Some predefined object sets have been defined in `rgb_object.PROP_SETS`.
These sets can be used for training and evaluation on the RGB-stacking task.

```python
object_set = 'rgb_test_set1'
_, obj_ids = rgb_object.PROP_SETS[object_set]
task_props = []
for i, obj_id in enumerate(objects):
  prop = rgb_object.RgbObjectProp(
        obj_id=obj_id, color=color_set[i], name=f'rgb_object_{obj_id}')
  task_props.append(prop)
```

We also provide some utilities to randomly sample objects. The following
snippet allows to create sets of random objects at every call of the
function `PROP_SETS[object_set]`.

```python
object_set = 'rgb_train_random'

for i in range(num_episodes):
  _, obj_ids = rgb_object.PROP_SETS[object_set]
  # `obj_ids` will provide different object ids for every access.
```

<!-- Hyperlinks  -->

[OnShapeAPI]: https://onshape-public.github.io/docs/

[RgbDocumentation]: ./
