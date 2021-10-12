# RGB-objects &#128721;&#129001;&#128311; for robotic manipulation

This folder provides the RGB-objects introduced in the paper
[Beyond Pick-and-Place: Tackling Robotic Stacking of Diverse Shapes][pick_and_place_paper]
and related code used by the [RGB-Stacking][rgb_stacking] environment.


## The RGB-objects family
The RGB-objects are all obtained by applying a certain deformation to a cube,
which is our seed object, referred to as `s0` in code.
We have defined four major axes of deformation of the seed object, which result
in different shapes. These shapes can also be thought of as the vertical
extrusion of a 2D shape, which is also the name of each axis. The names of the
axes in code are numerical:

- **Axis 2 - Polygon**: deformation obtained by transforming the
  extruded planar shape (i.e. the square) into a regular polygon.
- **Axis 3 - Trapezoid**: deformation obtained by progressively
  morphing the planar square to a isosceles trapezoid.
- **Axis 5 - Parallelogram**: deformation obtained by changing the
  orientation of the extrusion axis, from vertical (i.e. orthogonal to the plane
  of the planar shape) to progressively more slanted axes.
- **Axis 6 - Rectangle**: deformation by uniformly
  scaling the object along the X, Y or Z-axis. Note that the x-, y-,z- Rectangle axes are the same so we refer to these as a single major Rectangle axis.


These deformations and their combinations define a parametric family of objects.
For our Skill Generalization task (see the  [paper][pick_and_place_paper])
we designate the axes of all pairwise combined deformations as the
"training axes" and the ones of single deformation - the major axes above - as
the "held-out axes". Pairwise mixing of some of the major axes leads to
objects that are duplicates and therefore we omitted certain axes and objects.

Based on the resulting 15 axes, we created a training object set, which consists of 103
different shapes, and a held-out set. The diagram and gifs below shows a depiction of
all the objects in the benchmark. The seed object is at the center; all the other
objects are the result of deformations of this cube. The held-out objects
(major axes) are enclosed in the uper teal sector of the diagram; the training objects (pairwise
mixing of two major axes) are enclosed in the blue sector. Some objects cannot
be grasped with a parallel gripper with 85 mm aperture (i.e. the Robotiq 2F-85);
these objects are transparent and were omitted in our experiments.

![disk of objects][object_disk]


## STL assets
<section class="zippy open">
Under the [`meshes`][meshes_dir] directory there are 152 STL files representing
the RGB-object dataset.

### Assets directories structure
All mesh assets are located in the [`meshes`][meshes_dir] directory where they
are further split between following subdirectories:

- [`train`][meshes_train_dir] directory with 103 objects.
- [`test_triplets`][meshes_test_dir] directory with 13 objects including the
  seed `s0`, to form 5 test triplets.
- [`heldout`][meshes_heldout_dir] directory with 36 objects.

### File name convention
All asset files are in STL format and have an `*.stl` extention. The
file name starts with the object ID and is followed by a sequence of
parameters that uniquely describe the object.
For example, `b3_sds4_shr48_drf0_hlw0_shx0_shy0_scx46_scy49_scz63.stl` is the
filename for the `b3` object.

### Visualizing STL assets
[`meshlab`][meshlab] is a convinient tool to work with the provided meshes. To
load a mesh from a command line:

```bash
$ cd props/rgb_objects/assets/rgb_v1/meshes
$ meshlab test_triplets/b3_sds4_shr48_drf0_hlw0_shx0_shy0_scx46_scy49_scz63.stl
```

</section>


<section class="zippy close">
A map of the object IDs is provided here:

Held-Out Axis 2, **Polygon**: \
![Axis of transformation 2][axis2] \
Held-Out Axis 3, **Trapezoid**: \
![Axis of transformation 3][axis3] \
Held-Out Axis 5, **Parallelogram**: \
![Axis of transformation 5][axis5] \
Held-Out Axis 6, **Rectangle**: \
![Axis of transformation 6][axis6] \
Training Axis 23, **Polygon & Trapezoid**: \
![Axis of transformation 23][axis23] \
Training Axis 25, **Polygon & Parallelogram**: \
![Axis of transformation 25][axis25] \
Training Axis 26, **Polygon & Rectangle**: \
![Axis of transformation 26][axis26] \
Training Axis 35, **Trapezoid & Parallelogram**: \
![Axis of transformation 35][axis35] \
Training Axis 36, **Trapezoid & x-Rectangle**: \
![Axis of transformation 36][axis36] \
Training Axis 37, **Trapezoid & y-Rectangle**: \
![Axis of transformation 37][axis37] \
Training Axis 38, **Trapezoid & z-Rectangle**: \
![Axis of transformation 38][axis38] \
Training Axis 56, **Parallelogram & x-Rectangle**: \
![Axis of transformation 56][axis56] \
Training Axis 57, **Parallelogram & y-Rectangle**: \
![Axis of transformation 57][axis57] \
Training Axis 58, **Parallelogram & z-Rectangle**: \
![Axis of transformation 58][axis58] \
Training Axis 67, **x-Rectangle & y-Rectangle**: \
![Axis of transformation 67][axis67]

</section>

 <section class="zippy open">
The abbreviations used for the parameters that define each object are explained
  below:

1.  **`sds`: integer scalar [-]**. The number of edges used to
  generate the regular 2D shape. By default, every 2D shape is a regular one,
  which means that it has equal sides and equal angles.

2.  **`shr`: real-valued [%]**. The percentage of 'shrinking' applied to the 2D
    shape. In the 2D plane, the shape is y-shrunk along the x-axis according to
    the formula: $$ y_{new} = (1 - \mbox{shr} \cdot x)*y_{old} $$.

3.  **`drf`: real-valued [deg]**. The amount of
    [draft](https://en.wikipedia.org/wiki/Draft_\(engineering\)) used in the
    extrusion. Not used in this dataset, drf=0 for every object.

4.  **`hlw`: real-valued [%]**. The percentage of material to be removed from
    each of the generated faces of the solid (to make it hollow). In this dataset,
    no hollow object was created by this transformation, hlw=0 for every object.

5.  **`shx`: real-valued [deg]**. The solid angle of the extrusion axis along
    the x-axis.

6.  **`shy`: real-valued [deg]**. The solid angle of the extrusion axis along
    the y-axis.

7.  **`scx`: real-valued [mm]**. The global x-scale.

8.  **`scy`: real-valued [mm]**. The global y-scale.

9.  **`scz`: real-valued [mm]**. The global z-scale.

The values of these parameters are constrained in such a way that each valid
RGB-shape has a one-to-one mapping with the associated parameters.

</section>


### Test Triplets
<section class="zippy close">
We also provide the 5 fixed test triplets we have used during evaluation in our
  work.

- `rgb_test_triplet1`: `('r3', 's0', 'b2')`
- `rgb_test_triplet2`: `('r5', 'g2', 'b3')`
- `rgb_test_triplet3`: `('r6', 'g3', 'b5')`
- `rgb_test_triplet4`: `('s0', 'g5', 'b6')`
- `rgb_test_triplet5`: `('r2', 'g6', 's0')`

![Triplets][test_triplets]
</section>


## Usage example
<section class="zippy open">
RGB-objects are implemented in [`props/rgb_objects`][rgb_object_package] package,
  can be initialized and used as following.


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

# A list of all available objects could be accessed with
ids_list = rgb_object.RGB_OBJECTS_FULL_SET

```

Objects could be created individually as in the above snippet. Alternatively,
some object triplets have been defined in `rgb_object.PROP_TRIPLETS`.

```python
object_set = 'rgb_test_triplet1'
_, obj_ids = rgb_object.PROP_SETS[object_set]
task_props = []
for i, obj_id in enumerate(objects):
  prop = rgb_object.RgbObjectProp(
        obj_id=obj_id, color=color_set[i], name=f'rgb_object_{obj_id}')
  task_props.append(prop)
```
</section>

## Citing

<section class="zippy open">
If you use `rgb_objects` in your work, please cite the accompanying [paper][pick_and_place_paper]:

```bibtex
@inproceedings{lee2021rgbstacking,
    title={Beyond Pick-and-Place: Tackling Robotic Stacking of Diverse Shapes},
    author={Alex X. Lee and
            Coline Devin and
            Yuxiang Zhou and
            Thomas Lampe and
            Konstantinos Bousmalis and
            Jost Tobias Springenberg and
            Arunkumar Byravan and
            Abbas Abdolmaleki and
            Nimrod Gileadi and
            David Khosid and
            Claudio Fantacci and
            Jose Enrique Chen and
            Akhil Raju and
            Rae Jeong and
            Michael Neunert and
            Antoine Laurens and
            Stefano Saliceti and
            Federico Casarini and
            Martin Riedmiller and
            Raia Hadsell and
            Francesco Nori},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2021},
    url={https://openreview.net/forum?id=U0Q8CrtBJxJ}
}
```
</section>


<!-- Hyperlinks  -->
[pick_and_place_paper]: https://openreview.net/forum?id=U0Q8CrtBJxJ
[rgb_object_package]: https://github.com/deepmind/dm_robotics/tree/main/py/manipulation/props/rgb_objects
[rgb_stacking]: https://github.com/deepmind/rgb_stacking/tree/main/README.md
[meshes_dir]: https://github.com/deepmind/dm_robotics/tree/main/py/manipulation/props/rgb_objects/assets/rgb_v1/meshes
[meshes_heldout_dir]: https://github.com/deepmind/dm_robotics/tree/main/py/manipulation/props/rgb_objects/assets/rgb_v1/meshes/heldout
[meshes_test_dir]: https://github.com/deepmind/dm_robotics/tree/main/py/manipulation/props/rgb_objects/assets/rgb_v1/meshes/test_triplets
[meshes_train_dir]: https://github.com/deepmind/dm_robotics/tree/main/py/manipulation/props/rgb_objects/assets/rgb_v1/meshes/train
[object_disk]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/rgb_objects_disk.png?raw=true
[rgb_benchmark]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/rgb_benchmark.png?raw=true
[test_triplets]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_triplets.gif?raw=true
[axis2]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis2.gif?raw=true
[axis3]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis3.gif?raw=true
[axis5]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis5.gif?raw=true
[axis6]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis6.gif?raw=true
[axis23]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis23.gif?raw=true
[axis25]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis25.gif?raw=true
[axis26]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis26.gif?raw=true
[axis35]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis35.gif?raw=true
[axis36]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis36.gif?raw=true
[axis37]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis37.gif?raw=true
[axis38]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis38.gif?raw=true
[axis56]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis56.gif?raw=true
[axis57]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis57.gif?raw=true
[axis58]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis58.gif?raw=true
[axis67]: https://github.com/deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/doc/images/tile_axis67.gif?raw=true
[OnShapeAPI]: https://onshape-public.github.io/docs/
[meshlab]: https://www.meshlab.net
[STL format]: https://en.wikipedia.org/wiki/STL_(file_format)

