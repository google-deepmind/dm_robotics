# DeepMind Robotics Transformations

Transformations is a pure python library for rigid-body transformations
including velocities and forces.

The objectives for this library are **simplicity** and **comprehensiveness**
across all canonical representations (euler, axis-angle, quaternion,
homogeneous matrices).


## Supported conversions:
* Quaternion to Rotation matrix, Axis-angle and Euler-angle
* Axis-angle to Quaternion, Rotation matrix and Euler-angle
* Rotation matrix to Quaternion, Axis-angle and Euler-angle
* Euler-angle to Quaternion, Rotation matrix and Axis-angle

## Quaternions:
Quaternions are represented with the scalar part (w) first, e.g.

```python
identity_quaternion = np.asarray([1, 0, 0, 0])  # w, i, j, k
```

Supported quaternion operations:

* Difference
* Distance
* Multiplication
* Inverse
* Conjugate
* Logarithm and Exponent
* Slerp (spherical linear interpolation)
* Rotation of a vector by a quaternion.

## Euler-angles
All 24 from-euler orderings are supported.
7 of 24 to-euler orderings are supported.

## Transforms
This library supports force and velocity transforms.

## Usage Example

```python
from dm_robotics.transformations import transformations as tr

# Convert a pose, euler angle into a homogeneous matrix (a 4x4 matrix):
hmat = tr.poseuler_to_hmat(
        np.array([x, y, z, rot_x, rot_y, rot_z]), 'XYZ')

# Convert the homogeneous matrix to a twist (a 6 vector):
twist = tr.hmat_to_twist(hmat)
```

