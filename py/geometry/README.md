# Overview

`geometry.py` provides primitives for dealing with scene and robot geometry.

## Main Concepts

### Geometrical types

*   `Pose`: The 6D position and orientation of a body. Construction methods:

  ```python
  Pose(position=[x, y, z], quaternion=[w, i, j, k])
  Pose.from_hmat(homogeneous_matrix)
  Pose.from_poseuler(poseuler=[x,y,z, p,r,w], ordering="XYZ")
  ```
*   `Wrench` and `WrenchStamped` 6D force-torque.
*   `Twist` and `TwistStamped` 6D velocity.
*   `Accel` and `AccelStamped` 6D acceleration.

For debugging and visualization, `Pose` and `PoseStamped` objects can be
constructed with a `name` parameter.

### Frames and Stamped types

All quantities here come in plain and 'Stamped' varieties. A stamped quantity
(E.g. `TwistStamped`) is always defined as a plain type (E.g. `Twist`) with
respect to a `frame`.  A `frame` is something that can be evaluated
to a world-pose -- either a `PoseStamped` or a `Grounding`.  A grounding is 
optional (i.e. can use None to indicate world-frame).  This library supports
different grounding back-ends, e.g. mujoco-elements or observation-dicts.

For example, the pose of a plug in a robot gripper could be expressed as a
`PoseStamped` with a combination of the `Pose` of the gripper in world frame
and the pose of the plug with respect to the gripper.

A frame for a stamped quantity can be:

*   Another stamped quantity,
*   A "grounding" (a user chosen type).
   *   A user supplied `Physics` instance must be able to provide the world
       `Pose` of a grounding.  The library defined `MujocoPhysics` does this
       for `mjcf.Element` instances.
*   `None`, in which case, world frame is assumed.

Therefore in the above example, the first quantity might come from a mujoco
model (possibly synchronized with the real world), and the second quantity
might be known a priori or estimated by vision.

```python
gripper_in_world = PoseStamped(pose=None, frame=synced_gripper_mjcf_body)
plug_in_gripper = Pose(...)  # From vision pose model.
plug_pose = PoseStamped(plug_in_gripper, frame=gripper_in_world)
```

(A more realistic example would have the pose of the plug in the *camera* frame
then either having the camera in the kinematic tree of the robot (attached to
the robot) or having it and the robot in world frame)

Then the pose of the *plug* in the *world* frame can be calculated as:

```python
plug_in_world: Pose = plug_pose.get_world_pose()
```

### Immutability

All types in `geometry.py` are immutable. You can create modified copies of
existing values with the `replace` and `with_*` methods. `replace` is similar in
spirit to `collections.namedtuple._replace`, these methods return new objects.

### PoseStamped

This is the stamped version of a Pose.

*   Its frame hierarchy can be flattened with `to_world`, which returns a
    `PoseStamped` in world frame.
*   The `Pose` relative to another frame is returned from `get_relative_pose`.
*   The `Pose` relative to the world frame is returned from `get_world_pose`.

So, in the case of a gripper in camera frame which is not attached to the robot,
we could `to_world()` either the gripper or plug `PoseStamped`, giving a new
frame and then calculate thier relative pose (plug relative to robot) with
`get_relative_pose`.

### HybridPoseStamped

A `PoseStamped` where the position or orientation can be overriden. This is
useful to express the idea of (for example) the position of the gripper with
world orientation to allow for more intuitive operator control of the gripper.
