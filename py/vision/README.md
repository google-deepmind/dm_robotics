# DeepMind Robotics Vision

This package provides functions, methods and ROS nodes for colour-based blob
detection and 2D-to-3D (from pixel to 3D Cartesian coordinates) triangulation.

## Rationale

The goal of this package is to:

 1. detect uniquely coloured objects;
 2. evaluate their image (pixel space) barycenter;
 3. estimate their 3D Cartesian barycenter from multiple images via triangulation.

To run the triangulation algorithm, you need to run the blob detection algorithm
on at least two cameras. In ROS, this translates to running two blob detection
ROS nodes on two cameras and then the triangulation
ROS node on the published pixel barycenters.

## Run the ROS nodes

To run the a blob detector ROS node, simply

```python
python launch_blob_detector.py --camera=<camera name>
```

To run the triangulation ROS node

```python
python launch_blob_triangulation.py
```

In order to set up the ROS node for your needs, you need to edit the following
configuration files:

Name  | Description
:---: | :----------
`config_blob_detector.py` | Configure the blob detector ROS node.
`config_blob_triangulation.py` | Configure the blob-based triangulation ROS node.
`robot_config.py` | Robot-specific configuration file.

## Configuration files

Each configuration file exposes a getter function that returns a data class
containing all the parameters needed to run the algorithm/ROS nodes.

### `config_blob_detector.py`

This modules returns an instance of `BlobDetectorConfig` that defines the
following parameters:

 - **node_name**: the name of the ROS node.
 - **rate**: the desired frame rate.
 - **input_queue_size**: the input data queue size (see ROS documentation).
 - **output_queue_size**: the output data queue size (see ROS documentation).
 - **camera_namespaces**: a camera name to ROS namespace mapping.
 - **masks**: `(u, v)` coordinates defining closed regions of interest in the
     image where the blob detector will not look for blobs. Masks are defined
     as a list of list of points, as required by the `pts` argument of OpenCV's
     [`fillPoly` function](http://shortn/_NFOcqaZ2wq).
 - **scale**: image scaling factor to increase speed and frame rate.
 - **min_area**: minimum size in pixels above which a blob is deemed valid.

To get started, the two parameters to change are `camera_namespaces` and
`masks`. They both need to have the same key, which corresponds to a camera
of your choice, which you will need to pass to the `--camera` parameter of
`launch_blob_detector.py`.

The `camera_namespaces` is, as the name and description suggests, the base name
of a camera ROS topic. The blob detector will automatically append to it
`/image_raw` and the resulting ROS topic name will be used to collect images and
run the blob detector.

The `masks` must define a mask for each camera. It can be trivially set to
either an empty list or `None`, unless you have a camera configuration
where you know a priori where to not look for an object in the image (e.g. in
settings where the camera is fixed and the robot workspace is well defined).

### `robot_config.py`

Before jumping into the triangulation configuration file, we need to set up the
configuration file for the robot. In fact, the triangulation algorithm depends
on some robot parameters and this file collects them.

This modules returns an instance of `RobotConfig` that defines the
following parameters:

 - **name**: unique robot name.
 - **cameras**: collection of cameras.
 - **basket_center**: center of playground relative to the robot base frame in
     the xy plane.
 - **basket_height**: displacement of the playground from the robot base frame.
 - **base_frame_name**: the name (or id) that identifies the robot ROS base
     frame.

As you can see from the provided config file, the class `RobotType` can provide
multiple `RobotConfig`, each with a unique name, e.g. `STANDARD_SAWYER`.
When editing or adding a `RobotConfig`, it is important to note that the camera
names must be the same than the ones used in `config_blob_detector.py`'s
`camera_namespaces`. Also note that only the camera extrinsic parameters are
required as the triangulation algorithm will collect the intrinsics from the
camera ROS topic automatically.


### `config_blob_triangulation.py`

This modules returns an instance of `BlobTriangulationConfig` that defines the
following parameters:

 - **node_name**: the name of the ROS node.
 - **rate**: the desired frame rate.
 - **input_queue_size**: the input data queue size (see ROS documentation).
 - **output_queue_size**: the output data queue size (see ROS documentation).
 - **fuse_tolerance**: time in seconds after which data is considered outdated
     and not used for triangulation.
 - **extrinsics**: extrinsic camera parameters.
 - **limits**: Cartesian limits over which points are considered as outliers.
 - **base_frame**: the name of the robot base frame.
 - **deadzones**: additional Cartesian limits excluding volumes of the robot
     operative space where points are discarded.

If you configured the `robot_config.py`, most of the information can be
collected from there and possibly elaborated to populate `extrinsics`,
`base_frame`, `limits`, and `deadzones`. While the first two are quite
straightforward, the latter two requires a bit more consideration. Both are used
by the triangulation algorithm to validate and possibly reject triangulated
points. However, the two parameters carry different semantics. `limits` is
used to describe the robot's workspace volume, while `deadzones` are forbidden
zones. Points are considered valid when inside the `limits` and outside
`deadzones`.

## Launch files

This section describes the launch files and their input arguments for both blob detector and triangulator ROS nodes.

### `launch_blob_detector.py`

Note: the config file is automatically loaded and used.

The arguments are:

 - camera: the camera to use. Must be one of the keys in `cameras` in the
     configuration file.
 - props: defaults to `RED`, `GREEN` and `BLUE` objects defined in
     `blob_tracker_object_defs.py`. The names of the props to track.
 - visualize: defaults to `False`. Whether to publish helper images of the
     detected blobs or not.
 - toolkit: defaults to `False`. Whether to display a YUV GUI toolkit to find
     good YUV parameters to detect blobs. Sets `visualize = True`.

### `launch_blob_triangulation.py`

Note: the config file is automatically loaded and used.

The arguments are:

 - robot: defaults to `STANDARD_SAWYER`. The name of the robot. Must be one of
     the enums in the robot configuration file.
 - props: defaults to `RED`, `GREEN` and `BLUE` objects defined in
     `blob_tracker_object_defs.py`. The names of the props to track.
