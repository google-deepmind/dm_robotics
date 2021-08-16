# DeepMind Robotics
Libraries, tools and tasks created and used at DeepMind Robotics.

## Package overview
| Package | Summary |
| ---- | ---- |
| [Transformations](py/transformations/README.md) | Rigid body transformations |
| [Geometry](py/geometry/README.md) | Scene and Robot geometry primitives |
| [Vision](py/vision/README.md) | Visual blob detection and tracking |
| [AgentFlow](py/agentflow/README.md) | Reinforcement Learning agent composition library |
| [Manipulation](py/manipulation/README.md) | "RGB" object meshes for manipulation tasks |
| [MoMa](py/moma/README.md) | Manipulation environment definition library, for simulated and real robots |
| [Controllers](cpp/controllers/README.md) | QP-optimization based cartesian controller |
| [Controller Bindings](cpp/controllers_py/README.md) | Python bindings for the controller |
| [Least Squares QP](cpp/least_squares_qp/README.md) | QP task definition and solver |


## Installation
These libraries are distributed on PyPI, the packages are:

*  `dm_robotics-transformations`
*  `dm_robotics-geometry`
*  `dm_robotics-vision`
*  `dm_robotics-agentflow`
*  `dm_robotics-manipulation`
*  `dm_robotics-moma`
*  `dm_robotics-controllers`


## Dependencies
`MoMa`, `Manipulation` and `Controllers` depend on MuJoCo, the other packages do not.
See the individual packages for more information on their dependencies.

## Building

To build and test the libraries, run `build.sh`.  This script assumes:

*  MuJoCo is installed and licensed.
*  dm_control is installed.
*  cmake version >= 3.20.2 is installed.
*  Python 3.6 ,3.7 or 3.8 and system headers are installed.
*  GCC version 9 or later is installed.
*  numpy is installed.

The Python libraries are tested with `tox`, the C++ code is built and tested
with `cmake`.

Tox's `distshare` mechanism is used to share the built source distribution
packages between the packages.
