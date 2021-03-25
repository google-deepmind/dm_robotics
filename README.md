# DeepMind Robotics

Libraries, tools and tasks created and used at DeepMind Robotics.

## Development quickstart

### Building package distributions and testing the Python libraries

#### Dependencies
The Python packages depend on each other in a chain:

*  `moma` depends on `agentflow`,
*  `agentflow` depends on `geometry`,
*  `geometry` depends on `transformations`.

`moma` also depends on the C++ controllers and the Python bindings to them.

The Python libraries are tested with `tox`, the C++ code is built and tested
with `cmake`.

We use Tox's `distshare` mechanism to share the built source distribution
packages between the packages.

To build and test the libraries, run `build.sh`.  This script assumes:

*  MuJoCo is installed and licensed.
*  dm_control is installed.
*  cmake version >= 3.20.2 is installed.
*  Python 3.8 and its system headers are installed.
*  GCC version 9 or later is installed.
*  numpy is installed.
