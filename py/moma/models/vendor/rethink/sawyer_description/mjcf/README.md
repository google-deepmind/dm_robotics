# Sawyer robot

This directory contains MuJoCo MJCF models of the Rethink Sawyer robot and
several associated actuator modules.

## Original models

The MJCF models were automatically generated from the original URDF models,
which can be found together with their assets in:
https://github.com/RethinkRobotics/sawyer_robot/tree/master/sawyer_description/.

## Modifications

The following manual changes were made:

* Approximate geometries using primitive geoms were commented out.
* Dummy bodies (with no DoFs) were removed.
* Some collisions were explicitly excluded.
* Some MuJoCo-specific properties were added (sensors, contact parameters).
