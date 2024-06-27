# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data types for geometry."""

import chex
from dm_robotics.geometry.jax_geometry import frame_geometry

# There are two types of data arrangement:
#
# A Raster is a (..., C) array of spatially aranged vectors.
# A Cloud is a (N, C) array of arbitrarily arranged vectors.
#
# PointXXX means the values are points in C-dimensional space.
#
# Example:
# - An (N, C) array of opaque values is a Cloud.
# - If each row is a point in C-dimensional space then it is also a PointCloud.
# - If the ordering in dimension 0 is spatial then it is also a Raster.
# - If the ordering in dimension 0 is spatial and each row is a vector in
#   C-dimensional space then it is also a PointRaster.
#
# Example:
# - A (H, W, C) array of opaque values is a Raster.
# - Because there is more than one leading dimension the ordering is assumed to
#   be spatial. This value is never a Cloud, and we have no concept for
#   multidimensional arbitrary orders.
# - If each location in the Raster is a point in C-dimensional space then it is
#   also a PointRaster.
# - If C=1 or C=3 and the values are all in the range [0, 1] then it is also an
#   Image.
# - If C=1 and the avlues are all in the range [0, inf] then it is also a
#   DepthMap.
#
# Points are always stored in (x, y, z, ...) order, whereas the spatial
# arrangement in a Raster is in (y, x, z, ...) order. This means that if you
# have a coordinate [x, y] that is a point in the space sampled by a Raster of
# shape (H, W, C) then the coord refers to raster[y, x].
#
# Example:
# Suppose you have a Raster `raster` of shape (H, W, C), and a PointCloud
# `points` of shape (N, 2) of type int32 that contains points in the Raster
# coordinate system. To extract the referenced points from the raster you could
# write:
#
# ```
# x, y = points[..., 0], points[..., 1]
# values = raster[y, x]
# ```
#
# If `raster` had shape (H, W, D, C) and `points` had shape `(N, 3)` then you
# could write:
#
# ```
# x, y, z = points[..., 0], points[..., 1], points[..., 2]
# values = raster[y, x, z]
# ```
#
# Notice how in both cases the order of the first two coordinates in `points`
# are reversed from the order in which they are used to index `raster`.
#
# NB: The above example assumes `points` contains integers so its values can be
# used directly as indexes into `raster`. This is valid, but it is not a
# requirement of the PointCloud type.

# A Raster is a (..., C) array of values in spatial order.
Raster = chex.Array

# A point in C-d space.
Point = chex.Array

# A Cloud is a (N, C) array of values in arbitrary order.
Cloud = chex.Array

# A PointRaster is a Raster whose elements are points in C-d Space.
PointRaster = Raster

# A PointCloud is a Cloud whose elements are points in C-d space.
PointCloud = Cloud

# An Image is a Raster of shape (H, W, {1, 3}) with dtype=float32 and values in
# the range [0, 1].
Image = Raster

# A Segmentation is a raster of shape (H, W, 2) with dtype=int32. Value range
# and semantics depend on dataset.
Segmentation = Raster

# A DepthMap is a raster of shape (H, W, 1) with dtype=float32 and values in the
# range [0, inf].
DepthMap = Raster

# A Frame is a pose represented as a local->world transform.
Frame = frame_geometry.Pose
