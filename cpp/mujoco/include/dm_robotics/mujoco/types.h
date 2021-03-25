// Copyright 2020 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DM_ROBOTICS_MUJOCO_TYPES_H_
#define DM_ROBOTICS_MUJOCO_TYPES_H_

#include <string>
#include <utility>

#include "absl/container/btree_set.h"

namespace dm_robotics {

// A GeomGroup defines a set of geoms by their names.
using GeomGroup = absl::btree_set<std::string>;

// A CollisionPair defines a set of two geom groups that should avoid each
// other, i.e. every geom of the first group should avoid every geom of the
// second group, and vice versa.
using CollisionPair = std::pair<GeomGroup, GeomGroup>;

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_MUJOCO_TYPES_H_
