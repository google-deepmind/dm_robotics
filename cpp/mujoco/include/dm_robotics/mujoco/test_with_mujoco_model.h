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

#ifndef DM_ROBOTICS_MUJOCO_TEST_WITH_MUJOCO_MODEL_H_
#define DM_ROBOTICS_MUJOCO_TEST_WITH_MUJOCO_MODEL_H_

#include <memory>

#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics::testing {

// Fixture for MuJoCo tests with an mjModel and mjData object.
class TestWithMujocoModel : public ::testing::Test {
 private:
  struct MjModelDeleter {
    void operator()(mjModel* p) const { mj_deleteModel(p); }
  };
  struct MjDataDeleter {
    void operator()(mjData* p) const { mj_deleteData(p); }
  };

 protected:
  // Loads a new MuJoCo XML into `model_` and re-initializes `data_`.
  void LoadModelFromXmlPath(absl::string_view path_to_xml);

  std::unique_ptr<mjModel, MjModelDeleter> model_;
  std::unique_ptr<mjData, MjDataDeleter> data_;
};

}  // namespace dm_robotics::testing

#endif  // DM_ROBOTICS_MUJOCO_TEST_WITH_MUJOCO_MODEL_H_
