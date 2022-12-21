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

#include "dm_robotics/mujoco/test_with_mujoco_model.h"

#include <memory>
#include <string>

// Internal file library include
// Internal tools library include
#include "dm_robotics/support/status-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
// Internal flag library include
#include "absl/strings/string_view.h"

namespace dm_robotics::testing {

constexpr int kMujocoErrorSize = 1024;

void TestWithMujocoModel::LoadModelFromXmlPath(absl::string_view path_to_xml) {
  // Space for MuJoCo to store an error string (if any). The API guarantees that
  // the string is null-terminated.
  char mujoco_load_error[kMujocoErrorSize];

  std::string path(path_to_xml);
  model_.reset(mjlib_->mj_loadXML(
      path.c_str(), nullptr, mujoco_load_error, kMujocoErrorSize));

  ASSERT_NE(model_, nullptr)
      << "TestWithMujocoModel::LoadModelFromXmlPath: MuJoCo mj_loadXML failed "
         "with the following error: "
      << mujoco_load_error;

  // Reset mjData to match the model.
  data_.reset(mjlib_->mj_makeData(model_.get()));
  ASSERT_NE(data_, nullptr);

  // Ensure mjData fields are physically consistent.
  mjlib_->mj_forward(model_.get(), data_.get());
}

}  // namespace dm_robotics::testing
