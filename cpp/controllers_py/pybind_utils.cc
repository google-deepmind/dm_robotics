// Copyright 2020 DeepMind Technologies Limited.
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

#include "pybind_utils.h"  // controller

#include <errno.h>
#include <string.h>
#include <sys/stat.h>

#include <cstdint>
#include <string>

#include "dm_robotics/support/logging.h"
#include "absl/strings/substitute.h"
#include "mujoco/mujoco.h"
#include "pybind11/pybind11.h"  // pybind

namespace dm_robotics::internal {
namespace {

namespace py = pybind11;

constexpr char kMjModelClassName[] = "MjModel";
constexpr char kMjDataClassName[] = "MjData";

// Returns a pointer to the underlying C object pointed to by
// `handle`. Note that this does not increment the reference count.
template <class T>
T* GetPointer(py::handle handle) {
  return reinterpret_cast<T*>(handle.attr("_address").cast<std::uintptr_t>());
}

// Returns a pointer to a MuJoCo mjModel or mjData.
//
// Supports objects from either the `dm_control` library or the `mujoco`
// library.
template <class T>
T* GetMujocoPointer(py::handle mujoco_obj) {
  // If it has a `ptr` attribute, assume it is a `dm_control` object.
  if (py::hasattr(mujoco_obj, "ptr")) {
    return GetPointer<T>(mujoco_obj.attr("ptr"));
  }

  // Otherwise, assume it's a `mujoco` object.
  return GetPointer<T>(mujoco_obj);
}

}  // namespace

void RaiseRuntimeErrorWithMessage(absl::string_view message) {
  PyErr_SetString(PyExc_RuntimeError, std::string(message).c_str());
  throw py::error_already_set();
}

// Helper function for getting an mjModel object from a py::handle.
const mjModel* GetmjModelOrRaise(py::handle obj) {
  const std::string class_name =
      obj.attr("__class__").attr("__name__").cast<std::string>();
  if (class_name != kMjModelClassName) {
    RaiseRuntimeErrorWithMessage(absl::Substitute(
        "GetmjModelOrRaise: the class name of the argument [$0] does not match "
        "the expected type name for mjModel [$1].",
        class_name, kMjModelClassName));
  }
  return GetMujocoPointer<mjModel>(obj);
}

// Helper function for getting an mjData object from a py::handle.
const mjData* GetmjDataOrRaise(py::handle obj) {
  const std::string class_name =
      obj.attr("__class__").attr("__name__").cast<std::string>();
  if (class_name != kMjDataClassName) {
    RaiseRuntimeErrorWithMessage(absl::Substitute(
        "GetmjDataOrRaise: the class name of the argument [$0] does not match "
        "the expected type name for mjData [$1].",
        class_name, kMjDataClassName));
  }
  return GetMujocoPointer<mjData>(obj);
}

}  // namespace dm_robotics::internal
