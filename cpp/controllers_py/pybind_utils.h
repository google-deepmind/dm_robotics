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

#ifndef DM_ROBOTICS_PY_CONTROLLERS_PYBIND_UTILS_H_
#define DM_ROBOTICS_PY_CONTROLLERS_PYBIND_UTILS_H_

#include "absl/strings/string_view.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "pybind11/pytypes.h"  // pybind

namespace dm_robotics::internal {

// Raises a `RuntimeError` exception in Python with the message `message`.
void RaiseRuntimeErrorWithMessage(absl::string_view message);

// Loads a new MjLib object and activates it via dm_control's Python
// infrastructure. The user takes ownership of the allocated object. The
// recommended usage is to use this through the static-in-a-function pattern to
// instantiate a global singleton that is never deallocated.
//
// This is a helper-function for re-using the libmujoco.so and license file used
// by dm_control. It must only be used if dm_control is available; it will
// not work for binaries that are missing the symbols from the Python
// interpreter that come bundled with dm_control, causing the program to
// crash.
const MjLib* LoadMjLibFromDmControl();

// Extracts an mjModel pointer from a Python handle to a dm_control `MjModel`
// object. Raises a `RuntimeError` exception in Python if it fails to extract
// an mjModel object from the handle.
//
// Note that this does not increment the reference count.
const mjModel* GetmjModelOrRaise(pybind11::handle obj);

// Extracts an mjData pointer from a Python handle to a dm_control `MjData`
// object. Raises a `RuntimeError` exception in Python if it fails to extract
// an mjData object from the handle.
//
// Note that this does not increment the reference count.
const mjData* GetmjDataOrRaise(pybind11::handle obj);

}  // namespace dm_robotics::internal

#endif  // DM_ROBOTICS_PY_CONTROLLERS_PYBIND_UTILS_H_
