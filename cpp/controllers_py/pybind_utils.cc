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

#include <dlfcn.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>

#include <string>

#include "dm_robotics/support/logging.h"
#include "absl/strings/substitute.h"
#include "dm_robotics/mujoco/mjlib.h"
#include "pybind11/pybind11.h"  // pybind

namespace dm_robotics::internal {
namespace {

namespace py = pybind11;

constexpr char kDmControlMjModelClassName[] = "MjModel";
constexpr char kDmControlMjDataClassName[] = "MjData";

// Returns a ctypes module for use through pybind.
py::module GetCtypesModule() {
  static PyObject* const ctypes_module_ptr = []() -> PyObject* {
    py::module ctypes_module = py::module::import("ctypes");
    PyObject* ctypes_module_ptr = ctypes_module.ptr();
    Py_XINCREF(ctypes_module_ptr);
    return ctypes_module_ptr;
  }();
  return py::reinterpret_borrow<py::module>(ctypes_module_ptr);
}

// Returns a pointer to the underlying object pointed to by `ctypes_obj`.
// Note that does not increment the reference count.
template <class T>
T* GetPointer(py::handle mjwrapper_object) {
#ifdef DM_ROBOTICS_USE_NEW_MUJOCO_BINDINGS
  return reinterpret_cast<T*>(
      mjwrapper_object.attr("_address").cast<std::uintptr_t>());
#else
  auto ctypes_module = GetCtypesModule();
  return reinterpret_cast<T*>(
      ctypes_module.attr("addressof")(mjwrapper_object.attr("contents"))
          .cast<std::uintptr_t>());
#endif
}

// Returns a pointer to a MuJoCo mjModel or mjData from the dm_control wrappers
// around these objects. The dm_control wrappers are such that the `ptr`
// attributes return a ctypes object bound to underlying MuJoCo native type.
//
// Raises a `RuntimeError` exception in Python if the handle does not contain a
// `ptr` attribute.
template <class T>
T* GetMujocoPointerFromDmControlWrapperHandle(py::handle dm_control_wrapper) {
  if (!py::hasattr(dm_control_wrapper, "ptr")) {
    RaiseRuntimeErrorWithMessage(
        "GetMujocoPointerFromDmControlWrapperHandle: handle does not have a "
        "`ptr` attribute. This function assumes that dm_control wrappers "
        "around mjModel and mjData contain a `ptr` attribute with the MuJoCo "
        "native type.");
  }
  return GetPointer<T>(dm_control_wrapper.attr("ptr"));
}

}  // namespace

void RaiseRuntimeErrorWithMessage(absl::string_view message) {
  PyErr_SetString(PyExc_RuntimeError, std::string(message).c_str());
  throw py::error_already_set();
}

const MjLib* LoadMjLibFromDmControl() {
  py::gil_scoped_acquire gil;
  // Get the path to the mujoco library.
  const py::module mujoco(py::module::import("mujoco"));
  const auto path = mujoco.attr("__path__").cast<std::vector<std::string>>();
  const std::string version = mujoco.attr("__version__").cast<std::string>();
  if (path.empty()) {
    RaiseRuntimeErrorWithMessage("mujoco.__path__ is empty");
    return nullptr;
  }
  const std::string dso_path = path[0] + "/libmujoco.so." + version;

  struct stat buffer;
  if (stat(dso_path.c_str(), &buffer) != 0) {
    RaiseRuntimeErrorWithMessage(absl::Substitute(
        "LoadMjLibFromDmComtrol: Cannot access mujoco library file "
        "$0, error: $1",
        dso_path, strerror(errno)));
    return nullptr;
  }
  py::print("Loading mujoco from " + dso_path);

  // Create the MjLib object by dlopen'ing the DSO.
  return new MjLib(dso_path, RTLD_NOW);
}

// Helper function for getting an mjModel object from a py::handle.
const mjModel* GetmjModelOrRaise(py::handle obj) {
  const std::string class_name =
      obj.attr("__class__").attr("__name__").cast<std::string>();
  if (class_name != kDmControlMjModelClassName) {
    RaiseRuntimeErrorWithMessage(absl::Substitute(
        "GetmjModelOrRaise: the class name of the argument [$0] does not match "
        "the expected dm_control type name for mjModel [$1].",
        class_name, kDmControlMjModelClassName));
  }
  return GetMujocoPointerFromDmControlWrapperHandle<mjModel>(obj);
}

// Helper function for getting an mjData object from a py::handle.
const mjData* GetmjDataOrRaise(py::handle obj) {
  const std::string class_name =
      obj.attr("__class__").attr("__name__").cast<std::string>();
  if (class_name != kDmControlMjDataClassName) {
    RaiseRuntimeErrorWithMessage(absl::Substitute(
        "GetmjDataOrRaise: the class name of the argument [$0] does not match "
        "the expected dm_control type name for mjData [$1].",
        class_name, kDmControlMjDataClassName));
  }
  return GetMujocoPointerFromDmControlWrapperHandle<mjData>(obj);
}

}  // namespace dm_robotics::internal
