#include "cartesian_6d_to_joint_velocity_mapper_python_helpers.h"  // controller

#include <array>
#include <chrono>  // NOLINT(build/c++11)
#include <csignal>
#include <optional>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper.h"
#include "dm_robotics/least_squares_qp/core/utils.h"
#include "dm_robotics/mujoco/types.h"
#include "mujoco/mujoco.h"
#include "pybind_utils.h"  // controller
#include "pybind11/cast.h"  // pybind
#include "pybind11/pytypes.h"  // pybind

namespace dm_robotics::internal {
namespace {
namespace py = ::pybind11;

// Converts a non-ok `status` into a Python exception.
//
// Internal comment.
void RaiseIfNotOk(const absl::Status& status) {
  if (status.ok()) return;
  internal::RaiseRuntimeErrorWithMessage(status.message());
}

// Helper function for throwing if `ValidateParameters` fails, or returning a
// reference to the parameter otherwise.
const Cartesian6dToJointVelocityMapper::Parameters& RaiseIfNotValid(
    const Cartesian6dToJointVelocityMapper::Parameters& params) {
  RaiseIfNotOk(Cartesian6dToJointVelocityMapper::ValidateParameters(params));
  return params;
}

}  // namespace

PyCartesian6dToJointVelocityMapperParameters::
    PyCartesian6dToJointVelocityMapperParameters()
    : py_model_(py::cast(nullptr)) {
  params_.model = nullptr;
}

const Cartesian6dToJointVelocityMapper::Parameters&
PyCartesian6dToJointVelocityMapperParameters::RawParameters() const {
  return params_;
}

PyCartesian6dToJointVelocityMapperParameters
PyCartesian6dToJointVelocityMapperParameters::FromRawParameters(
    const Cartesian6dToJointVelocityMapper::Parameters& params) {
  PyCartesian6dToJointVelocityMapperParameters py_params;
  py_params.params_ = params;
  // Lib and model will not be valid. Leave them as they have been constructed.
  return py_params;
}

py::object PyCartesian6dToJointVelocityMapperParameters::GetModel() {
  return py_model_;
}

void PyCartesian6dToJointVelocityMapperParameters::SetModel(py::object model) {
  py_model_ = model;
  params_.model = internal::GetmjModelOrRaise(model);
}

std::vector<int> PyCartesian6dToJointVelocityMapperParameters::GetJointIds() {
  return std::vector<int>(params_.joint_ids.begin(), params_.joint_ids.end());
}

void PyCartesian6dToJointVelocityMapperParameters::SetJointIds(
    const std::vector<int>& val) {
  params_.joint_ids = absl::btree_set<int>(val.begin(), val.end());
}

int PyCartesian6dToJointVelocityMapperParameters::GetObjectType() {
  return static_cast<int>(params_.object_type);
}

void PyCartesian6dToJointVelocityMapperParameters::SetObjectType(int val) {
  params_.object_type = static_cast<mjtObj>(val);
}

std::chrono::microseconds
PyCartesian6dToJointVelocityMapperParameters::GetIntegrationTimestep() {
  return absl::ToChronoMicroseconds(params_.integration_timestep);
}

void PyCartesian6dToJointVelocityMapperParameters::SetIntegrationTimestep(
    std::chrono::microseconds val) {
  params_.integration_timestep = absl::FromChrono(val);
}

std::vector<PybindCollisionPair>
PyCartesian6dToJointVelocityMapperParameters::GetCollisionPairs() {
  std::vector<PybindCollisionPair> collision_pairs;
  for (const auto& collision_pair : params_.collision_pairs) {
    PybindGeomGroup first(collision_pair.first.begin(),
                          collision_pair.first.end());
    PybindGeomGroup second(collision_pair.second.begin(),
                           collision_pair.second.end());
    collision_pairs.push_back(PybindCollisionPair(first, second));
  }
  return collision_pairs;
}

void PyCartesian6dToJointVelocityMapperParameters::SetCollisionPairs(
    const std::vector<PybindCollisionPair>& collision_pairs) {
  for (const auto& collision_pair : collision_pairs) {
    GeomGroup first(collision_pair.first.begin(), collision_pair.first.end());
    GeomGroup second(collision_pair.second.begin(),
                     collision_pair.second.end());
    params_.collision_pairs.insert(CollisionPair(first, second));
  }
}

PyCartesian6dToJointVelocityMapper::PyCartesian6dToJointVelocityMapper(
    const internal::PyCartesian6dToJointVelocityMapperParameters& py_params)
    : py_model_(py_params.py_model_),
      mapper_(RaiseIfNotValid(py_params.params_)),
      enable_nullspace_control_(py_params.params_.enable_nullspace_control),
      num_dof_(py_params.params_.joint_ids.size()) {}

std::vector<double> PyCartesian6dToJointVelocityMapper::ComputeJointVelocities(
    py::handle data,
    const std::array<double, 6>& target_6d_cartesian_velocity) {
  if (enable_nullspace_control_) {
    internal::RaiseRuntimeErrorWithMessage(
        "Cartesian6dToJointVelocityMapper::ComputeJointVelocities: "
        "Function overload without `nullspace_bias` parameter must "
        "not be called when `enable_nullspace_control` is true.");
  }

  // Get solution.
  const mjData& data_ref = *internal::GetmjDataOrRaise(data);
  absl::StatusOr<absl::Span<const double>> solution_or =
      mapper_.ComputeJointVelocities(data_ref, target_6d_cartesian_velocity);

  // Re-raise SIGINT if it was interrupted. This will propagate SIGINT to
  // Python.
  if (solution_or.status().code() == absl::StatusCode::kCancelled) {
    std::raise(SIGINT);
  }

  // Handle error, if any, or return a copy of the solution.
  RaiseIfNotOk(solution_or.status());
  return AsCopy(*solution_or);
}

std::vector<double> PyCartesian6dToJointVelocityMapper::ComputeJointVelocities(
    py::handle data, const std::array<double, 6>& target_6d_cartesian_velocity,
    const std::vector<double>& nullspace_bias) {
  if (!enable_nullspace_control_) {
    internal::RaiseRuntimeErrorWithMessage(
        "Cartesian6dToJointVelocityMapper::ComputeJointVelocities: "
        "Function overload with `nullspace_bias` parameter must "
        "not be called when `enable_nullspace_control` is false.");
  }
  if (nullspace_bias.size() != num_dof_) {
    internal::RaiseRuntimeErrorWithMessage(absl::Substitute(
        "Cartesian6dToJointVelocityMapper::ComputeJointVelocities: "
        "Size of the `nullspace_bias` array [$0] does not match the "
        "number of DoF being controlled [$1].",
        nullspace_bias.size(), num_dof_));
  }

  // Get solution.
  const mjData& data_ref = *internal::GetmjDataOrRaise(data);
  absl::StatusOr<absl::Span<const double>> solution_or =
      mapper_.ComputeJointVelocities(data_ref, target_6d_cartesian_velocity,
                                     nullspace_bias);

  // Re-raise SIGINT if it was interrupted. This will propagate SIGINT to
  // Python.
  if (solution_or.status().code() == absl::StatusCode::kCancelled) {
    std::raise(SIGINT);
  }

  // Handle error, if any, or return a copy of the solution.
  RaiseIfNotOk(solution_or.status());
  return AsCopy(*solution_or);
}

std::vector<double> PyCartesian6dToJointVelocityMapper::ComputeJointVelocities(
    pybind11::handle data,
    const std::array<double, 6>& target_6d_cartesian_velocity,
    const std::optional<std::vector<double>>& nullspace_bias) {
  if (nullspace_bias.has_value()) {
    return ComputeJointVelocities(data, target_6d_cartesian_velocity,
                                  nullspace_bias.value());
  }
  return ComputeJointVelocities(data, target_6d_cartesian_velocity);
}

}  // namespace dm_robotics::internal
