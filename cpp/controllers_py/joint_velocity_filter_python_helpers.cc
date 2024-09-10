#include "joint_velocity_filter_python_helpers.h"  // controller

#include <chrono>  // NOLINT(build/c++11)
#include <csignal>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "dm_robotics/controllers/lsqp/joint_velocity_filter.h"
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
const JointVelocityFilter::Parameters& RaiseIfNotValid(
    const JointVelocityFilter::Parameters& params) {
  RaiseIfNotOk(JointVelocityFilter::ValidateParameters(params));
  return params;
}

}  // namespace

PyJointVelocityFilterParameters::PyJointVelocityFilterParameters()
    : py_model_(py::cast(nullptr)) {
  params_.model = nullptr;
}

const JointVelocityFilter::Parameters&
PyJointVelocityFilterParameters::RawParameters() const {
  return params_;
}

PyJointVelocityFilterParameters
PyJointVelocityFilterParameters::FromRawParameters(
    const JointVelocityFilter::Parameters& params) {
  PyJointVelocityFilterParameters py_params;
  py_params.params_ = params;
  // Lib and model will not be valid. Leave them as they have been constructed.
  return py_params;
}

py::object PyJointVelocityFilterParameters::GetModel() { return py_model_; }

void PyJointVelocityFilterParameters::SetModel(py::object model) {
  py_model_ = model;
  params_.model = internal::GetmjModelOrRaise(model);
}

std::vector<int> PyJointVelocityFilterParameters::GetJointIds() {
  return std::vector<int>(params_.joint_ids.begin(), params_.joint_ids.end());
}

void PyJointVelocityFilterParameters::SetJointIds(const std::vector<int>& val) {
  params_.joint_ids = absl::btree_set<int>(val.begin(), val.end());
}

std::chrono::microseconds
PyJointVelocityFilterParameters::GetIntegrationTimestep() {
  return absl::ToChronoMicroseconds(params_.integration_timestep);
}

void PyJointVelocityFilterParameters::SetIntegrationTimestep(
    std::chrono::microseconds val) {
  params_.integration_timestep = absl::FromChrono(val);
}

std::vector<PybindCollisionPair>
PyJointVelocityFilterParameters::GetCollisionPairs() {
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

void PyJointVelocityFilterParameters::SetCollisionPairs(
    const std::vector<PybindCollisionPair>& collision_pairs) {
  for (const auto& collision_pair : collision_pairs) {
    GeomGroup first(collision_pair.first.begin(), collision_pair.first.end());
    GeomGroup second(collision_pair.second.begin(),
                     collision_pair.second.end());
    params_.collision_pairs.insert(CollisionPair(first, second));
  }
}

PyJointVelocityFilter::PyJointVelocityFilter(
    const internal::PyJointVelocityFilterParameters& py_params)
    : py_model_(py_params.py_model_),
      filter_(RaiseIfNotValid(py_params.params_)),
      num_dof_(py_params.params_.joint_ids.size()) {}

std::vector<double> PyJointVelocityFilter::FilterJointVelocities(
    py::handle data, const std::vector<double>& target_joint_velocities) {
  // Get solution.
  const mjData& data_ref = *internal::GetmjDataOrRaise(data);
  absl::StatusOr<absl::Span<const double>> solution_or =
      filter_.FilterJointVelocities(data_ref, target_joint_velocities);

  // Re-raise SIGINT if it was interrupted. This will propagate SIGINT to
  // Python.
  if (solution_or.status().code() == absl::StatusCode::kCancelled) {
    std::raise(SIGINT);
  }

  // Handle error, if any, or return a copy of the solution.
  RaiseIfNotOk(solution_or.status());
  return AsCopy(*solution_or);
}

}  // namespace dm_robotics::internal
