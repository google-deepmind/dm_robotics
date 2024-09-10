#ifndef DM_ROBOTICS_PY_CONTROLLERS_JOINT_VELOCITY_FILTER_PYTHON_HELPERS_H_
#define DM_ROBOTICS_PY_CONTROLLERS_JOINT_VELOCITY_FILTER_PYTHON_HELPERS_H_

#include <chrono>  // NOLINT(build/c++11)
#include <string>
#include <utility>
#include <vector>

#include "dm_robotics/controllers/lsqp/joint_velocity_filter.h"
#include "pybind11/chrono.h"  // pybind
#include "pybind11/pybind11.h"  // pybind
#include "pybind11/pytypes.h"  // pybind
#include "pybind11/stl.h"  // pybind
// Internal status include

namespace dm_robotics::internal {

using PybindGeomGroup = std::vector<std::string>;
using PybindCollisionPair = std::pair<PybindGeomGroup, PybindGeomGroup>;

// Helper class for binding JointVelocityFilter::Parameters that also allows us
// to keep a Python mjModel object alive as long as the instance is alive.
class PyJointVelocityFilterParameters {
 public:
  PyJointVelocityFilterParameters();

  // `model` property.
  pybind11::object GetModel();

  void SetModel(pybind11::object model);

  // `joint_ids` property.
  std::vector<int> GetJointIds();

  void SetJointIds(const std::vector<int>& val);

  // `integration_timestep` property.
  // Note that in Pybind:
  // * `float`s will be interpreted in seconds and casted to a
  //   `std::chrono::duration`;
  // * `datetime.timedelta`s will be casted to a `std::chrono::duration` with
  //   microsecond precision.
  // For more information refer to:
  // https://pybind11.readthedocs.io/en/stable/advanced/cast/chrono.html
  std::chrono::microseconds GetIntegrationTimestep();

  void SetIntegrationTimestep(std::chrono::microseconds val);

  // `collision_pairs` property.
  std::vector<PybindCollisionPair> GetCollisionPairs();

  void SetCollisionPairs(
      const std::vector<PybindCollisionPair>& collision_pairs);

  // MACRO-based property function definitions.
#define PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(var_name, getter_name, \
                                                    setter_name)           \
  decltype(JointVelocityFilter::Parameters::var_name) getter_name() {      \
    return params_.var_name;                                               \
  }                                                                        \
  void setter_name(                                                        \
      const decltype(JointVelocityFilter::Parameters::var_name)& val) {    \
    params_.var_name = val;                                                \
  }

  // `enable_joint_position_limits` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(enable_joint_position_limits,
                                              GetEnableJointPositionLimits,
                                              SetEnableJointPositionLimits)

  // `joint_position_limit_velocity_scale` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(
      joint_position_limit_velocity_scale, GetJointPositionLimitVelocityScale,
      SetJointPositionLimitVelocityScale)

  // `minimum_distance_from_joint_position_limit` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(
      minimum_distance_from_joint_position_limit,
      GetMinimumDistanceFromJointPositionLimit,
      SetMinimumDistanceFromJointPositionLimit)

  // `enable_joint_velocity_limits` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(enable_joint_velocity_limits,
                                              GetEnableJointVelocityLimits,
                                              SetEnableJointVelocityLimits)

  // `joint_velocity_magnitude_limits` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(joint_velocity_magnitude_limits,
                                              GetJointVelocityMagnitudeLimits,
                                              SetJointVelocityMagnitudeLimits)

  // `enable_collision_avoidance` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(enable_collision_avoidance,
                                              GetEnableCollisionAvoidance,
                                              SetEnableCollisionAvoidance)

  // `use_minimum_distance_contacts_only' property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(
      use_minimum_distance_contacts_only, GetUseMinimumDistanceContactsOnly,
      SetUseMinimumDistanceContactsOnly)

  // `collision_avoidance_normal_velocity_scale` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(
      collision_avoidance_normal_velocity_scale,
      GetCollisionAvoidanceNormalVelocityScale,
      SetCollisionAvoidanceNormalVelocityScale)

  // `minimum_distance_from_collisions` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(minimum_distance_from_collisions,
                                              GetMinimumDistanceFromCollisions,
                                              SetMinimumDistanceFromCollisions)

  // `collision_detection_distance` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(collision_detection_distance,
                                              GetCollisionDetectionDistance,
                                              SetCollisionDetectionDistance)

  // `check_solution_validity` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(check_solution_validity,
                                              GetCheckSolutionValidity,
                                              SetCheckSolutionValidity)

  // `max_qp_solver_iterations` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(max_qp_solver_iterations,
                                              GetMaxQpSolverIterations,
                                              SetMaxQpSolverIterations)

  // `solution_tolerance` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(solution_tolerance,
                                              GetSolutionTolerance,
                                              SetSolutionTolerance)

  // `use_adaptive_step_size` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(use_adaptive_step_size,
                                              GetUseAdaptiveStepSize,
                                              SetUseAdaptiveStepSize)

  // `log_collision_warnings` property.
  PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY(log_collision_warnings,
                                              GetLogCollisionWarnings,
                                              SetLogCollisionWarnings)
#undef PYBIND_JOINT_VEL_FILTER_PARAMETERS_PROPERTY

  const JointVelocityFilter::Parameters& RawParameters() const;

  static PyJointVelocityFilterParameters FromRawParameters(
      const JointVelocityFilter::Parameters& params);

 private:
  friend class PyJointVelocityFilter;
  pybind11::object py_model_;
  JointVelocityFilter::Parameters params_;
};

// Helper class for binding JointVelocityFilter that also allows
// us to keep a Python mjModel object alive as long as the instance is alive.
class PyJointVelocityFilter {
 public:
  explicit PyJointVelocityFilter(
      const internal::PyJointVelocityFilterParameters& py_params);

  std::vector<double> FilterJointVelocities(
      pybind11::handle data,
      const std::vector<double>& target_joint_velocities);

 private:
  pybind11::object py_model_;
  JointVelocityFilter filter_;

  // We keep a copy of these to detect failed preconditions, which allows us to
  // throw an exception on precondition violations.
  int num_dof_;
};

}  // namespace dm_robotics::internal

#endif  // DM_ROBOTICS_PY_CONTROLLERS_JOINT_VELOCITY_FILTER_PYTHON_HELPERS_H_
