#ifndef DM_ROBOTICS_PY_CONTROLLERS_CARTESIAN_6D_TO_JOINT_VELOCITY_MAPPER_PARAMETERS_H_
#define DM_ROBOTICS_PY_CONTROLLERS_CARTESIAN_6D_TO_JOINT_VELOCITY_MAPPER_PARAMETERS_H_

#include <array>
#include <chrono>  // NOLINT(build/c++11)
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper.h"
#include "pybind11/chrono.h"  // pybind
#include "pybind11/pybind11.h"  // pybind
#include "pybind11/pytypes.h"  // pybind
#include "pybind11/stl.h"  // pybind
// Internal status include

namespace dm_robotics::internal {

using PybindGeomGroup = std::vector<std::string>;
using PybindCollisionPair = std::pair<PybindGeomGroup, PybindGeomGroup>;

// Helper class for binding Cartesian6dToJointVelocityMapper::Parameters that
// also allows us to keep a Python mjModel object alive as long as the instance
// is alive.
class PyCartesian6dToJointVelocityMapperParameters {
 public:
  PyCartesian6dToJointVelocityMapperParameters();

  // `model` property.
  pybind11::object GetModel();

  void SetModel(pybind11::object model);

  // `joint_ids` property.
  std::vector<int> GetJointIds();

  void SetJointIds(const std::vector<int>& val);

  // `object_type` property.
  int GetObjectType();

  void SetObjectType(int val);

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
#define PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(var_name, getter_name,    \
                                                    setter_name)              \
  decltype(Cartesian6dToJointVelocityMapper::Parameters::var_name)            \
  getter_name() {                                                             \
    return params_.var_name;                                                  \
  }                                                                           \
  void setter_name(                                                           \
      const decltype(Cartesian6dToJointVelocityMapper::Parameters::var_name)& \
          val) {                                                              \
    params_.var_name = val;                                                   \
  }

  // `object_name` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(object_name, GetObjectName,
                                              SetObjectName)

  // `enable_joint_position_limits` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(enable_joint_position_limits,
                                              GetEnableJointPositionLimits,
                                              SetEnableJointPositionLimits)

  // `joint_position_limit_velocity_scale` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      joint_position_limit_velocity_scale, GetJointPositionLimitVelocityScale,
      SetJointPositionLimitVelocityScale)

  // `minimum_distance_from_joint_position_limit` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      minimum_distance_from_joint_position_limit,
      GetMinimumDistanceFromJointPositionLimit,
      SetMinimumDistanceFromJointPositionLimit)

  // `enable_joint_velocity_limits` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(enable_joint_velocity_limits,
                                              GetEnableJointVelocityLimits,
                                              SetEnableJointVelocityLimits)

  // `joint_velocity_magnitude_limits` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(joint_velocity_magnitude_limits,
                                              GetJointVelocityMagnitudeLimits,
                                              SetJointVelocityMagnitudeLimits)

  // `enable_joint_acceleration_limits` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(enable_joint_acceleration_limits,
                                              GetEnableJointAccelerationLimits,
                                              SetEnableJointAccelerationLimits)

  // `remove_joint_acceleration_limits_if_in_conflict` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      remove_joint_acceleration_limits_if_in_conflict,
      GetRemoveJointAccelerationLimitsIfInConflict,
      SetRemoveJointAccelerationLimitsIfInConflict)

  // `joint_acceleration_magnitude_limits` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      joint_acceleration_magnitude_limits, GetJointAccelerationMagnitudeLimits,
      SetJointAccelerationMagnitudeLimits)

  // `enable_collision_avoidance` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(enable_collision_avoidance,
                                              GetEnableCollisionAvoidance,
                                              SetEnableCollisionAvoidance)

  // `use_minimum_distance_contacts_only' property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      use_minimum_distance_contacts_only, GetUseMinimumDistanceContactsOnly,
      SetUseMinimumDistanceContactsOnly)

  // `collision_avoidance_normal_velocity_scale` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      collision_avoidance_normal_velocity_scale,
      GetCollisionAvoidanceNormalVelocityScale,
      SetCollisionAvoidanceNormalVelocityScale)

  // `minimum_distance_from_collisions` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(minimum_distance_from_collisions,
                                              GetMinimumDistanceFromCollisions,
                                              SetMinimumDistanceFromCollisions)

  // `collision_detection_distance` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(collision_detection_distance,
                                              GetCollisionDetectionDistance,
                                              SetCollisionDetectionDistance)

  // `cartesian_velocity_task_weighting_matrix` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      cartesian_velocity_task_weighting_matrix,
      GetCartesianVelocityTaskWeightingMatrix,
      SetCartesianVelocityTaskWeightingMatrix)

  // `cartesian_velocity_direction_task_weight` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      cartesian_velocity_direction_task_weight,
      GetCartesianVelocityDirectionTaskWeight,
      SetCartesianVelocityDirectionTaskWeight)

  // `cartesian_velocity_direction_task_weighting_matrix` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      cartesian_velocity_direction_task_weighting_matrix,
      GetCartesianVelocityDirectionTaskWeightingMatrix,
      SetCartesianVelocityDirectionTaskWeightingMatrix)

  // `enable_cartesian_velocity_direction_constraint` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      enable_cartesian_velocity_direction_constraint,
      GetEnableCartesianVelocityDirectionConstraint,
      SetEnableCartesianVelocityDirectionConstraint)

  // `cartesian_velocity_direction_constraint_axes` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      cartesian_velocity_direction_constraint_axes,
      GetCartesianVelocityDirectionConstraintAxes,
      SetCartesianVelocityDirectionConstraintAxes)

  // `check_solution_validity` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(check_solution_validity,
                                              GetCheckSolutionValidity,
                                              SetCheckSolutionValidity)

  // `max_cartesian_velocity_control_iterations` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      max_cartesian_velocity_control_iterations,
      GetMaxCartesianVelocityControlIterations,
      SetMaxCartesianVelocityControlIterations)

  // `regularization_weight` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(regularization_weight,
                                              GetRegularizationWeight,
                                              SetRegularizationWeight)

  // `solution_tolerance` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(solution_tolerance,
                                              GetSolutionTolerance,
                                              SetSolutionTolerance)

  // `enable_nullspace_control` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(enable_nullspace_control,
                                              GetEnableNullspaceControl,
                                              SetEnableNullspaceControl)

  // `return_error_on_nullspace_failure` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(return_error_on_nullspace_failure,
                                              GetReturnErrorOnNullspaceFailure,
                                              SetReturnErrorOnNullspaceFailure)

  // `clamp_nullspace_bias_to_feasible_space` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(
      clamp_nullspace_bias_to_feasible_space,
      GetClampNullspaceBiasToFeasibleSpace,
      SetClampNullspaceBiasToFeasibleSpace)

  // `max_nullspace_control_iterations` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(max_nullspace_control_iterations,
                                              GetMaxNullspaceControlIterations,
                                              SetMaxNullspaceControlIterations)

  // `nullspace_projection_slack` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(nullspace_projection_slack,
                                              GetNullspaceProjectionSlack,
                                              SetNullspaceProjectionSlack)

  // `use_adaptive_step_size` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(use_adaptive_step_size,
                                              GetUseAdaptiveStepSize,
                                              SetUseAdaptiveStepSize)

  // `log_nullspace_failure_warnings` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(log_nullspace_failure_warnings,
                                              GetLogNullspaceFailureWarnings,
                                              SetLogNullspaceFailureWarnings)

  // `log_collision_warnings` property.
  PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY(log_collision_warnings,
                                              GetLogCollisionWarnings,
                                              SetLogCollisionWarnings)
#undef PYBIND_CARTESIAN_MAPPER_PARAMETERS_PROPERTY

  const Cartesian6dToJointVelocityMapper::Parameters& RawParameters() const;

  static PyCartesian6dToJointVelocityMapperParameters FromRawParameters(
      const Cartesian6dToJointVelocityMapper::Parameters& params);

 private:
  friend class PyCartesian6dToJointVelocityMapper;
  pybind11::object py_model_;
  Cartesian6dToJointVelocityMapper::Parameters params_;
};

// Helper class for binding Cartesian6dToJointVelocityMapper that also allows us
// to keep a Python mjModel object alive as long as the instance is alive.
class PyCartesian6dToJointVelocityMapper {
 public:
  explicit PyCartesian6dToJointVelocityMapper(
      const internal::PyCartesian6dToJointVelocityMapperParameters& py_params);

  std::vector<double> ComputeJointVelocities(
      pybind11::handle data,
      const std::array<double, 6>& target_6d_cartesian_velocity);

  std::vector<double> ComputeJointVelocities(
      pybind11::handle data,
      const std::array<double, 6>& target_6d_cartesian_velocity,
      const std::vector<double>& nullspace_bias);

  std::vector<double> ComputeJointVelocities(
      pybind11::handle data,
      const std::array<double, 6>& target_6d_cartesian_velocity,
      const std::optional<std::vector<double>>& nullspace_bias);

 private:
  pybind11::object py_model_;
  Cartesian6dToJointVelocityMapper mapper_;

  // We keep a copy of these to detect failed preconditions, which allows us to
  // throw an exception on precondition violations.
  bool enable_nullspace_control_;
  int num_dof_;
};

}  // namespace dm_robotics::internal

#endif  // DM_ROBOTICS_PY_CONTROLLERS_CARTESIAN_6D_TO_JOINT_VELOCITY_MAPPER_PARAMETERS_H_
