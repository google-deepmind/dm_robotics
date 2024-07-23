#ifndef DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_VELOCITY_DIRECTION_CONSTRAINT_H_
#define DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_VELOCITY_DIRECTION_CONSTRAINT_H_

#include <array>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/types/span.h"
#include "dm_robotics/least_squares_qp/core/lsqp_constraint.h"
#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {

// MuJoCo joint velocity control constraint that prevents the realized Cartesian
// velocity direction to shift by more than 180 degrees from the desired
// direction. This constraint is a 1 DoF constraint.
//
// The QP's decision variables are assumed to be the joint velocities of a
// pre-defined subset of MuJoCo joints, in ascending order according to their
// DoF IDs.
//
// Instances of this class cannot be moved or copied.
class Cartesian6dVelocityDirectionConstraint : public LsqpConstraint {
 public:
  // Initialization parameters for Cartesian6dVelocityDirectionConstraint that
  // define a MuJoCo reference frame and a subset of MuJoCo joints.
  //
  // The reference frame is defined by its MuJoCo mjtObj type and a string
  // representing its name. The MuJoCo object can be either a body, geom, or
  // site. Only 1 DoF joints are supported at the moment.
  //
  // The `enable_axis_constraint` parameter determines which components of the
  // velocity are constrained. At least one velocity component must be enabled.
  //
  // The caller retains ownership of `model`.
  // It is the caller's responsibility to ensure the *model object
  // outlives any `Cartesian6dVelocityDirectionConstraint` instances created
  // with this object.
  struct Parameters {
    const mjModel* model;
    absl::btree_set<int> joint_ids;
    mjtObj object_type;
    std::string object_name;
    std::array<bool, 6> enable_axis_constraint = {true, true, true,
                                                  true, true, true};
  };

  // Constructs a Cartesian6dVelocityDirectionConstraint.
  //
  // The task's coefficients and bias are initialized according to the provided
  // `data` and `target_cartesian_velocity` parameters, which encode the current
  // MuJoCo environment state and the target Cartesian 6D velocity,
  // respectively. At every iteration, the coefficients and bias can be updated
  // through a call to `UpdateCoefficientsAndBias`.
  //
  // Note: all the necessary MuJoCo computations should have been performed on
  // the `data` parameter for the Jacobian computations to to be accurate.
  // Namely, mj_kinematics and mj_comPos at the very least.
  Cartesian6dVelocityDirectionConstraint(
      const Parameters& params, const mjData& data,
      absl::Span<const double> target_cartesian_velocity);

  Cartesian6dVelocityDirectionConstraint(
      const Cartesian6dVelocityDirectionConstraint&) = delete;
  Cartesian6dVelocityDirectionConstraint& operator=(
      const Cartesian6dVelocityDirectionConstraint&) = delete;

  // Updates the coefficients based on an mjData and a target velocity. This
  // function does not perform dynamic memory allocation. Note that the target
  // velocity is always assumed to be 6D [Vx, Vy, Vz, Wx, Wy, Wz], but some
  // components will be ignored as depending on the `enable_axis_constraint`
  // parameter.
  //
  // Note: all the necessary MuJoCo computations should have been performed on
  // the `data` parameter for the Jacobian computations to to be accurate.
  // Namely, mj_kinematics and mj_comPos at the very least.
  //
  // Check-fails if target_cartesian_velocity is not a view over a 6-dimensional
  // array.
  void UpdateCoefficients(const mjData& data,
                          absl::Span<const double> target_cartesian_velocity);

  // LsqpConstraint virtual members.
  absl::Span<const double> GetCoefficientMatrix() const override;
  absl::Span<const double> GetLowerBound() const override;
  absl::Span<const double> GetUpperBound() const override;
  int GetNumberOfDof() const override;
  int GetBoundsLength() const override;

 private:
  const mjModel& model_;
  mjtObj object_type_;
  int object_id_;
  std::vector<int> velocity_indexer_;

  absl::btree_set<int> joint_dof_ids_;
  std::vector<double> jacobian_buffer_;
  std::vector<double> joint_dof_jacobian_buffer_;
  std::vector<double> velocity_direction_buffer_;

  std::array<double, 1> lower_bound_;
  std::array<double, 1> upper_bound_;
  std::vector<double> coefficient_matrix_;
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_CONTROLLERS_LSQP_CARTESIAN_6D_VELOCITY_DIRECTION_CONSTRAINT_H_
