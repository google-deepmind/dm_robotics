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

#include "dm_robotics/mujoco/mjlib.h"

#include <dlfcn.h>

#include <memory>
#include <string>

#include "dm_robotics/support/logging.h"
#include "absl/memory/memory.h"

namespace dm_robotics {
namespace {

// An RAII wrapper for the POSIX dlopen/dlclose API.
struct LibMujocoHandleDeleter {
  void operator()(void* handle) const { dlclose(handle); }
};
using LibMujocoHandle = std::unique_ptr<void, LibMujocoHandleDeleter>;

LibMujocoHandle DlOpen(const std::string& path, int dlopen_flags) {
  LibMujocoHandle handle(dlopen(path.c_str(), dlopen_flags));
  CHECK(handle != nullptr)
      << "Failed to dlopen '" << path << "', error: " << dlerror();
  return handle;
}

}  // namespace

struct MjLib::Impl {
  const LibMujocoHandle libmujoco_handle;
  Impl(const std::string& path, int dlopen_flags)
      : libmujoco_handle(DlOpen(path, dlopen_flags)) {}
};

// We terminate the program if any of the symbol handles are null.
#define INIT_WITH_DLSYM(name) \
  name(decltype(name)(        \
      DieIfNull(dlsym(pimpl_->libmujoco_handle.get(), #name))))

#define INIT_WITH_DLSYM_NULLABLE(name) \
  name(decltype(name)(dlsym(pimpl_->libmujoco_handle.get(), #name)))

#define INIT_CALLBACK_WITH_DLSYM(name) \
  name(*(decltype(&name))(             \
      DieIfNull(dlsym(pimpl_->libmujoco_handle.get(), #name))))

MjLib::MjLib(const std::string& libmujoco_path, int dlopen_flags)
    : pimpl_(absl::make_unique<Impl>(libmujoco_path, dlopen_flags)),
      INIT_WITH_DLSYM(mj_activate),
      INIT_WITH_DLSYM(mj_deactivate),
      INIT_WITH_DLSYM(mj_defaultVFS),
      INIT_WITH_DLSYM(mj_addFileVFS),
      INIT_WITH_DLSYM(mj_makeEmptyFileVFS),
      INIT_WITH_DLSYM(mj_findFileVFS),
      INIT_WITH_DLSYM(mj_deleteFileVFS),
      INIT_WITH_DLSYM(mj_deleteVFS),
      INIT_WITH_DLSYM(mj_saveLastXML),
      INIT_WITH_DLSYM(mj_freeLastXML),
      INIT_WITH_DLSYM(mj_printSchema),
      INIT_WITH_DLSYM(mj_step),
      INIT_WITH_DLSYM(mj_step1),
      INIT_WITH_DLSYM(mj_step2),
      INIT_WITH_DLSYM(mj_forward),
      INIT_WITH_DLSYM(mj_inverse),
      INIT_WITH_DLSYM(mj_forwardSkip),
      INIT_WITH_DLSYM(mj_inverseSkip),
      INIT_WITH_DLSYM(mj_defaultLROpt),
      INIT_WITH_DLSYM(mj_defaultSolRefImp),
      INIT_WITH_DLSYM(mj_defaultOption),
      INIT_WITH_DLSYM(mj_defaultVisual),
      INIT_WITH_DLSYM(mj_copyModel),
      INIT_WITH_DLSYM(mj_saveModel),
      INIT_WITH_DLSYM(mj_loadModel),
      INIT_WITH_DLSYM(mj_deleteModel),
      INIT_WITH_DLSYM(mj_sizeModel),
      INIT_WITH_DLSYM(mj_makeData),
      INIT_WITH_DLSYM(mj_copyData),
      INIT_WITH_DLSYM(mj_resetData),
      INIT_WITH_DLSYM(mj_resetDataDebug),
      INIT_WITH_DLSYM(mj_resetDataKeyframe),
      INIT_WITH_DLSYM(mj_stackAlloc),
      INIT_WITH_DLSYM(mj_deleteData),
      INIT_WITH_DLSYM(mj_resetCallbacks),
      INIT_WITH_DLSYM(mj_setConst),
      INIT_WITH_DLSYM(mj_setLengthRange),
      INIT_WITH_DLSYM(mj_printModel),
      INIT_WITH_DLSYM(mj_printData),
      INIT_WITH_DLSYM(mju_printMat),
      INIT_WITH_DLSYM(mju_printMatSparse),
      INIT_WITH_DLSYM(mj_fwdPosition),
      INIT_WITH_DLSYM(mj_fwdVelocity),
      INIT_WITH_DLSYM(mj_fwdActuation),
      INIT_WITH_DLSYM(mj_fwdAcceleration),
      INIT_WITH_DLSYM(mj_fwdConstraint),
      INIT_WITH_DLSYM(mj_Euler),
      INIT_WITH_DLSYM(mj_RungeKutta),
      INIT_WITH_DLSYM(mj_invPosition),
      INIT_WITH_DLSYM(mj_invVelocity),
      INIT_WITH_DLSYM(mj_invConstraint),
      INIT_WITH_DLSYM(mj_compareFwdInv),
      INIT_WITH_DLSYM(mj_sensorPos),
      INIT_WITH_DLSYM(mj_sensorVel),
      INIT_WITH_DLSYM(mj_sensorAcc),
      INIT_WITH_DLSYM(mj_energyPos),
      INIT_WITH_DLSYM(mj_energyVel),
      INIT_WITH_DLSYM(mj_checkPos),
      INIT_WITH_DLSYM(mj_checkVel),
      INIT_WITH_DLSYM(mj_checkAcc),
      INIT_WITH_DLSYM(mj_kinematics),
      INIT_WITH_DLSYM(mj_comPos),
      INIT_WITH_DLSYM(mj_camlight),
      INIT_WITH_DLSYM(mj_tendon),
      INIT_WITH_DLSYM(mj_transmission),
      INIT_WITH_DLSYM(mj_crb),
      INIT_WITH_DLSYM(mj_factorM),
      INIT_WITH_DLSYM(mj_solveM),
      INIT_WITH_DLSYM(mj_solveM2),
      INIT_WITH_DLSYM(mj_comVel),
      INIT_WITH_DLSYM(mj_passive),
      INIT_WITH_DLSYM(mj_subtreeVel),
      INIT_WITH_DLSYM(mj_rne),
      INIT_WITH_DLSYM(mj_rnePostConstraint),
      INIT_WITH_DLSYM(mj_collision),
      INIT_WITH_DLSYM(mj_makeConstraint),
      INIT_WITH_DLSYM(mj_projectConstraint),
      INIT_WITH_DLSYM(mj_referenceConstraint),
      INIT_WITH_DLSYM(mj_constraintUpdate),
      INIT_WITH_DLSYM(mj_addContact),
      INIT_WITH_DLSYM(mj_isPyramidal),
      INIT_WITH_DLSYM(mj_isSparse),
      INIT_WITH_DLSYM(mj_isDual),
      INIT_WITH_DLSYM(mj_mulJacVec),
      INIT_WITH_DLSYM(mj_mulJacTVec),
      INIT_WITH_DLSYM(mj_jac),
      INIT_WITH_DLSYM(mj_jacBody),
      INIT_WITH_DLSYM(mj_jacBodyCom),
      INIT_WITH_DLSYM(mj_jacGeom),
      INIT_WITH_DLSYM(mj_jacSite),
      INIT_WITH_DLSYM(mj_jacPointAxis),
      INIT_WITH_DLSYM(mj_name2id),
      INIT_WITH_DLSYM(mj_id2name),
      INIT_WITH_DLSYM(mj_fullM),
      INIT_WITH_DLSYM(mj_mulM),
      INIT_WITH_DLSYM(mj_mulM2),
      INIT_WITH_DLSYM(mj_addM),
      INIT_WITH_DLSYM(mj_applyFT),
      INIT_WITH_DLSYM(mj_objectVelocity),
      INIT_WITH_DLSYM(mj_objectAcceleration),
      INIT_WITH_DLSYM(mj_contactForce),
      INIT_WITH_DLSYM(mj_differentiatePos),
      INIT_WITH_DLSYM(mj_integratePos),
      INIT_WITH_DLSYM(mj_normalizeQuat),
      INIT_WITH_DLSYM(mj_local2Global),
      INIT_WITH_DLSYM(mj_getTotalmass),
      INIT_WITH_DLSYM(mj_setTotalmass),
      INIT_WITH_DLSYM(mj_version),
      INIT_WITH_DLSYM(mj_ray),
      INIT_WITH_DLSYM(mj_rayHfield),
      INIT_WITH_DLSYM(mj_rayMesh),
      INIT_WITH_DLSYM(mju_rayGeom),
      INIT_WITH_DLSYM(mju_raySkin),
      INIT_WITH_DLSYM(mjv_defaultCamera),
      INIT_WITH_DLSYM(mjv_defaultPerturb),
      INIT_WITH_DLSYM(mjv_room2model),
      INIT_WITH_DLSYM(mjv_model2room),
      INIT_WITH_DLSYM(mjv_cameraInModel),
      INIT_WITH_DLSYM(mjv_cameraInRoom),
      INIT_WITH_DLSYM(mjv_frustumHeight),
      INIT_WITH_DLSYM(mjv_alignToCamera),
      INIT_WITH_DLSYM(mjv_moveCamera),
      INIT_WITH_DLSYM(mjv_movePerturb),
      INIT_WITH_DLSYM(mjv_moveModel),
      INIT_WITH_DLSYM(mjv_initPerturb),
      INIT_WITH_DLSYM(mjv_applyPerturbPose),
      INIT_WITH_DLSYM(mjv_applyPerturbForce),
      INIT_WITH_DLSYM(mjv_averageCamera),
      INIT_WITH_DLSYM(mjv_select),
      INIT_WITH_DLSYM(mjv_defaultOption),
      INIT_WITH_DLSYM(mjv_defaultFigure),
      INIT_WITH_DLSYM(mjv_initGeom),
      INIT_WITH_DLSYM(mjv_makeConnector),
      INIT_WITH_DLSYM(mjv_defaultScene),
      INIT_WITH_DLSYM(mjv_makeScene),
      INIT_WITH_DLSYM(mjv_freeScene),
      INIT_WITH_DLSYM(mjv_updateScene),
      INIT_WITH_DLSYM(mjv_addGeoms),
      INIT_WITH_DLSYM(mjv_makeLights),
      INIT_WITH_DLSYM(mjv_updateCamera),
      INIT_WITH_DLSYM(mjv_updateSkin),
      // Rendering and UI-related symbols may be null if we use the no_gl DSO.
      INIT_WITH_DLSYM_NULLABLE(mjr_defaultContext),
      INIT_WITH_DLSYM_NULLABLE(mjr_makeContext),
      INIT_WITH_DLSYM_NULLABLE(mjr_changeFont),
      INIT_WITH_DLSYM_NULLABLE(mjr_addAux),
      INIT_WITH_DLSYM_NULLABLE(mjr_freeContext),
      INIT_WITH_DLSYM_NULLABLE(mjr_uploadTexture),
      INIT_WITH_DLSYM_NULLABLE(mjr_uploadMesh),
      INIT_WITH_DLSYM_NULLABLE(mjr_uploadHField),
      INIT_WITH_DLSYM_NULLABLE(mjr_restoreBuffer),
      INIT_WITH_DLSYM_NULLABLE(mjr_setBuffer),
      INIT_WITH_DLSYM_NULLABLE(mjr_readPixels),
      INIT_WITH_DLSYM_NULLABLE(mjr_drawPixels),
      INIT_WITH_DLSYM_NULLABLE(mjr_blitBuffer),
      INIT_WITH_DLSYM_NULLABLE(mjr_setAux),
      INIT_WITH_DLSYM_NULLABLE(mjr_blitAux),
      INIT_WITH_DLSYM_NULLABLE(mjr_text),
      INIT_WITH_DLSYM_NULLABLE(mjr_overlay),
      INIT_WITH_DLSYM_NULLABLE(mjr_maxViewport),
      INIT_WITH_DLSYM_NULLABLE(mjr_rectangle),
      INIT_WITH_DLSYM_NULLABLE(mjr_figure),
      INIT_WITH_DLSYM_NULLABLE(mjr_render),
      INIT_WITH_DLSYM_NULLABLE(mjr_finish),
      INIT_WITH_DLSYM_NULLABLE(mjr_getError),
      INIT_WITH_DLSYM_NULLABLE(mjr_findRect),
      INIT_WITH_DLSYM_NULLABLE(mjui_themeSpacing),
      INIT_WITH_DLSYM_NULLABLE(mjui_themeColor),
      INIT_WITH_DLSYM_NULLABLE(mjui_add),
      INIT_WITH_DLSYM_NULLABLE(mjui_resize),
      INIT_WITH_DLSYM_NULLABLE(mjui_update),
      INIT_WITH_DLSYM_NULLABLE(mjui_event),
      INIT_WITH_DLSYM_NULLABLE(mjui_render),
      INIT_WITH_DLSYM(mju_error),
      INIT_WITH_DLSYM(mju_error_i),
      INIT_WITH_DLSYM(mju_error_s),
      INIT_WITH_DLSYM(mju_warning),
      INIT_WITH_DLSYM(mju_warning_i),
      INIT_WITH_DLSYM(mju_warning_s),
      INIT_WITH_DLSYM(mju_clearHandlers),
      INIT_WITH_DLSYM(mju_malloc),
      INIT_WITH_DLSYM(mju_free),
      INIT_WITH_DLSYM(mj_warning),
      INIT_WITH_DLSYM(mju_writeLog),
      INIT_WITH_DLSYM(mju_zero3),
      INIT_WITH_DLSYM(mju_copy3),
      INIT_WITH_DLSYM(mju_scl3),
      INIT_WITH_DLSYM(mju_add3),
      INIT_WITH_DLSYM(mju_sub3),
      INIT_WITH_DLSYM(mju_addTo3),
      INIT_WITH_DLSYM(mju_subFrom3),
      INIT_WITH_DLSYM(mju_addToScl3),
      INIT_WITH_DLSYM(mju_addScl3),
      INIT_WITH_DLSYM(mju_normalize3),
      INIT_WITH_DLSYM(mju_norm3),
      INIT_WITH_DLSYM(mju_dot3),
      INIT_WITH_DLSYM(mju_dist3),
      INIT_WITH_DLSYM(mju_rotVecMat),
      INIT_WITH_DLSYM(mju_rotVecMatT),
      INIT_WITH_DLSYM(mju_cross),
      INIT_WITH_DLSYM(mju_zero4),
      INIT_WITH_DLSYM(mju_unit4),
      INIT_WITH_DLSYM(mju_copy4),
      INIT_WITH_DLSYM(mju_normalize4),
      INIT_WITH_DLSYM(mju_zero),
      INIT_WITH_DLSYM(mju_copy),
      INIT_WITH_DLSYM(mju_sum),
      INIT_WITH_DLSYM(mju_L1),
      INIT_WITH_DLSYM(mju_scl),
      INIT_WITH_DLSYM(mju_add),
      INIT_WITH_DLSYM(mju_sub),
      INIT_WITH_DLSYM(mju_addTo),
      INIT_WITH_DLSYM(mju_subFrom),
      INIT_WITH_DLSYM(mju_addToScl),
      INIT_WITH_DLSYM(mju_addScl),
      INIT_WITH_DLSYM(mju_normalize),
      INIT_WITH_DLSYM(mju_norm),
      INIT_WITH_DLSYM(mju_dot),
      INIT_WITH_DLSYM(mju_mulMatVec),
      INIT_WITH_DLSYM(mju_mulMatTVec),
      INIT_WITH_DLSYM(mju_transpose),
      INIT_WITH_DLSYM(mju_mulMatMat),
      INIT_WITH_DLSYM(mju_mulMatMatT),
      INIT_WITH_DLSYM(mju_mulMatTMat),
      INIT_WITH_DLSYM(mju_sqrMatTD),
      INIT_WITH_DLSYM(mju_transformSpatial),
      INIT_WITH_DLSYM(mju_rotVecQuat),
      INIT_WITH_DLSYM(mju_negQuat),
      INIT_WITH_DLSYM(mju_mulQuat),
      INIT_WITH_DLSYM(mju_mulQuatAxis),
      INIT_WITH_DLSYM(mju_axisAngle2Quat),
      INIT_WITH_DLSYM(mju_quat2Vel),
      INIT_WITH_DLSYM(mju_subQuat),
      INIT_WITH_DLSYM(mju_quat2Mat),
      INIT_WITH_DLSYM(mju_mat2Quat),
      INIT_WITH_DLSYM(mju_derivQuat),
      INIT_WITH_DLSYM(mju_quatIntegrate),
      INIT_WITH_DLSYM(mju_quatZ2Vec),
      INIT_WITH_DLSYM(mju_mulPose),
      INIT_WITH_DLSYM(mju_negPose),
      INIT_WITH_DLSYM(mju_trnVecPose),
      INIT_WITH_DLSYM(mju_cholFactor),
      INIT_WITH_DLSYM(mju_cholSolve),
      INIT_WITH_DLSYM(mju_cholUpdate),
      INIT_WITH_DLSYM(mju_eig3),
      INIT_WITH_DLSYM(mju_muscleGain),
      INIT_WITH_DLSYM(mju_muscleBias),
      INIT_WITH_DLSYM(mju_muscleDynamics),
      INIT_WITH_DLSYM(mju_encodePyramid),
      INIT_WITH_DLSYM(mju_decodePyramid),
      INIT_WITH_DLSYM(mju_springDamper),
      INIT_WITH_DLSYM(mju_min),
      INIT_WITH_DLSYM(mju_max),
      INIT_WITH_DLSYM(mju_sign),
      INIT_WITH_DLSYM(mju_round),
      INIT_WITH_DLSYM(mju_type2Str),
      INIT_WITH_DLSYM(mju_str2Type),
      INIT_WITH_DLSYM(mju_warningText),
      INIT_WITH_DLSYM(mju_isBad),
      INIT_WITH_DLSYM(mju_isZero),
      INIT_WITH_DLSYM(mju_standardNormal),
      INIT_WITH_DLSYM(mju_f2n),
      INIT_WITH_DLSYM(mju_n2f),
      INIT_WITH_DLSYM(mju_d2n),
      INIT_WITH_DLSYM(mju_n2d),
      INIT_WITH_DLSYM(mju_insertionSort),
      INIT_WITH_DLSYM(mju_Halton),
      INIT_WITH_DLSYM(mju_strncpy),
      INIT_CALLBACK_WITH_DLSYM(mjcb_passive),
      INIT_CALLBACK_WITH_DLSYM(mjcb_control),
      INIT_CALLBACK_WITH_DLSYM(mjcb_contactfilter),
      INIT_CALLBACK_WITH_DLSYM(mjcb_sensor),
      INIT_CALLBACK_WITH_DLSYM(mjcb_time),
      INIT_CALLBACK_WITH_DLSYM(mjcb_act_dyn),
      INIT_CALLBACK_WITH_DLSYM(mjcb_act_gain),
      INIT_CALLBACK_WITH_DLSYM(mjcb_act_bias),
      INIT_CALLBACK_WITH_DLSYM(mjCOLLISIONFUNC),
      INIT_WITH_DLSYM(mj_loadXML) {}

MjLib::~MjLib() = default;

#undef INIT_CALLBACK_WITH_DLSYM
#undef INIT_WITH_DLSYM_NULLABLE
#undef INIT_WITH_DLSYM

}  // namespace dm_robotics
