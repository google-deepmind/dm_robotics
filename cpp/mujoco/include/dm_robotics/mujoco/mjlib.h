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

#ifndef DM_ROBOTICS_MUJOCO_MJLIB_H_
#define DM_ROBOTICS_MUJOCO_MJLIB_H_

#include <memory>
#include <string>

// MuJoCo headers must be included in the same order as "mujoco.h".
#include "mujoco/mjdata.h"  //NOLINT
#include "mujoco/mjmodel.h"  //NOLINT
#include "mujoco/mjrender.h"  //NOLINT
#include "mujoco/mjui.h"  //NOLINT
#include "mujoco/mjvisualize.h"  //NOLINT

namespace dm_robotics {

extern "C" {

using MjActivateFunc = int(const char*);
using MjDeactivateFunc = void();
using MjDefaultVFSFunc = void(mjVFS*);
using MjAddFileVFSFunc = int(mjVFS*, const char*, const char*);
using MjMakeEmptyFileVFSFunc = int(mjVFS*, const char*, int);
using MjFindFileVFSFunc = int(const mjVFS*, const char*);
using MjDeleteFileVFSFunc = int(mjVFS*, const char*);
using MjDeleteVFSFunc = void(mjVFS*);
using MjLoadXmlFunc = mjModel*(const char*, const mjVFS*, char*, int);
using MjSaveLastXMLFunc = int(const char*, const mjModel*, char*, int);
using MjFreeLastXMLFunc = void();
using MjPrintSchemaFunc = int(const char*, char*, int, int, int);
using MjStepFunc = void(const mjModel*, mjData*);
using MjStep1Func = void(const mjModel*, mjData*);
using MjStep2Func = void(const mjModel*, mjData*);
using MjForwardFunc = void(const mjModel*, mjData*);
using MjInverseFunc = void(const mjModel*, mjData*);
using MjForwardSkipFunc = void(const mjModel*, mjData*, int, int);
using MjInverseSkipFunc = void(const mjModel*, mjData*, int, int);
using MjDefaultLROptFunc = void(mjLROpt*);
using MjDefaultSolRefImpFunc = void(mjtNum*, mjtNum*);
using MjDefaultOptionFunc = void(mjOption*);
using MjDefaultVisualFunc = void(mjVisual*);
using MjCopyModelFunc = mjModel*(mjModel*, const mjModel*);
using MjSaveModelFunc = void(const mjModel*, const char*, void*, int);
using MjLoadModelFunc = mjModel*(const char*, const mjVFS*);
using MjDeleteModelFunc = void(mjModel*);
using MjSizeModelFunc = int(const mjModel*);
using MjMakeDataFunc = mjData*(const mjModel*);
using MjCopyDataFunc = mjData*(mjData*, const mjModel*, const mjData*);
using MjResetDataFunc = void(const mjModel*, mjData*);
using MjResetDataDebugFunc = void(const mjModel*, mjData*, unsigned char);
using MjResetDataKeyframeFunc = void(const mjModel*, mjData*, int);
using MjStackAllocFunc = mjtNum(mjData*, int);
using MjDeleteDataFunc = void(mjData*);
using MjResetCallbacksFunc = void();
using MjSetConstFunc = void(mjModel*, mjData*);
using MjSetLengthRangeFunc = int(mjModel*, mjData*, int, const mjLROpt*, char*,
                                 int);
using MjPrintModelFunc = void(const mjModel*, const char*);
using MjPrintDataFunc = void(const mjModel*, mjData*, const char*);
using MjuPrintMatFunc = void(const mjtNum*, int, int);
using MjuPrintMatSparseFunc = void(const mjtNum*, int, const int*, const int*,
                                   const int*);
using MjFwdPositionFunc = void(const mjModel*, mjData*);
using MjFwdVelocityFunc = void(const mjModel*, mjData*);
using MjFwdActuationFunc = void(const mjModel*, mjData*);
using MjFwdAccelerationFunc = void(const mjModel*, mjData*);
using MjFwdConstraintFunc = void(const mjModel*, mjData*);
using MjEulerFunc = void(const mjModel*, mjData*);
using MjRungeKuttaFunc = void(const mjModel*, mjData*, int);
using MjInvPositionFunc = void(const mjModel*, mjData*);
using MjInvVelocityFunc = void(const mjModel*, mjData*);
using MjInvConstraintFunc = void(const mjModel*, mjData*);
using MjCompareFwdInvFunc = void(const mjModel*, mjData*);
using MjSensorPosFunc = void(const mjModel*, mjData*);
using MjSensorVelFunc = void(const mjModel*, mjData*);
using MjSensorAccFunc = void(const mjModel*, mjData*);
using MjEnergyPosFunc = void(const mjModel*, mjData*);
using MjEnergyVelFunc = void(const mjModel*, mjData*);
using MjCheckPosFunc = void(const mjModel*, mjData*);
using MjCheckVelFunc = void(const mjModel*, mjData*);
using MjCheckAccFunc = void(const mjModel*, mjData*);
using MjKinematicsFunc = void(const mjModel*, mjData*);
using MjComPosFunc = void(const mjModel*, mjData*);
using MjCamlightFunc = void(const mjModel*, mjData*);
using MjTendonFunc = void(const mjModel*, mjData*);
using MjTransmissionFunc = void(const mjModel*, mjData*);
using MjCrbFunc = void(const mjModel*, mjData*);
using MjFactorMFunc = void(const mjModel*, mjData*);
using MjSolveMFunc = void(const mjModel*, mjData*, mjtNum*, const mjtNum*, int);
using MjSolveM2Func = void(const mjModel*, mjData*, mjtNum*, const mjtNum*,
                           int);
using MjComVelFunc = void(const mjModel*, mjData*);
using MjPassiveFunc = void(const mjModel*, mjData*);
using MjSubtreeVelFunc = void(const mjModel*, mjData*);
using MjRneFunc = void(const mjModel*, mjData*, int, mjtNum*);
using MjRnePostConstraintFunc = void(const mjModel*, mjData*);
using MjCollisionFunc = void(const mjModel*, mjData*);
using MjMakeConstraintFunc = void(const mjModel*, mjData*);
using MjProjectConstraintFunc = void(const mjModel*, mjData*);
using MjReferenceConstraintFunc = void(const mjModel*, mjData*);
using MjConstraintUpdateFunc = void(const mjModel*, mjData*, const mjtNum*,
                                    mjtNum*, int);
using MjAddContactFunc = int(const mjModel*, mjData*, const mjContact*);
using MjIsPyramidalFunc = int(const mjModel*);
using MjIsSparseFunc = int(const mjModel*);
using MjIsDualFunc = int(const mjModel*);
using MjMulJacVecFunc = void(const mjModel*, mjData*, mjtNum*, const mjtNum*);
using MjMulJacTVecFunc = void(const mjModel*, mjData*, mjtNum*, const mjtNum*);
using MjJacFunc = void(const mjModel*, const mjData*, mjtNum*, mjtNum*,
                       const mjtNum[3], int);
using MjJacBodyFunc = void(const mjModel*, const mjData*, mjtNum*, mjtNum*,
                           int);
using MjJacBodyComFunc = void(const mjModel*, const mjData*, mjtNum*, mjtNum*,
                              int);
using MjJacGeomFunc = void(const mjModel*, const mjData*, mjtNum*, mjtNum*,
                           int);
using MjJacSiteFunc = void(const mjModel*, const mjData*, mjtNum*, mjtNum*,
                           int);
using MjJacPointAxisFunc = void(const mjModel*, mjData*, mjtNum*, mjtNum*,
                                const mjtNum[3], const mjtNum[3], int);
using MjName2IdFunc = int(const mjModel*, int, const char*);
using MjId2NameFunc = const char*(const mjModel*, int, int);
using MjFullMFunc = void(const mjModel*, mjtNum*, const mjtNum*);
using MjMulMFunc = void(const mjModel*, const mjData*, mjtNum*, const mjtNum*);
using MjMulM2Func = void(const mjModel*, const mjData*, mjtNum*, const mjtNum*);
using MjAddMFunc = void(const mjModel*, mjData*, mjtNum*, int*, int*, int*);
using MjApplyFTFunc = void(const mjModel*, mjData*, const mjtNum*,
                           const mjtNum*, const mjtNum*, int, mjtNum*);
using MjObjectVelocityFunc = void(const mjModel*, const mjData*, int, int,
                                  mjtNum*, int);
using MjObjectAccelerationFunc = void(const mjModel*, const mjData*, int, int,
                                      mjtNum*, int);
using MjContactForceFunc = void(const mjModel*, const mjData*, int, mjtNum*);
using MjDifferentiatePosFunc = void(const mjModel*, mjtNum*, mjtNum,
                                    const mjtNum*, const mjtNum*);
using MjIntegratePosFunc = void(const mjModel*, mjtNum*, const mjtNum*, mjtNum);
using MjNormalizeQuatFunc = void(const mjModel*, mjtNum*);
using MjLocal2GlobalFunc = void(mjData*, mjtNum*, mjtNum*, const mjtNum*,
                                const mjtNum*, int, mjtByte);
using MjGetTotalmassFunc = mjtNum(const mjModel*);
using MjSetTotalmassFunc = void(mjModel*, mjtNum);
using MjVersionFunc = int();
using MjRayFunc = mjtNum(const mjModel*, const mjData*, const mjtNum*,
                         const mjtNum*, const mjtByte*, mjtByte, int, int*);
using MjRayHfieldFunc = mjtNum(const mjModel*, const mjData*, int,
                               const mjtNum*, const mjtNum*);
using MjRayMeshFunc = mjtNum(const mjModel*, const mjData*, int, const mjtNum*,
                             const mjtNum*);
using MjuRayGeomFunc = mjtNum(const mjtNum*, const mjtNum*, const mjtNum*,
                              const mjtNum*, const mjtNum*, int);
using MjuRaySkinFunc = mjtNum(int, int, const int*, const float*, const mjtNum*,
                              const mjtNum*, int*);
using MjvDefaultCameraFunc = void(mjvCamera*);
using MjvDefaultPerturbFunc = void(mjvPerturb*);
using MjvRoom2ModelFunc = void(mjtNum*, mjtNum*, const mjtNum*, const mjtNum*,
                               const mjvScene*);
using MjvModel2RoomFunc = void(mjtNum*, mjtNum*, const mjtNum*, const mjtNum*,
                               const mjvScene*);
using MjvCameraInModelFunc = void(mjtNum*, mjtNum*, mjtNum*, const mjvScene*);
using MjvCameraInRoomFunc = void(mjtNum*, mjtNum*, mjtNum*, const mjvScene*);
using MjvFrustumHeightFunc = mjtNum(const mjvScene*);
using MjvAlignToCameraFunc = void(mjtNum*, const mjtNum*, const mjtNum*);
using MjvMoveCameraFunc = void(const mjModel*, int, mjtNum, mjtNum,
                               const mjvScene*, mjvCamera*);
using MjvMovePerturbFunc = void(const mjModel*, const mjData*, int, mjtNum,
                                mjtNum, const mjvScene*, mjvPerturb*);
using MjvMoveModelFunc = void(const mjModel*, int, mjtNum, mjtNum,
                              const mjtNum*, mjvScene*);
using MjvInitPerturbFunc = void(const mjModel*, const mjData*, const mjvScene*,
                                mjvPerturb*);
using MjvApplyPerturbPoseFunc = void(const mjModel*, mjData*, const mjvPerturb*,
                                     int);
using MjvApplyPerturbForceFunc = void(const mjModel*, mjData*,
                                      const mjvPerturb*);
using MjvAverageCameraFunc = mjvGLCamera(const mjvGLCamera*,
                                         const mjvGLCamera*);
using MjvSelectFunc = int(const mjModel*, const mjData*, const mjvOption*,
                          mjtNum, mjtNum, mjtNum, const mjvScene*, mjtNum*,
                          int*, int*);
using MjvDefaultOptionFunc = void(mjvOption*);
using MjvDefaultFigureFunc = void(mjvFigure*);
using MjvInitGeomFunc = void(mjvGeom*, int, const mjtNum*, const mjtNum*,
                             const mjtNum*, const float*);
using MjvMakeConnectorFunc = void(mjvGeom*, int, mjtNum, mjtNum, mjtNum, mjtNum,
                                  mjtNum, mjtNum, mjtNum);
using MjvDefaultSceneFunc = void(mjvScene*);
using MjvMakeSceneFunc = void(const mjModel*, mjvScene*, int);
using MjvFreeSceneFunc = void(mjvScene*);
using MjvUpdateSceneFunc = void(const mjModel*, mjData*, const mjvOption*,
                                const mjvPerturb*, mjvCamera*, int, mjvScene*);
using MjvAddGeomsFunc = void(const mjModel*, mjData*, const mjvOption*,
                             const mjvPerturb*, int, mjvScene*);
using MjvMakeLightsFunc = void(const mjModel*, mjData*, mjvScene*);
using MjvUpdateCameraFunc = void(const mjModel*, mjData*, mjvCamera*,
                                 mjvScene*);
using MjvUpdateSkinFunc = void(const mjModel*, mjData*, mjvScene*);
using MjrDefaultContextFunc = void(mjrContext*);
using MjrMakeContextFunc = void(const mjModel*, mjrContext*, int);
using MjrChangeFontFunc = void(int, mjrContext*);
using MjrAddAuxFunc = void(int, int, int, int, mjrContext*);
using MjrFreeContextFunc = void(mjrContext*);
using MjrUploadTextureFunc = void(const mjModel*, const mjrContext*, int);
using MjrUploadMeshFunc = void(const mjModel*, const mjrContext*, int);
using MjrUploadHFieldFunc = void(const mjModel*, const mjrContext*, int);
using MjrRestoreBufferFunc = void(const mjrContext*);
using MjrSetBufferFunc = void(int, mjrContext*);
using MjrReadPixelsFunc = void(unsigned char*, float*, mjrRect,
                               const mjrContext*);
using MjrDrawPixelsFunc = void(const unsigned char*, const float*, mjrRect,
                               const mjrContext*);
using MjrBlitBufferFunc = void(mjrRect, mjrRect, int, int, const mjrContext*);
using MjrSetAuxFunc = void(int, const mjrContext*);
using MjrBlitAuxFunc = void(int, mjrRect, int, int, const mjrContext*);
using MjrTextFunc = void(int, const char*, const mjrContext*, float, float,
                         float, float, float);
using MjrOverlayFunc = void(int, int, mjrRect, const char*, const char*,
                            const mjrContext*);
using MjrMaxViewportFunc = mjrRect(const mjrContext*);
using MjrRectangleFunc = void(mjrRect, float, float, float, float);
using MjrFigureFunc = void(mjrRect, mjvFigure*, const mjrContext*);
using MjrRenderFunc = void(mjrRect, mjvScene*, const mjrContext*);
using MjrFinishFunc = void();
using MjrGetErrorFunc = int();
using MjrFindRectFunc = int(int, int, int, const mjrRect*);
using MjuiThemeSpacingFunc = mjuiThemeSpacing(int);
using MjuiThemeColorFunc = mjuiThemeColor(int);
using MjuiAddFunc = void(mjUI*, const mjuiDef*);
using MjuiResizeFunc = void(mjUI*, const mjrContext*);
using MjuiUpdateFunc = void(int, int, const mjUI*, const mjuiState*,
                            const mjrContext*);
using MjuiEventFunc = mjuiItem*(mjUI*, mjuiState*, const mjrContext*);
using MjuiRenderFunc = void(mjUI*, const mjuiState*, const mjrContext*);
using MjuErrorFunc = void(const char*);
using MjuErrorIFunc = void(const char*, int);
using MjuErrorSFunc = void(const char*, const char*);
using MjuWarningFunc = void(const char*);
using MjuWarningIFunc = void(const char*, int);
using MjuWarningSFunc = void(const char*, const char*);
using MjuClearHandlersFunc = void();
using MjuMallocFunc = void(int);
using MjuFreeFunc = void(void*);
using MjWarningFunc = void(mjData*, int, int);
using MjuWriteLogFunc = void(const char*, const char*);
using MjuZero3Func = void(mjtNum[3]);
using MjuCopy3Func = void(mjtNum[3], const mjtNum[3]);
using MjuScl3Func = void(mjtNum[3], const mjtNum[3], mjtNum);
using MjuAdd3Func = void(mjtNum[3], const mjtNum[3], const mjtNum[3]);
using MjuSub3Func = void(mjtNum[3], const mjtNum[3], const mjtNum[3]);
using MjuAddTo3Func = void(mjtNum[3], const mjtNum[3]);
using MjuSubFrom3Func = void(mjtNum[3], const mjtNum[3]);
using MjuAddToScl3Func = void(mjtNum[3], const mjtNum[3], mjtNum);
using MjuAddScl3Func = void(mjtNum[3], const mjtNum[3], const mjtNum[3],
                            mjtNum);
using MjuNormalize3Func = mjtNum(mjtNum[3]);
using MjuNorm3Func = mjtNum(const mjtNum[3]);
using MjuDot3Func = mjtNum(const mjtNum[3], const mjtNum[3]);
using MjuDist3Func = mjtNum(const mjtNum[3], const mjtNum[3]);
using MjuRotVecMatFunc = void(mjtNum[3], const mjtNum[3], const mjtNum[9]);
using MjuRotVecMatTFunc = void(mjtNum[3], const mjtNum[3], const mjtNum[9]);
using MjuCrossFunc = void(mjtNum[3], const mjtNum[3], const mjtNum[3]);
using MjuZero4Func = void(mjtNum[4]);
using MjuUnit4Func = void(mjtNum[4]);
using MjuCopy4Func = void(mjtNum[4], const mjtNum[4]);
using MjuNormalize4Func = mjtNum(mjtNum[4]);
using MjuZeroFunc = void(mjtNum*, int);
using MjuCopyFunc = void(mjtNum*, const mjtNum*, int);
using MjuSumFunc = mjtNum(const mjtNum*, int);
using MjuL1Func = mjtNum(const mjtNum*, int);
using MjuSclFunc = void(mjtNum*, const mjtNum*, mjtNum, int);
using MjuAddFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int);
using MjuSubFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int);
using MjuAddToFunc = void(mjtNum*, const mjtNum*, int);
using MjuSubFromFunc = void(mjtNum*, const mjtNum*, int);
using MjuAddToSclFunc = void(mjtNum*, const mjtNum*, mjtNum, int);
using MjuAddSclFunc = void(mjtNum*, const mjtNum*, const mjtNum*, mjtNum, int);
using MjuNormalizeFunc = mjtNum(mjtNum*, int);
using MjuNormFunc = mjtNum(const mjtNum*, int);
using MjuDotFunc = mjtNum(const mjtNum*, const mjtNum*, const int);
using MjuMulMatVecFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int, int);
using MjuMulMatTVecFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int, int);
using MjuTransposeFunc = void(mjtNum*, const mjtNum*, int, int);
using MjuMulMatMatFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int, int,
                              int);
using MjuMulMatMatTFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int, int,
                               int);
using MjuMulMatTMatFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int, int,
                               int);
using MjuSqrMatTDFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int, int);
using MjuTransformSpatialFunc = void(mjtNum[6], const mjtNum[6], int,
                                     const mjtNum[3], const mjtNum[3],
                                     const mjtNum[9]);
using MjuRotVecQuatFunc = void(mjtNum[3], const mjtNum[3], const mjtNum[4]);
using MjuNegQuatFunc = void(mjtNum[4], const mjtNum[4]);
using MjuMulQuatFunc = void(mjtNum[4], const mjtNum[4], const mjtNum[4]);
using MjuMulQuatAxisFunc = void(mjtNum[4], const mjtNum[4], const mjtNum[3]);
using MjuAxisAngle2QuatFunc = void(mjtNum[4], const mjtNum[3], mjtNum);
using MjuQuat2VelFunc = void(mjtNum[3], const mjtNum[4], mjtNum);
using MjuSubQuatFunc = void(mjtNum[3], const mjtNum[4], const mjtNum[4]);
using MjuQuat2MatFunc = void(mjtNum[9], const mjtNum[4]);
using MjuMat2QuatFunc = void(mjtNum[4], const mjtNum[9]);
using MjuDerivQuatFunc = void(mjtNum[4], const mjtNum[4], const mjtNum[3]);
using MjuQuatIntegrateFunc = void(mjtNum[4], const mjtNum[3], mjtNum);
using MjuQuatZ2VecFunc = void(mjtNum[4], const mjtNum[3]);
using MjuMulPoseFunc = void(mjtNum[3], mjtNum[4], const mjtNum[3],
                            const mjtNum[4], const mjtNum[3], const mjtNum[4]);
using MjuNegPoseFunc = void(mjtNum[3], mjtNum[4], const mjtNum[3],
                            const mjtNum[4]);
using MjuTrnVecPoseFunc = void(mjtNum[3], const mjtNum[3], const mjtNum[4],
                               const mjtNum[3]);
using MjuCholFactorFunc = int(mjtNum*, int, mjtNum);
using MjuCholSolveFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int);
using MjuCholUpdateFunc = int(mjtNum*, mjtNum*, int, int);
using MjuEig3Func = int(mjtNum*, mjtNum*, mjtNum*, const mjtNum*);
using MjuMuscleGainFunc = mjtNum(mjtNum, mjtNum, const mjtNum[2], mjtNum,
                                 const mjtNum[9]);
using MjuMuscleBiasFunc = mjtNum(mjtNum, const mjtNum[2], mjtNum,
                                 const mjtNum[9]);
using MjuMuscleDynamicsFunc = mjtNum(mjtNum, mjtNum, const mjtNum[2]);
using MjuEncodePyramidFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int);
using MjuDecodePyramidFunc = void(mjtNum*, const mjtNum*, const mjtNum*, int);
using MjuSpringDamperFunc = mjtNum(mjtNum, mjtNum, mjtNum, mjtNum, mjtNum);
using MjuMinFunc = mjtNum(mjtNum, mjtNum);
using MjuMaxFunc = mjtNum(mjtNum, mjtNum);
using MjuSignFunc = mjtNum(mjtNum);
using MjuRoundFunc = int(mjtNum);
using MjuType2StrFunc = const char*(int);
using MjuStr2TypeFunc = int(const char*);
using MjuWarningTextFunc = const char*(int, int);
using MjuIsBadFunc = int(mjtNum);
using MjuIsZeroFunc = int(mjtNum*, int);
using MjuStandardNormalFunc = mjtNum(mjtNum*);
using MjuF2NFunc = void(mjtNum*, const float*, int);
using MjuN2FFunc = void(float*, const mjtNum*, int);
using MjuD2NFunc = void(mjtNum*, const double*, int);
using MjuN2DFunc = void(double*, const mjtNum*, int);
using MjuInsertionSortFunc = void(mjtNum*, int);
using MjuHaltonFunc = mjtNum(int, int);
using MjuStrncpyFunc = char*(char*, const char*, int);

}  // extern "C"

// A struct containing pointers to MuJoCo public API functions.
//
// This is provided for use with binaries that cannot depend on `libmujoco.so`
// at link time, for example when MuJoCo is loaded through Python's `ctypes`
// machinery. This struct resolves MuJoCo symbols using `dlsym` in its
// constructor, and exposes public member function pointers with the same
// names and signatures as in `mujoco.h`.
class MjLib {
 private:
  // Note that pimpl_ needs to be initialized before any function pointers, as
  // it holds the handle to the MuJoCo library.
  struct Impl;
  std::unique_ptr<Impl> pimpl_;

 public:
  // Initializes an MjLib object by loading the MuJoCo dynamic library.
  //
  // Args:
  //   libmujoco_path: The path to the MuJoCo dynamic library.
  //   dlopen_flags: The flags variable to be passed to `dlopen`. See the man
  //     page for `dlopen(3)` for the list of valid flags (also available at
  //     http://man7.org/linux/man-pages/man3/dlopen.3.html).
  MjLib(const std::string& libmujoco_path, int dlopen_flags);
  ~MjLib();

  MjActivateFunc* const mj_activate;                        // NOLINT
  MjDeactivateFunc* const mj_deactivate;                    // NOLINT
  MjDefaultVFSFunc* const mj_defaultVFS;                    // NOLINT
  MjAddFileVFSFunc* const mj_addFileVFS;                    // NOLINT
  MjMakeEmptyFileVFSFunc* const mj_makeEmptyFileVFS;        // NOLINT
  MjFindFileVFSFunc* const mj_findFileVFS;                  // NOLINT
  MjDeleteFileVFSFunc* const mj_deleteFileVFS;              // NOLINT
  MjDeleteVFSFunc* const mj_deleteVFS;                      // NOLINT
  MjSaveLastXMLFunc* const mj_saveLastXML;                  // NOLINT
  MjFreeLastXMLFunc* const mj_freeLastXML;                  // NOLINT
  MjPrintSchemaFunc* const mj_printSchema;                  // NOLINT
  MjStepFunc* const mj_step;                                // NOLINT
  MjStep1Func* const mj_step1;                              // NOLINT
  MjStep2Func* const mj_step2;                              // NOLINT
  MjForwardFunc* const mj_forward;                          // NOLINT
  MjInverseFunc* const mj_inverse;                          // NOLINT
  MjForwardSkipFunc* const mj_forwardSkip;                  // NOLINT
  MjInverseSkipFunc* const mj_inverseSkip;                  // NOLINT
  MjDefaultLROptFunc* const mj_defaultLROpt;                // NOLINT
  MjDefaultSolRefImpFunc* const mj_defaultSolRefImp;        // NOLINT
  MjDefaultOptionFunc* const mj_defaultOption;              // NOLINT
  MjDefaultVisualFunc* const mj_defaultVisual;              // NOLINT
  MjCopyModelFunc* const mj_copyModel;                      // NOLINT
  MjSaveModelFunc* const mj_saveModel;                      // NOLINT
  MjLoadModelFunc* const mj_loadModel;                      // NOLINT
  MjDeleteModelFunc* const mj_deleteModel;                  // NOLINT
  MjSizeModelFunc* const mj_sizeModel;                      // NOLINT
  MjMakeDataFunc* const mj_makeData;                        // NOLINT
  MjCopyDataFunc* const mj_copyData;                        // NOLINT
  MjResetDataFunc* const mj_resetData;                      // NOLINT
  MjResetDataDebugFunc* const mj_resetDataDebug;            // NOLINT
  MjResetDataKeyframeFunc* const mj_resetDataKeyframe;      // NOLINT
  MjStackAllocFunc* const mj_stackAlloc;                    // NOLINT
  MjDeleteDataFunc* const mj_deleteData;                    // NOLINT
  MjResetCallbacksFunc* const mj_resetCallbacks;            // NOLINT
  MjSetConstFunc* const mj_setConst;                        // NOLINT
  MjSetLengthRangeFunc* const mj_setLengthRange;            // NOLINT
  MjPrintModelFunc* const mj_printModel;                    // NOLINT
  MjPrintDataFunc* const mj_printData;                      // NOLINT
  MjuPrintMatFunc* const mju_printMat;                      // NOLINT
  MjuPrintMatSparseFunc* const mju_printMatSparse;          // NOLINT
  MjFwdPositionFunc* const mj_fwdPosition;                  // NOLINT
  MjFwdVelocityFunc* const mj_fwdVelocity;                  // NOLINT
  MjFwdActuationFunc* const mj_fwdActuation;                // NOLINT
  MjFwdAccelerationFunc* const mj_fwdAcceleration;          // NOLINT
  MjFwdConstraintFunc* const mj_fwdConstraint;              // NOLINT
  MjEulerFunc* const mj_Euler;                              // NOLINT
  MjRungeKuttaFunc* const mj_RungeKutta;                    // NOLINT
  MjInvPositionFunc* const mj_invPosition;                  // NOLINT
  MjInvVelocityFunc* const mj_invVelocity;                  // NOLINT
  MjInvConstraintFunc* const mj_invConstraint;              // NOLINT
  MjCompareFwdInvFunc* const mj_compareFwdInv;              // NOLINT
  MjSensorPosFunc* const mj_sensorPos;                      // NOLINT
  MjSensorVelFunc* const mj_sensorVel;                      // NOLINT
  MjSensorAccFunc* const mj_sensorAcc;                      // NOLINT
  MjEnergyPosFunc* const mj_energyPos;                      // NOLINT
  MjEnergyVelFunc* const mj_energyVel;                      // NOLINT
  MjCheckPosFunc* const mj_checkPos;                        // NOLINT
  MjCheckVelFunc* const mj_checkVel;                        // NOLINT
  MjCheckAccFunc* const mj_checkAcc;                        // NOLINT
  MjKinematicsFunc* const mj_kinematics;                    // NOLINT
  MjComPosFunc* const mj_comPos;                            // NOLINT
  MjCamlightFunc* const mj_camlight;                        // NOLINT
  MjTendonFunc* const mj_tendon;                            // NOLINT
  MjTransmissionFunc* const mj_transmission;                // NOLINT
  MjCrbFunc* const mj_crb;                                  // NOLINT
  MjFactorMFunc* const mj_factorM;                          // NOLINT
  MjSolveMFunc* const mj_solveM;                            // NOLINT
  MjSolveM2Func* const mj_solveM2;                          // NOLINT
  MjComVelFunc* const mj_comVel;                            // NOLINT
  MjPassiveFunc* const mj_passive;                          // NOLINT
  MjSubtreeVelFunc* const mj_subtreeVel;                    // NOLINT
  MjRneFunc* const mj_rne;                                  // NOLINT
  MjRnePostConstraintFunc* const mj_rnePostConstraint;      // NOLINT
  MjCollisionFunc* const mj_collision;                      // NOLINT
  MjMakeConstraintFunc* const mj_makeConstraint;            // NOLINT
  MjProjectConstraintFunc* const mj_projectConstraint;      // NOLINT
  MjReferenceConstraintFunc* const mj_referenceConstraint;  // NOLINT
  MjConstraintUpdateFunc* const mj_constraintUpdate;        // NOLINT
  MjAddContactFunc* const mj_addContact;                    // NOLINT
  MjIsPyramidalFunc* const mj_isPyramidal;                  // NOLINT
  MjIsSparseFunc* const mj_isSparse;                        // NOLINT
  MjIsDualFunc* const mj_isDual;                            // NOLINT
  MjMulJacVecFunc* const mj_mulJacVec;                      // NOLINT
  MjMulJacTVecFunc* const mj_mulJacTVec;                    // NOLINT
  MjJacFunc* const mj_jac;                                  // NOLINT
  MjJacBodyFunc* const mj_jacBody;                          // NOLINT
  MjJacBodyComFunc* const mj_jacBodyCom;                    // NOLINT
  MjJacGeomFunc* const mj_jacGeom;                          // NOLINT
  MjJacSiteFunc* const mj_jacSite;                          // NOLINT
  MjJacPointAxisFunc* const mj_jacPointAxis;                // NOLINT
  MjName2IdFunc* const mj_name2id;                          // NOLINT
  MjId2NameFunc* const mj_id2name;                          // NOLINT
  MjFullMFunc* const mj_fullM;                              // NOLINT
  MjMulMFunc* const mj_mulM;                                // NOLINT
  MjMulM2Func* const mj_mulM2;                              // NOLINT
  MjAddMFunc* const mj_addM;                                // NOLINT
  MjApplyFTFunc* const mj_applyFT;                          // NOLINT
  MjObjectVelocityFunc* const mj_objectVelocity;            // NOLINT
  MjObjectAccelerationFunc* const mj_objectAcceleration;    // NOLINT
  MjContactForceFunc* const mj_contactForce;                // NOLINT
  MjDifferentiatePosFunc* const mj_differentiatePos;        // NOLINT
  MjIntegratePosFunc* const mj_integratePos;                // NOLINT
  MjNormalizeQuatFunc* const mj_normalizeQuat;              // NOLINT
  MjLocal2GlobalFunc* const mj_local2Global;                // NOLINT
  MjGetTotalmassFunc* const mj_getTotalmass;                // NOLINT
  MjSetTotalmassFunc* const mj_setTotalmass;                // NOLINT
  MjVersionFunc* const mj_version;                          // NOLINT
  MjRayFunc* const mj_ray;                                  // NOLINT
  MjRayHfieldFunc* const mj_rayHfield;                      // NOLINT
  MjRayMeshFunc* const mj_rayMesh;                          // NOLINT
  MjuRayGeomFunc* const mju_rayGeom;                        // NOLINT
  MjuRaySkinFunc* const mju_raySkin;                        // NOLINT
  MjvDefaultCameraFunc* const mjv_defaultCamera;            // NOLINT
  MjvDefaultPerturbFunc* const mjv_defaultPerturb;          // NOLINT
  MjvRoom2ModelFunc* const mjv_room2model;                  // NOLINT
  MjvModel2RoomFunc* const mjv_model2room;                  // NOLINT
  MjvCameraInModelFunc* const mjv_cameraInModel;            // NOLINT
  MjvCameraInRoomFunc* const mjv_cameraInRoom;              // NOLINT
  MjvFrustumHeightFunc* const mjv_frustumHeight;            // NOLINT
  MjvAlignToCameraFunc* const mjv_alignToCamera;            // NOLINT
  MjvMoveCameraFunc* const mjv_moveCamera;                  // NOLINT
  MjvMovePerturbFunc* const mjv_movePerturb;                // NOLINT
  MjvMoveModelFunc* const mjv_moveModel;                    // NOLINT
  MjvInitPerturbFunc* const mjv_initPerturb;                // NOLINT
  MjvApplyPerturbPoseFunc* const mjv_applyPerturbPose;      // NOLINT
  MjvApplyPerturbForceFunc* const mjv_applyPerturbForce;    // NOLINT
  MjvAverageCameraFunc* const mjv_averageCamera;            // NOLINT
  MjvSelectFunc* const mjv_select;                          // NOLINT
  MjvDefaultOptionFunc* const mjv_defaultOption;            // NOLINT
  MjvDefaultFigureFunc* const mjv_defaultFigure;            // NOLINT
  MjvInitGeomFunc* const mjv_initGeom;                      // NOLINT
  MjvMakeConnectorFunc* const mjv_makeConnector;            // NOLINT
  MjvDefaultSceneFunc* const mjv_defaultScene;              // NOLINT
  MjvMakeSceneFunc* const mjv_makeScene;                    // NOLINT
  MjvFreeSceneFunc* const mjv_freeScene;                    // NOLINT
  MjvUpdateSceneFunc* const mjv_updateScene;                // NOLINT
  MjvAddGeomsFunc* const mjv_addGeoms;                      // NOLINT
  MjvMakeLightsFunc* const mjv_makeLights;                  // NOLINT
  MjvUpdateCameraFunc* const mjv_updateCamera;              // NOLINT
  MjvUpdateSkinFunc* const mjv_updateSkin;                  // NOLINT
  MjrDefaultContextFunc* const mjr_defaultContext;          // NOLINT
  MjrMakeContextFunc* const mjr_makeContext;                // NOLINT
  MjrChangeFontFunc* const mjr_changeFont;                  // NOLINT
  MjrAddAuxFunc* const mjr_addAux;                          // NOLINT
  MjrFreeContextFunc* const mjr_freeContext;                // NOLINT
  MjrUploadTextureFunc* const mjr_uploadTexture;            // NOLINT
  MjrUploadMeshFunc* const mjr_uploadMesh;                  // NOLINT
  MjrUploadHFieldFunc* const mjr_uploadHField;              // NOLINT
  MjrRestoreBufferFunc* const mjr_restoreBuffer;            // NOLINT
  MjrSetBufferFunc* const mjr_setBuffer;                    // NOLINT
  MjrReadPixelsFunc* const mjr_readPixels;                  // NOLINT
  MjrDrawPixelsFunc* const mjr_drawPixels;                  // NOLINT
  MjrBlitBufferFunc* const mjr_blitBuffer;                  // NOLINT
  MjrSetAuxFunc* const mjr_setAux;                          // NOLINT
  MjrBlitAuxFunc* const mjr_blitAux;                        // NOLINT
  MjrTextFunc* const mjr_text;                              // NOLINT
  MjrOverlayFunc* const mjr_overlay;                        // NOLINT
  MjrMaxViewportFunc* const mjr_maxViewport;                // NOLINT
  MjrRectangleFunc* const mjr_rectangle;                    // NOLINT
  MjrFigureFunc* const mjr_figure;                          // NOLINT
  MjrRenderFunc* const mjr_render;                          // NOLINT
  MjrFinishFunc* const mjr_finish;                          // NOLINT
  MjrGetErrorFunc* const mjr_getError;                      // NOLINT
  MjrFindRectFunc* const mjr_findRect;                      // NOLINT
  MjuiThemeSpacingFunc* const mjui_themeSpacing;            // NOLINT
  MjuiThemeColorFunc* const mjui_themeColor;                // NOLINT
  MjuiAddFunc* const mjui_add;                              // NOLINT
  MjuiResizeFunc* const mjui_resize;                        // NOLINT
  MjuiUpdateFunc* const mjui_update;                        // NOLINT
  MjuiEventFunc* const mjui_event;                          // NOLINT
  MjuiRenderFunc* const mjui_render;                        // NOLINT
  MjuErrorFunc* const mju_error;                            // NOLINT
  MjuErrorIFunc* const mju_error_i;                         // NOLINT
  MjuErrorSFunc* const mju_error_s;                         // NOLINT
  MjuWarningFunc* const mju_warning;                        // NOLINT
  MjuWarningIFunc* const mju_warning_i;                     // NOLINT
  MjuWarningSFunc* const mju_warning_s;                     // NOLINT
  MjuClearHandlersFunc* const mju_clearHandlers;            // NOLINT
  MjuMallocFunc* const mju_malloc;                          // NOLINT
  MjuFreeFunc* const mju_free;                              // NOLINT
  MjWarningFunc* const mj_warning;                          // NOLINT
  MjuWriteLogFunc* const mju_writeLog;                      // NOLINT
  MjuZero3Func* const mju_zero3;                            // NOLINT
  MjuCopy3Func* const mju_copy3;                            // NOLINT
  MjuScl3Func* const mju_scl3;                              // NOLINT
  MjuAdd3Func* const mju_add3;                              // NOLINT
  MjuSub3Func* const mju_sub3;                              // NOLINT
  MjuAddTo3Func* const mju_addTo3;                          // NOLINT
  MjuSubFrom3Func* const mju_subFrom3;                      // NOLINT
  MjuAddToScl3Func* const mju_addToScl3;                    // NOLINT
  MjuAddScl3Func* const mju_addScl3;                        // NOLINT
  MjuNormalize3Func* const mju_normalize3;                  // NOLINT
  MjuNorm3Func* const mju_norm3;                            // NOLINT
  MjuDot3Func* const mju_dot3;                              // NOLINT
  MjuDist3Func* const mju_dist3;                            // NOLINT
  MjuRotVecMatFunc* const mju_rotVecMat;                    // NOLINT
  MjuRotVecMatTFunc* const mju_rotVecMatT;                  // NOLINT
  MjuCrossFunc* const mju_cross;                            // NOLINT
  MjuZero4Func* const mju_zero4;                            // NOLINT
  MjuUnit4Func* const mju_unit4;                            // NOLINT
  MjuCopy4Func* const mju_copy4;                            // NOLINT
  MjuNormalize4Func* const mju_normalize4;                  // NOLINT
  MjuZeroFunc* const mju_zero;                              // NOLINT
  MjuCopyFunc* const mju_copy;                              // NOLINT
  MjuSumFunc* const mju_sum;                                // NOLINT
  MjuL1Func* const mju_L1;                                  // NOLINT
  MjuSclFunc* const mju_scl;                                // NOLINT
  MjuAddFunc* const mju_add;                                // NOLINT
  MjuSubFunc* const mju_sub;                                // NOLINT
  MjuAddToFunc* const mju_addTo;                            // NOLINT
  MjuSubFromFunc* const mju_subFrom;                        // NOLINT
  MjuAddToSclFunc* const mju_addToScl;                      // NOLINT
  MjuAddSclFunc* const mju_addScl;                          // NOLINT
  MjuNormalizeFunc* const mju_normalize;                    // NOLINT
  MjuNormFunc* const mju_norm;                              // NOLINT
  MjuDotFunc* const mju_dot;                                // NOLINT
  MjuMulMatVecFunc* const mju_mulMatVec;                    // NOLINT
  MjuMulMatTVecFunc* const mju_mulMatTVec;                  // NOLINT
  MjuTransposeFunc* const mju_transpose;                    // NOLINT
  MjuMulMatMatFunc* const mju_mulMatMat;                    // NOLINT
  MjuMulMatMatTFunc* const mju_mulMatMatT;                  // NOLINT
  MjuMulMatTMatFunc* const mju_mulMatTMat;                  // NOLINT
  MjuSqrMatTDFunc* const mju_sqrMatTD;                      // NOLINT
  MjuTransformSpatialFunc* const mju_transformSpatial;      // NOLINT
  MjuRotVecQuatFunc* const mju_rotVecQuat;                  // NOLINT
  MjuNegQuatFunc* const mju_negQuat;                        // NOLINT
  MjuMulQuatFunc* const mju_mulQuat;                        // NOLINT
  MjuMulQuatAxisFunc* const mju_mulQuatAxis;                // NOLINT
  MjuAxisAngle2QuatFunc* const mju_axisAngle2Quat;          // NOLINT
  MjuQuat2VelFunc* const mju_quat2Vel;                      // NOLINT
  MjuSubQuatFunc* const mju_subQuat;                        // NOLINT
  MjuQuat2MatFunc* const mju_quat2Mat;                      // NOLINT
  MjuMat2QuatFunc* const mju_mat2Quat;                      // NOLINT
  MjuDerivQuatFunc* const mju_derivQuat;                    // NOLINT
  MjuQuatIntegrateFunc* const mju_quatIntegrate;            // NOLINT
  MjuQuatZ2VecFunc* const mju_quatZ2Vec;                    // NOLINT
  MjuMulPoseFunc* const mju_mulPose;                        // NOLINT
  MjuNegPoseFunc* const mju_negPose;                        // NOLINT
  MjuTrnVecPoseFunc* const mju_trnVecPose;                  // NOLINT
  MjuCholFactorFunc* const mju_cholFactor;                  // NOLINT
  MjuCholSolveFunc* const mju_cholSolve;                    // NOLINT
  MjuCholUpdateFunc* const mju_cholUpdate;                  // NOLINT
  MjuEig3Func* const mju_eig3;                              // NOLINT
  MjuMuscleGainFunc* const mju_muscleGain;                  // NOLINT
  MjuMuscleBiasFunc* const mju_muscleBias;                  // NOLINT
  MjuMuscleDynamicsFunc* const mju_muscleDynamics;          // NOLINT
  MjuEncodePyramidFunc* const mju_encodePyramid;            // NOLINT
  MjuDecodePyramidFunc* const mju_decodePyramid;            // NOLINT
  MjuSpringDamperFunc* const mju_springDamper;              // NOLINT
  MjuMinFunc* const mju_min;                                // NOLINT
  MjuMaxFunc* const mju_max;                                // NOLINT
  MjuSignFunc* const mju_sign;                              // NOLINT
  MjuRoundFunc* const mju_round;                            // NOLINT
  MjuType2StrFunc* const mju_type2Str;                      // NOLINT
  MjuStr2TypeFunc* const mju_str2Type;                      // NOLINT
  MjuWarningTextFunc* const mju_warningText;                // NOLINT
  MjuIsBadFunc* const mju_isBad;                            // NOLINT
  MjuIsZeroFunc* const mju_isZero;                          // NOLINT
  MjuStandardNormalFunc* const mju_standardNormal;          // NOLINT
  MjuF2NFunc* const mju_f2n;                                // NOLINT
  MjuN2FFunc* const mju_n2f;                                // NOLINT
  MjuD2NFunc* const mju_d2n;                                // NOLINT
  MjuN2DFunc* const mju_n2d;                                // NOLINT
  MjuInsertionSortFunc* const mju_insertionSort;            // NOLINT
  MjuHaltonFunc* const mju_Halton;                          // NOLINT
  MjuStrncpyFunc* const mju_strncpy;                        // NOLINT

  // References to global callback function pointers.
  mjfGeneric& mjcb_passive;                                     // NOLINT
  mjfGeneric& mjcb_control;                                     // NOLINT
  mjfGeneric& mjcb_contactfilter;                               // NOLINT
  mjfSensor& mjcb_sensor;                                       // NOLINT
  mjfTime& mjcb_time;                                           // NOLINT
  mjfAct& mjcb_act_dyn;                                         // NOLINT
  mjfAct& mjcb_act_gain;                                        // NOLINT
  mjfAct& mjcb_act_bias;                                        // NOLINT
  mjfCollision (&mjCOLLISIONFUNC)[mjNGEOMTYPES][mjNGEOMTYPES];  // NOLINT

  // Note: mj_loadXML has a memory leak when the model has <user> sensor
  // elements.
  MjLoadXmlFunc* const mj_loadXML;  // NOLINT
};

}  // namespace dm_robotics

#endif  // DM_ROBOTICS_MUJOCO_MJLIB_H_
