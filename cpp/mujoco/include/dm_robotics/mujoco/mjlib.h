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

#include <mujoco/mujoco.h>  //NOLINT

namespace dm_robotics {

extern "C" {

using MjDefaultVFSFunc = decltype(mj_defaultVFS);
using MjAddFileVFSFunc = decltype(mj_addFileVFS);
using MjMakeEmptyFileVFSFunc = decltype(mj_makeEmptyFileVFS);
using MjFindFileVFSFunc = decltype(mj_findFileVFS);
using MjDeleteFileVFSFunc = decltype(mj_deleteFileVFS);
using MjDeleteVFSFunc = decltype(mj_deleteVFS);
using MjLoadXmlFunc = decltype(mj_loadXML);
using MjSaveLastXMLFunc = decltype(mj_saveLastXML);
using MjFreeLastXMLFunc = decltype(mj_freeLastXML);
using MjPrintSchemaFunc = decltype(mj_printSchema);
using MjStepFunc = decltype(mj_step);
using MjStep1Func = decltype(mj_step1);
using MjStep2Func = decltype(mj_step2);
using MjForwardFunc = decltype(mj_forward);
using MjInverseFunc = decltype(mj_inverse);
using MjForwardSkipFunc = decltype(mj_forwardSkip);
using MjInverseSkipFunc = decltype(mj_inverseSkip);
using MjDefaultLROptFunc = decltype(mj_defaultLROpt);
using MjDefaultSolRefImpFunc = decltype(mj_defaultSolRefImp);
using MjDefaultOptionFunc = decltype(mj_defaultOption);
using MjDefaultVisualFunc = decltype(mj_defaultVisual);
using MjCopyModelFunc = decltype(mj_copyModel);
using MjSaveModelFunc = decltype(mj_saveModel);
using MjLoadModelFunc = decltype(mj_loadModel);
using MjDeleteModelFunc = decltype(mj_deleteModel);
using MjSizeModelFunc = decltype(mj_sizeModel);
using MjMakeDataFunc = decltype(mj_makeData);
using MjCopyDataFunc = decltype(mj_copyData);
using MjResetDataFunc = decltype(mj_resetData);
using MjResetDataDebugFunc = decltype(mj_resetDataDebug);
using MjResetDataKeyframeFunc = decltype(mj_resetDataKeyframe);
using MjDeleteDataFunc = decltype(mj_deleteData);
using MjResetCallbacksFunc = decltype(mj_resetCallbacks);
using MjSetConstFunc = decltype(mj_setConst);
using MjSetLengthRangeFunc = decltype(mj_setLengthRange);
using MjPrintModelFunc = decltype(mj_printModel);
using MjPrintDataFunc = decltype(mj_printData);
using MjuPrintMatFunc = decltype(mju_printMat);
using MjuPrintMatSparseFunc = decltype(mju_printMatSparse);
using MjFwdPositionFunc = decltype(mj_fwdPosition);
using MjFwdVelocityFunc = decltype(mj_fwdVelocity);
using MjFwdActuationFunc = decltype(mj_fwdActuation);
using MjFwdAccelerationFunc = decltype(mj_fwdAcceleration);
using MjFwdConstraintFunc = decltype(mj_fwdConstraint);
using MjEulerFunc = decltype(mj_Euler);
using MjRungeKuttaFunc = decltype(mj_RungeKutta);
using MjInvPositionFunc = decltype(mj_invPosition);
using MjInvVelocityFunc = decltype(mj_invVelocity);
using MjInvConstraintFunc = decltype(mj_invConstraint);
using MjCompareFwdInvFunc = decltype(mj_compareFwdInv);
using MjSensorPosFunc = decltype(mj_sensorPos);
using MjSensorVelFunc = decltype(mj_sensorVel);
using MjSensorAccFunc = decltype(mj_sensorAcc);
using MjEnergyPosFunc = decltype(mj_energyPos);
using MjEnergyVelFunc = decltype(mj_energyVel);
using MjCheckPosFunc = decltype(mj_checkPos);
using MjCheckVelFunc = decltype(mj_checkVel);
using MjCheckAccFunc = decltype(mj_checkAcc);
using MjKinematicsFunc = decltype(mj_kinematics);
using MjComPosFunc = decltype(mj_comPos);
using MjCamlightFunc = decltype(mj_camlight);
using MjTendonFunc = decltype(mj_tendon);
using MjTransmissionFunc = decltype(mj_transmission);
using MjCrbFunc = decltype(mj_crb);
using MjFactorMFunc = decltype(mj_factorM);
using MjSolveMFunc = decltype(mj_solveM);
using MjSolveM2Func = decltype(mj_solveM2);
using MjComVelFunc = decltype(mj_comVel);
using MjPassiveFunc = decltype(mj_passive);
using MjSubtreeVelFunc = decltype(mj_subtreeVel);
using MjRneFunc = decltype(mj_rne);
using MjRnePostConstraintFunc = decltype(mj_rnePostConstraint);
using MjCollisionFunc = decltype(mj_collision);
using MjMakeConstraintFunc = decltype(mj_makeConstraint);
using MjProjectConstraintFunc = decltype(mj_projectConstraint);
using MjReferenceConstraintFunc = decltype(mj_referenceConstraint);
using MjConstraintUpdateFunc = decltype(mj_constraintUpdate);
using MjAddContactFunc = decltype(mj_addContact);
using MjIsPyramidalFunc = decltype(mj_isPyramidal);
using MjIsSparseFunc = decltype(mj_isSparse);
using MjIsDualFunc = decltype(mj_isDual);
using MjMulJacVecFunc = decltype(mj_mulJacVec);
using MjMulJacTVecFunc = decltype(mj_mulJacTVec);
using MjJacFunc = decltype(mj_jac);
using MjJacBodyFunc = decltype(mj_jacBody);
using MjJacBodyComFunc = decltype(mj_jacBodyCom);
using MjJacGeomFunc = decltype(mj_jacGeom);
using MjJacSiteFunc = decltype(mj_jacSite);
using MjJacPointAxisFunc = decltype(mj_jacPointAxis);
using MjName2IdFunc = decltype(mj_name2id);
using MjId2NameFunc = decltype(mj_id2name);
using MjFullMFunc = decltype(mj_fullM);
using MjMulMFunc = decltype(mj_mulM);
using MjMulM2Func = decltype(mj_mulM2);
using MjAddMFunc = decltype(mj_addM);
using MjApplyFTFunc = decltype(mj_applyFT);
using MjObjectVelocityFunc = decltype(mj_objectVelocity);
using MjObjectAccelerationFunc = decltype(mj_objectAcceleration);
using MjContactForceFunc = decltype(mj_contactForce);
using MjDifferentiatePosFunc = decltype(mj_differentiatePos);
using MjIntegratePosFunc = decltype(mj_integratePos);
using MjNormalizeQuatFunc = decltype(mj_normalizeQuat);
using MjLocal2GlobalFunc = decltype(mj_local2Global);
using MjGetTotalmassFunc = decltype(mj_getTotalmass);
using MjSetTotalmassFunc = decltype(mj_setTotalmass);
using MjVersionFunc = decltype(mj_version);
using MjRayFunc = decltype(mj_ray);
using MjRayHfieldFunc = decltype(mj_rayHfield);
using MjRayMeshFunc = decltype(mj_rayMesh);
using MjuRayGeomFunc = decltype(mju_rayGeom);
using MjuRaySkinFunc = decltype(mju_raySkin);
using MjvDefaultCameraFunc = decltype(mjv_defaultCamera);
using MjvDefaultPerturbFunc = decltype(mjv_defaultPerturb);
using MjvRoom2ModelFunc = decltype(mjv_room2model);
using MjvModel2RoomFunc = decltype(mjv_model2room);
using MjvCameraInModelFunc = decltype(mjv_cameraInModel);
using MjvCameraInRoomFunc = decltype(mjv_cameraInRoom);
using MjvFrustumHeightFunc = decltype(mjv_frustumHeight);
using MjvAlignToCameraFunc = decltype(mjv_alignToCamera);
using MjvMoveCameraFunc = decltype(mjv_moveCamera);
using MjvMovePerturbFunc = decltype(mjv_movePerturb);
using MjvMoveModelFunc = decltype(mjv_moveModel);
using MjvInitPerturbFunc = decltype(mjv_initPerturb);
using MjvApplyPerturbPoseFunc = decltype(mjv_applyPerturbPose);
using MjvApplyPerturbForceFunc = decltype(mjv_applyPerturbForce);
using MjvAverageCameraFunc = decltype(mjv_averageCamera);
using MjvSelectFunc = decltype(mjv_select);
using MjvDefaultOptionFunc = decltype(mjv_defaultOption);
using MjvDefaultFigureFunc = decltype(mjv_defaultFigure);
using MjvInitGeomFunc = decltype(mjv_initGeom);
using MjvMakeConnectorFunc = decltype(mjv_makeConnector);
using MjvDefaultSceneFunc = decltype(mjv_defaultScene);
using MjvMakeSceneFunc = decltype(mjv_makeScene);
using MjvFreeSceneFunc = decltype(mjv_freeScene);
using MjvUpdateSceneFunc = decltype(mjv_updateScene);
using MjvAddGeomsFunc = decltype(mjv_addGeoms);
using MjvMakeLightsFunc = decltype(mjv_makeLights);
using MjvUpdateCameraFunc = decltype(mjv_updateCamera);
using MjvUpdateSkinFunc = decltype(mjv_updateSkin);
using MjrDefaultContextFunc = decltype(mjr_defaultContext);
using MjrMakeContextFunc = decltype(mjr_makeContext);
using MjrChangeFontFunc = decltype(mjr_changeFont);
using MjrAddAuxFunc = decltype(mjr_addAux);
using MjrFreeContextFunc = decltype(mjr_freeContext);
using MjrUploadTextureFunc = decltype(mjr_uploadTexture);
using MjrUploadMeshFunc = decltype(mjr_uploadMesh);
using MjrUploadHFieldFunc = decltype(mjr_uploadHField);
using MjrRestoreBufferFunc = decltype(mjr_restoreBuffer);
using MjrSetBufferFunc = decltype(mjr_setBuffer);
using MjrReadPixelsFunc = decltype(mjr_readPixels);
using MjrDrawPixelsFunc = decltype(mjr_drawPixels);
using MjrBlitBufferFunc = decltype(mjr_blitBuffer);
using MjrSetAuxFunc = decltype(mjr_setAux);
using MjrBlitAuxFunc = decltype(mjr_blitAux);
using MjrTextFunc = decltype(mjr_text);
using MjrOverlayFunc = decltype(mjr_overlay);
using MjrMaxViewportFunc = decltype(mjr_maxViewport);
using MjrRectangleFunc = decltype(mjr_rectangle);
using MjrFigureFunc = decltype(mjr_figure);
using MjrRenderFunc = decltype(mjr_render);
using MjrFinishFunc = decltype(mjr_finish);
using MjrGetErrorFunc = decltype(mjr_getError);
using MjrFindRectFunc = decltype(mjr_findRect);
using MjuiThemeSpacingFunc = decltype(mjui_themeSpacing);
using MjuiThemeColorFunc = decltype(mjui_themeColor);
using MjuiAddFunc = decltype(mjui_add);
using MjuiResizeFunc = decltype(mjui_resize);
using MjuiUpdateFunc = decltype(mjui_update);
using MjuiEventFunc = decltype(mjui_event);
using MjuiRenderFunc = decltype(mjui_render);
using MjuErrorFunc = decltype(mju_error);
using MjuErrorIFunc = decltype(mju_error_i);
using MjuErrorSFunc = decltype(mju_error_s);
using MjuWarningFunc = decltype(mju_warning);
using MjuWarningIFunc = decltype(mju_warning_i);
using MjuWarningSFunc = decltype(mju_warning_s);
using MjuClearHandlersFunc = decltype(mju_clearHandlers);
using MjuMallocFunc = decltype(mju_malloc);
using MjuFreeFunc = decltype(mju_free);
using MjWarningFunc = decltype(mj_warning);
using MjuWriteLogFunc = decltype(mju_writeLog);
using MjuZero3Func = decltype(mju_zero3);
using MjuCopy3Func = decltype(mju_copy3);
using MjuScl3Func = decltype(mju_scl3);
using MjuAdd3Func = decltype(mju_add3);
using MjuSub3Func = decltype(mju_sub3);
using MjuAddTo3Func = decltype(mju_addTo3);
using MjuSubFrom3Func = decltype(mju_subFrom3);
using MjuAddToScl3Func = decltype(mju_addToScl3);
using MjuAddScl3Func = decltype(mju_addScl3);
using MjuNormalize3Func = decltype(mju_normalize3);
using MjuNorm3Func = decltype(mju_norm3);
using MjuDot3Func = decltype(mju_dot3);
using MjuDist3Func = decltype(mju_dist3);
using MjuRotVecMatFunc = decltype(mju_rotVecMat);
using MjuRotVecMatTFunc = decltype(mju_rotVecMatT);
using MjuCrossFunc = decltype(mju_cross);
using MjuZero4Func = decltype(mju_zero4);
using MjuUnit4Func = decltype(mju_unit4);
using MjuCopy4Func = decltype(mju_copy4);
using MjuNormalize4Func = decltype(mju_normalize4);
using MjuZeroFunc = decltype(mju_zero);
using MjuCopyFunc = decltype(mju_copy);
using MjuSumFunc = decltype(mju_sum);
using MjuL1Func = decltype(mju_L1);
using MjuSclFunc = decltype(mju_scl);
using MjuAddFunc = decltype(mju_add);
using MjuSubFunc = decltype(mju_sub);
using MjuAddToFunc = decltype(mju_addTo);
using MjuSubFromFunc = decltype(mju_subFrom);
using MjuAddToSclFunc = decltype(mju_addToScl);
using MjuAddSclFunc = decltype(mju_addScl);
using MjuNormalizeFunc = decltype(mju_normalize);
using MjuNormFunc = decltype(mju_norm);
using MjuDotFunc = decltype(mju_dot);
using MjuMulMatVecFunc = decltype(mju_mulMatVec);
using MjuMulMatTVecFunc = decltype(mju_mulMatTVec);
using MjuTransposeFunc = decltype(mju_transpose);
using MjuMulMatMatFunc = decltype(mju_mulMatMat);
using MjuMulMatMatTFunc = decltype(mju_mulMatMatT);
using MjuMulMatTMatFunc = decltype(mju_mulMatTMat);
using MjuSqrMatTDFunc = decltype(mju_sqrMatTD);
using MjuTransformSpatialFunc = decltype(mju_transformSpatial);
using MjuRotVecQuatFunc = decltype(mju_rotVecQuat);
using MjuNegQuatFunc = decltype(mju_negQuat);
using MjuMulQuatFunc = decltype(mju_mulQuat);
using MjuMulQuatAxisFunc = decltype(mju_mulQuatAxis);
using MjuAxisAngle2QuatFunc = decltype(mju_axisAngle2Quat);
using MjuQuat2VelFunc = decltype(mju_quat2Vel);
using MjuSubQuatFunc = decltype(mju_subQuat);
using MjuQuat2MatFunc = decltype(mju_quat2Mat);
using MjuMat2QuatFunc = decltype(mju_mat2Quat);
using MjuDerivQuatFunc = decltype(mju_derivQuat);
using MjuQuatIntegrateFunc = decltype(mju_quatIntegrate);
using MjuQuatZ2VecFunc = decltype(mju_quatZ2Vec);
using MjuMulPoseFunc = decltype(mju_mulPose);
using MjuNegPoseFunc = decltype(mju_negPose);
using MjuTrnVecPoseFunc = decltype(mju_trnVecPose);
using MjuCholFactorFunc = decltype(mju_cholFactor);
using MjuCholSolveFunc = decltype(mju_cholSolve);
using MjuCholUpdateFunc = decltype(mju_cholUpdate);
using MjuEig3Func = decltype(mju_eig3);
using MjuMuscleGainFunc = decltype(mju_muscleGain);
using MjuMuscleBiasFunc = decltype(mju_muscleBias);
using MjuMuscleDynamicsFunc = decltype(mju_muscleDynamics);
using MjuEncodePyramidFunc = decltype(mju_encodePyramid);
using MjuDecodePyramidFunc = decltype(mju_decodePyramid);
using MjuSpringDamperFunc = decltype(mju_springDamper);
using MjuMinFunc = decltype(mju_min);
using MjuMaxFunc = decltype(mju_max);
using MjuSignFunc = decltype(mju_sign);
using MjuRoundFunc = decltype(mju_round);
using MjuType2StrFunc = decltype(mju_type2Str);
using MjuStr2TypeFunc = decltype(mju_str2Type);
using MjuWarningTextFunc = decltype(mju_warningText);
using MjuIsBadFunc = decltype(mju_isBad);
using MjuIsZeroFunc = decltype(mju_isZero);
using MjuStandardNormalFunc = decltype(mju_standardNormal);
using MjuF2NFunc = decltype(mju_f2n);
using MjuN2FFunc = decltype(mju_n2f);
using MjuD2NFunc = decltype(mju_d2n);
using MjuN2DFunc = decltype(mju_n2d);
using MjuInsertionSortFunc = decltype(mju_insertionSort);
using MjuHaltonFunc = decltype(mju_Halton);
using MjuStrncpyFunc = decltype(mju_strncpy);

}  // extern "C"

// A struct containing pointers to MuJoCo public API functions.
//
// This is in the process of being phased out, but is necessary due to legacy
// code.
class MjLib {
 public:
  // Initializes an MjLib object.
  //
  // Args:
  //   libmujoco_path: Ignored, function signature is only for backwards
  //     compatibility purposes.
  //   dlopen_flags: Ignored, function signature is only for backwards
  //     compatibility purposes.
  MjLib(const std::string& libmujoco_path, int dlopen_flags);
  ~MjLib();

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
  mjfConFilt& mjcb_contactfilter;                               // NOLINT
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
