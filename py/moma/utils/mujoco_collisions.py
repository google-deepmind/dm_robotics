# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions related to MuJoCo's collision-detection.

This module provides a way of excluding collision detection between geoms that
cannot collide because of the contype and conaffinity parameters of Mujoco. This
speeds up the simulation.
"""

from typing import Optional

from dm_control import mjcf
from dm_control import mujoco


def exclude_bodies_based_on_contype_conaffinity(
    mjcf_model: mjcf.RootElement, exclude_exception_str: Optional[str] = None):
  """Adds a contact-exclude MJCF element for the body pairs described below.

  A contact-exclude MJCF element is added for every body pair for which all of
  their geoms do not pass the contype/conaffinity check. This ensures that the
  contype/conaffinity filtering happens right after the broad-phase collision
  detection mechanism, usually decreasing the amount of geoms that are checked
  through MuJoCo's bounding sphere test. This may improve computation
  significantly in cases where the broad-phase collision detection is
  ineffective in pruning geoms.

  This should be called after mjcf_model has been finalized, to ensure that the
  body names in the exclude element match those of the compiled model.

  Args:
    mjcf_model: mjcf.RootElement of the finalized MuJoCo XML for the scene.
    exclude_exception_str: if this string is found in the name of the body that
      body is exempt from the contact.add exclude operation.
  """
  # We compile the model first to make sure all the contypes/conaffinities are
  # set.
  xml_string = mjcf_model.to_xml_string()
  assets = mjcf_model.get_assets()
  model = mujoco.wrapper.MjModel.from_xml_string(xml_string, assets=assets)

  # We loop through all body pairs, and exclude them if none of the geoms in the
  # first body pass the contype/conaffinity check against the geoms in the
  # second body.
  for body1_id in range(model.nbody):
    body1_name = model.id2name(body1_id, "body")
    if (exclude_exception_str is None or
        exclude_exception_str not in body1_name):
      for body2_id in range(body1_id):
        body2_name = model.id2name(body2_id, "body")
        if (exclude_exception_str is None or
            exclude_exception_str not in body2_name):
          if not _is_any_geom_pass_contype_conaffinity_check(
              model, body1_id, body2_id):
            mjcf_model.contact.add(
                "exclude", body1=body1_name, body2=body2_name)


def _is_any_geom_pass_contype_conaffinity_check(model: mujoco.wrapper.MjModel,
                                                body1_id: int, body2_id: int):
  """Returns true if any geom pair passes the contype/conaff check."""
  body1_geomadr = model.body_geomadr[body1_id]
  body2_geomadr = model.body_geomadr[body2_id]
  body1_geomnum = model.body_geomnum[body1_id]
  body2_geomnum = model.body_geomnum[body2_id]

  # If one of the bodies has no geoms we can skip.
  if body1_geomnum == 0 or body2_geomnum == 0:
    return True

  for geom1_id in range(body1_geomadr, body1_geomadr + body1_geomnum):
    for geom2_id in range(body2_geomadr, body2_geomadr + body2_geomnum):
      if _is_pass_contype_conaffinity_check(model, geom1_id, geom2_id):
        return True

  return False


def _is_pass_contype_conaffinity_check(model: mujoco.wrapper.MjModel,
                                       geom1_id: int, geom2_id: int):
  """Returns true if the geoms pass the contype/conaffinity check."""
  return model.geom_contype[geom1_id] & model.geom_conaffinity[
      geom2_id] or model.geom_contype[geom2_id] & model.geom_conaffinity[
          geom1_id]
