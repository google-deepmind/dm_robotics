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
"""YCB Dataset Prop.

Original YCB dataset:  https://www.ycbbenchmarks.com/object-models
Original YCB paper: https://www.eng.yale.edu/grablab/pubs/Calli_RAM2015.pdf
"""

import dataclasses
import enum
import glob
import os
from typing import List, Optional

from dm_robotics.manipulation.props import mesh_object

# Internal imports.

_DATASET_PATH = "'YOUR LOCAL PATH TO YCB MESH ASSETS.'"
# Arbitrary  mass value for objects which masses are not listed.
# Could be corrected by weighing of physical objects.
_DEFAULT_MASS = 0.100
_DEFAULT_MESH_SCALE = 1.0


class YcbDatasetVersion(enum.Enum):
  """YCB object set versions.

  Differences are described in the document with images above. Based on the
  purchase date, in the Robotics Lab we probably have a physical dataset V1.
  """
  V1 = enum.auto()    # default
  V2 = enum.auto()
  V3 = enum.auto()


class YcbClass(enum.Enum):
  """YCB object class (category)."""
  FOOD = enum.auto()
  KITCHEN = enum.auto()
  TOOL = enum.auto()
  SHAPE = enum.auto()
  TASK = enum.auto()


@dataclasses.dataclass
class ObjectPropertiesType:
  """Class to describe object properties."""
  object_class: YcbClass       # category the object belongs to.
  mass: float = _DEFAULT_MASS  # mass in kg.
  mesh_scale: float = _DEFAULT_MESH_SCALE   # to adapt size of objects in sim.
  version: YcbDatasetVersion = YcbDatasetVersion.V1  # object set version.


# Source: Calli 2015, "Benchmarking in Manipulation Research", pg. 43
# https://www.eng.yale.edu/grablab/pubs/Calli_RAM2015.pdf
# If the mass field is missiing, a default mass value is assigned.
_ALL_OBJECT_PROPERTIES = {
    "001_chips_can": ObjectPropertiesType(YcbClass.FOOD, 0.205, mesh_scale=0.3),
    "002_master_chef_can": ObjectPropertiesType(YcbClass.FOOD, 0.414),
    "003_cracker_box": ObjectPropertiesType(YcbClass.FOOD, 0.411),
    "004_sugar_box": ObjectPropertiesType(YcbClass.FOOD, 0.514),
    "005_tomato_soup_can": ObjectPropertiesType(YcbClass.FOOD, 0.349),
    "006_mustard_bottle": ObjectPropertiesType(YcbClass.FOOD, 0.603),
    "007_tuna_fish_can": ObjectPropertiesType(YcbClass.FOOD, 0.171),
    "008_pudding_box": ObjectPropertiesType(YcbClass.FOOD, 0.187),
    "009_gelatin_box": ObjectPropertiesType(YcbClass.FOOD, 0.097),
    "010_potted_meat_can": ObjectPropertiesType(YcbClass.FOOD, 0.370),
    "011_banana": ObjectPropertiesType(YcbClass.FOOD, 0.066),
    "012_strawberry": ObjectPropertiesType(YcbClass.FOOD, 0.018),
    "013_apple": ObjectPropertiesType(YcbClass.FOOD, 0.068),
    "014_lemon": ObjectPropertiesType(YcbClass.FOOD, 0.029),
    "015_peach": ObjectPropertiesType(YcbClass.FOOD, 0.033),
    "016_pear": ObjectPropertiesType(YcbClass.FOOD, 0.049),
    "017_orange": ObjectPropertiesType(YcbClass.FOOD, 0.047),
    "018_plum": ObjectPropertiesType(YcbClass.FOOD, 0.025),
    "019_pitcher_base": ObjectPropertiesType(YcbClass.KITCHEN, 0.178),
    # there is no "020" in the dataset.
    "021_bleach_cleanser": ObjectPropertiesType(YcbClass.KITCHEN, 1.131),
    "022_windex_bottle": ObjectPropertiesType(YcbClass.KITCHEN, 1.022),
    "023_wine_glass": ObjectPropertiesType(YcbClass.KITCHEN, 0.133),
    "024_bowl": ObjectPropertiesType(YcbClass.KITCHEN, 0.147),
    "025_mug": ObjectPropertiesType(YcbClass.KITCHEN, 0.118),
    "026_sponge": ObjectPropertiesType(YcbClass.KITCHEN, 6.2e-3),
    "027_skillet": ObjectPropertiesType(YcbClass.KITCHEN, 0.950),
    "028_skillet_lid": ObjectPropertiesType(YcbClass.KITCHEN, 0.652),
    "029_plate": ObjectPropertiesType(YcbClass.KITCHEN, 0.279),
    "030_fork": ObjectPropertiesType(YcbClass.KITCHEN, 0.034),
    "031_spoon": ObjectPropertiesType(YcbClass.KITCHEN, 0.030),
    "032_knife": ObjectPropertiesType(YcbClass.KITCHEN, 0.031),
    "033_spatula": ObjectPropertiesType(YcbClass.KITCHEN, 0.0515),
    # there is no "034" in the dataset.
    "035_power_drill": ObjectPropertiesType(YcbClass.TOOL, 0.895),
    "036_wood_block": ObjectPropertiesType(YcbClass.TOOL, 0.729),
    "037_scissors": ObjectPropertiesType(YcbClass.TOOL, 0.082),
    "038_padlock": ObjectPropertiesType(YcbClass.TOOL, 0.304),
    "039_key": ObjectPropertiesType(YcbClass.TOOL, 0.0101),
    "040_large_marker": ObjectPropertiesType(YcbClass.TOOL, 0.0158),
    "041_small_marker": ObjectPropertiesType(YcbClass.TOOL, 0.0082),
    "042_adjustable_wrench": ObjectPropertiesType(YcbClass.TOOL, 0.252),
    "043_phillips_screwdriver": ObjectPropertiesType(YcbClass.TOOL, 0.097),
    "044_flat_screwdriver": ObjectPropertiesType(YcbClass.TOOL, 0.0984),
    # there is no "045" in the dataset.
    "046_plastic_bolt": ObjectPropertiesType(YcbClass.TOOL, 3.6e-3),
    "047_plastic_nut": ObjectPropertiesType(YcbClass.TOOL, 1e-3),
    "048_hammer": ObjectPropertiesType(YcbClass.TOOL, 0.665),
    "049_small_clamp": ObjectPropertiesType(YcbClass.TOOL, 0.0192),
    "050_medium_clamp": ObjectPropertiesType(YcbClass.TOOL, 0.059),
    "051_large_clamp": ObjectPropertiesType(YcbClass.TOOL, 0.125),
    "052_extra_large_clamp": ObjectPropertiesType(YcbClass.TOOL, 0.202),
    "053_mini_soccer_ball": ObjectPropertiesType(YcbClass.SHAPE, 0.123),
    "054_softball": ObjectPropertiesType(YcbClass.SHAPE, 0.191),
    "055_baseball": ObjectPropertiesType(YcbClass.SHAPE, 0.148),
    "056_tennis_ball": ObjectPropertiesType(YcbClass.SHAPE, 0.058),
    "057_racquetball": ObjectPropertiesType(YcbClass.SHAPE, 0.041),
    "058_golf_ball": ObjectPropertiesType(YcbClass.SHAPE, 0.046),
    "059_chain": ObjectPropertiesType(YcbClass.SHAPE, 0.098),
    # there is no "060" in the dataset.
    "061_foam_brick": ObjectPropertiesType(YcbClass.SHAPE, 0.028),
    "062_dice": ObjectPropertiesType(YcbClass.SHAPE, 5.2e-3),
    "063-a_marbles": ObjectPropertiesType(YcbClass.SHAPE),
    "063-b_marbles": ObjectPropertiesType(YcbClass.SHAPE),
    "063-c_marbles": ObjectPropertiesType(YcbClass.SHAPE),
    "063-d_marbles": ObjectPropertiesType(YcbClass.SHAPE),
    "063-e_marbles": ObjectPropertiesType(YcbClass.SHAPE),
    "063-f_marbles": ObjectPropertiesType(YcbClass.SHAPE),
    # there is no "064" in the dataset.
    "065-a_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.013),
    "065-b_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.014),
    "065-c_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.017),
    "065-d_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.019),
    "065-e_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.021),
    "065-f_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.026),
    "065-g_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.028),
    "065-h_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.031),
    "065-i_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.035),
    "065-j_cups": ObjectPropertiesType(YcbClass.SHAPE, 0.038),
    # "066", "067", "068", "069" are not in the dataset.
    "070-a_colored_wood_blocks": ObjectPropertiesType(YcbClass.TASK, 0.0108),
    "070-b_colored_wood_blocks": ObjectPropertiesType(YcbClass.TASK, 0.0108),
    "071_nine_hole_peg_test": ObjectPropertiesType(YcbClass.TASK, 1.435),
    "072-a_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-b_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-c_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-d_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-e_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-f_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-g_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-h_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-i_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-j_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "072-k_toy_airplane": ObjectPropertiesType(YcbClass.TASK),
    "073-a_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-b_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-c_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-d_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-e_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-f_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-g_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-h_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-i_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-j_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-k_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-l_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    "073-m_lego_duplo": ObjectPropertiesType(YcbClass.TASK),
    # "074" and "075" are not in the dataset.
    "076_timer": ObjectPropertiesType(YcbClass.TASK, 0.102),
    "077_rubiks_cube": ObjectPropertiesType(YcbClass.TASK, 0.094),
}

_HAS_GOOGLE_16K = [
    # From downloaded data examination.
    "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
    "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
    "011_banana", "012_strawberry", "013_apple", "014_lemon", "015_peach",
    "016_pear", "017_orange", "018_plum", "019_pitcher_base",
    "021_bleach_cleanser", "022_windex_bottle", "024_bowl", "025_mug",
    "026_sponge", "027_skillet", "028_skillet_lid", "029_plate", "030_fork",
    "031_spoon", "032_knife", "033_spatula", "035_power_drill",
    "036_wood_block", "037_scissors", "038_padlock", "040_large_marker",
    "042_adjustable_wrench", "043_phillips_screwdriver", "044_flat_screwdriver",
    "048_hammer", "050_medium_clamp", "051_large_clamp",
    "052_extra_large_clamp", "053_mini_soccer_ball", "054_softball",
    "055_baseball", "056_tennis_ball", "057_racquetball", "058_golf_ball",
    "059_chain", "061_foam_brick", "062_dice", "063-a_marbles", "063-b_marbles",
    "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
    "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups",
    "070-a_colored_wood_blocks", "070-b_colored_wood_blocks",
    "071_nine_hole_peg_test", "072-a_toy_airplane", "072-b_toy_airplane",
    "072-c_toy_airplane", "072-d_toy_airplane", "072-e_toy_airplane",
    "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
    "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo",
    "073-g_lego_duplo", "077_rubiks_cube"
]

_HAS_TSDF = [
    # From downloaded data examination.
    "001_chips_can", "002_master_chef_can", "003_cracker_box", "004_sugar_box",
    "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
    "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana",
    "012_strawberry", "013_apple", "014_lemon", "015_peach", "016_pear",
    "017_orange", "018_plum", "019_pitcher_base", "021_bleach_cleanser",
    "022_windex_bottle", "023_wine_glass", "024_bowl", "025_mug", "026_sponge",
    "029_plate", "030_fork", "031_spoon", "032_knife", "033_spatula",
    "035_power_drill", "036_wood_block", "037_scissors", "038_padlock",
    "039_key", "040_large_marker", "041_small_marker", "042_adjustable_wrench",
    "043_phillips_screwdriver", "044_flat_screwdriver", "047_plastic_nut",
    "048_hammer", "049_small_clamp", "050_medium_clamp", "051_large_clamp",
    "052_extra_large_clamp", "053_mini_soccer_ball", "054_softball",
    "055_baseball", "056_tennis_ball", "057_racquetball", "058_golf_ball",
    "059_chain", "061_foam_brick", "062_dice", "063-a_marbles", "063-d_marbles",
    "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
    "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups",
    "070-a_colored_wood_blocks", "071_nine_hole_peg_test", "072-a_toy_airplane",
    "072-b_toy_airplane", "072-c_toy_airplane", "072-d_toy_airplane",
    "072-e_toy_airplane", "072-f_toy_airplane", "072-g_toy_airplane",
    "072-h_toy_airplane", "072-i_toy_airplane", "072-j_toy_airplane",
    "072-k_toy_airplane", "073-a_lego_duplo", "073-b_lego_duplo",
    "073-c_lego_duplo", "073-d_lego_duplo", "073-e_lego_duplo",
    "073-f_lego_duplo", "073-g_lego_duplo", "073-h_lego_duplo",
    "073-i_lego_duplo", "073-j_lego_duplo", "073-k_lego_duplo",
    "073-l_lego_duplo", "073-m_lego_duplo", "076_timer", "077_rubiks_cube"
]

_DO_NOT_USE_IDS = [
    # Missing texture information in all supported formats.
    "039_key", "047_plastic_nut", "063-d_marbles", "072-f_toy_airplane",
    "072-g_toy_airplane", "072-i_toy_airplane", "072-j_toy_airplane",
]

OBJECT_IDS = set(_HAS_GOOGLE_16K).union(set(_HAS_TSDF)) - set(_DO_NOT_USE_IDS)
OBJECT_PROPERTIES = {
    k: v for k, v in _ALL_OBJECT_PROPERTIES.items() if k in OBJECT_IDS
}

PROP_SETS = {
    "ycb_demo_static_set1": (
        # graspable red objects.
        YcbDatasetVersion.V1,
        ("013_apple", "007_tuna_fish_can", "056_tennis_ball")),
}


class YcbProp(mesh_object.MeshProp):
  """Represents an object originated in XML and meshes."""

  def _build(self,
             obj_id: str,
             dataset_path: str = _DATASET_PATH,
             load_collision_meshes: bool = True,
             name: Optional[str] = "ycb_object",
             size: Optional[List[float]] = None,
             color: Optional[str] = None,
             pos: Optional[List[float]] = None,
             mjcf_model_export_dir: Optional[str] = None,
             mass: Optional[float] = None):

    size = size or [1.] * 3

    if obj_id in _HAS_GOOGLE_16K:
      object_path = os.path.join(dataset_path, obj_id, "google_16k")
    elif obj_id in _HAS_TSDF:
      object_path = os.path.join(dataset_path, obj_id, "tsdf")
    else:
      raise ValueError(f"No mesh directory exists for {obj_id}")
    visual_meshes = list(glob.glob(os.path.join(object_path, "*.obj")))
    if not visual_meshes:
      raise ValueError(f"No visual mesh files in '{object_path}'")
    collision_meshes = list(
        glob.glob(object_path + "*.stl")) if load_collision_meshes else None
    image_files = list(glob.glob(os.path.join(object_path, "*.png")))
    # We expect a single texture image in the directory.
    if len(image_files) != 1:
      raise ValueError(f"{len(image_files)} PNG textures in '{object_path}'")
    texture_file = image_files[0]
    if mass:
      masses = [mass]
    elif obj_id in OBJECT_PROPERTIES and OBJECT_PROPERTIES[obj_id].mass:
      masses = [OBJECT_PROPERTIES[obj_id].mass]
    else:
      raise ValueError(
          f"Mass not specified and default mass not available for prop {obj_id}."
      )
    return super()._build(
        visual_meshes=visual_meshes,
        collision_meshes=collision_meshes,
        texture_file=texture_file,
        name=name,
        size=size,
        color=color,
        pos=pos,
        masses=masses,
        mjcf_model_export_dir=mjcf_model_export_dir)
