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

"""Defines Robotiq 2-finger 85 adaptive gripper constants."""

import os  # pylint: disable=unused-import


NO_GRASP = 1  # no object grasped
INWARD_GRASP = 2  # grasp while moving inwards
OUTWARD_GRASP = 3  # grasp while moving outwards


# pylint: disable=line-too-long
# XML path of the Robotiq 2F85 robot hand.
XML_PATH = (os.path.join(os.path.dirname(__file__), '..',  '..', 'vendor', 'robotiq_beta_robots', 'mujoco', 'robotiq_2f85_v2.xml'))
# pylint: enable=line-too-long

# Located at the center of the finger contact surface.
TCP_SITE_POS = (0., 0., 0.1489)
