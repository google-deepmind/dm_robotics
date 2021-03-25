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

"""Module defining the interface for an image detector."""

import abc
from typing import Callable, Tuple

from dmr_vision import types
import numpy as np

Signature = Callable[[np.ndarray], Tuple[types.Centers, types.Detections]]


class ImageDetector(abc.ABC):
  """Image-based blob detector."""

  @abc.abstractmethod
  def __call__(self,
               image: np.ndarray) -> Tuple[types.Centers, types.Detections]:
    """Detects something of interest in an image.

    Args:
      image: the input image.

    Returns:
      A dictionary mapping a detection name with
       - the (u, v) coordinate of its barycenter, if found;
       - `None`, otherwise or other conditions are met;
      and a dictionary mapping a detection name with
       - its contour superimposed on the input image;
       - `None`, otherwise or other conditions are met;
    """
    raise NotImplementedError
