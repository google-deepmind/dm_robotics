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

# python3
"""Utilities for subtask logging."""

from typing import Sequence


def compute_return(episode_rewards: Sequence[float],
                   episode_discounts: Sequence[float]) -> float:
  """Computes the return of an episode from a list of rewards and discounts."""
  if len(episode_rewards) <= 0:
    raise ValueError('Length of episode_rewards must be greater than zero.')
  if len(episode_discounts) <= 0:
    raise ValueError('Length of episode_discounts must be greater than zero.')
  if len(episode_rewards) != len(episode_discounts):
    raise ValueError('episode_rewards and episode_discounts must be same length'
                     ' but are {episode_rewards} and {episode_discounts}')
  episode_return = episode_rewards[0]
  total_discount = episode_discounts[0]
  for reward, discount in zip(episode_rewards[1:],
                              episode_discounts[1:]):
    episode_return += reward * total_discount
    total_discount *= discount

  return episode_return
