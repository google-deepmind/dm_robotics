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
"""Utility functions, currently for terminal logging."""

import contextlib
import sys
from absl import logging
from dm_robotics.agentflow import core

# ANSI color codes for pretty-printing
ANSI_COLORS = {
    'pink': '\033[95m',
    'blue': '\033[94m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    'end': '\033[0m'
}


def log_warning(string, color=None):
  if color is None or not sys.stderr.isatty():
    logging.warning(string)
  else:
    logging.warning('%s%s%s', ANSI_COLORS[color], string, ANSI_COLORS['end'])


def log_info(string, color=None):
  if color is None or not sys.stderr.isatty():
    logging.info(string)
  else:
    logging.info('%s%s%s', ANSI_COLORS[color], string, ANSI_COLORS['end'])


def log_termination_reason(cur_option: core.Option,
                           option_result: core.OptionResult):
  """Pretty-print a status update."""
  termination_reason = option_result.termination_reason
  if termination_reason == core.TerminationType.SUCCESS:
    text = 'Option \"{}\" successful. {}'.format(cur_option.name,
                                                 option_result.termination_text)
    color = option_result.termination_color or 'green'
    log_info(text, color)

  elif termination_reason == core.TerminationType.FAILURE:
    text = 'Option \"{}\" failed. {}'.format(cur_option.name,
                                             option_result.termination_text)
    color = option_result.termination_color or 'red'
    log_warning(text, color)

  elif termination_reason == core.TerminationType.PREEMPTED:
    text = 'Option \"{}\" preempted. {}'.format(cur_option.name,
                                                option_result.termination_text)
    color = option_result.termination_color or 'yellow'
    log_warning(text, color)

  else:
    raise ValueError('Unknown exit code from subtask.')


if hasattr(contextlib, 'nullcontext'):
  nullcontext = contextlib.nullcontext  # pylint: disable=invalid-name
else:
  nullcontext = contextlib.suppress  # pylint: disable=invalid-name
