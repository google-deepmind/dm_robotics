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

"""Top-level names for agentflow."""

from dm_robotics.agentflow.action_spaces import CastActionSpace  # pytype: disable=pyi-error
from dm_robotics.agentflow.action_spaces import CompositeActionSpace  # pytype: disable=pyi-error
from dm_robotics.agentflow.action_spaces import FixedActionSpace  # pytype: disable=pyi-error
from dm_robotics.agentflow.action_spaces import prefix_slicer  # pytype: disable=pyi-error
from dm_robotics.agentflow.action_spaces import SequentialActionSpace  # pytype: disable=pyi-error
from dm_robotics.agentflow.action_spaces import ShrinkToFitActionSpace  # pytype: disable=pyi-error

from dm_robotics.agentflow.core import ActionSpace
from dm_robotics.agentflow.core import Arg
from dm_robotics.agentflow.core import ArgSpec
from dm_robotics.agentflow.core import IdentityActionSpace
from dm_robotics.agentflow.core import MetaOption
from dm_robotics.agentflow.core import Option
from dm_robotics.agentflow.core import OptionResult
from dm_robotics.agentflow.core import Policy
from dm_robotics.agentflow.core import TerminationType

from dm_robotics.agentflow.loggers.subtask_logger import Aggregator
from dm_robotics.agentflow.loggers.subtask_logger import EpisodeReturnAggregator
from dm_robotics.agentflow.loggers.subtask_logger import SubTaskLogger

from dm_robotics.agentflow.meta_options.control_flow.cond import Cond
from dm_robotics.agentflow.meta_options.control_flow.loop_ops import Repeat
from dm_robotics.agentflow.meta_options.control_flow.loop_ops import While
from dm_robotics.agentflow.meta_options.control_flow.sequence import Sequence

from dm_robotics.agentflow.options.basic_options import all_terminate
from dm_robotics.agentflow.options.basic_options import any_terminates
from dm_robotics.agentflow.options.basic_options import ArgAdaptor
from dm_robotics.agentflow.options.basic_options import ConcurrentOption
from dm_robotics.agentflow.options.basic_options import DelegateOption
from dm_robotics.agentflow.options.basic_options import FixedOp
from dm_robotics.agentflow.options.basic_options import LambdaOption
from dm_robotics.agentflow.options.basic_options import OptionAdapter
from dm_robotics.agentflow.options.basic_options import options_terminate
from dm_robotics.agentflow.options.basic_options import PadOption
from dm_robotics.agentflow.options.basic_options import PolicyAdapter
from dm_robotics.agentflow.options.basic_options import RandomOption

from dm_robotics.agentflow.preprocessors.timestep_preprocessor import PreprocessorTimestep
from dm_robotics.agentflow.preprocessors.timestep_preprocessor import TimestepPreprocessor

from dm_robotics.agentflow.subtask import SubTask
from dm_robotics.agentflow.subtask import SubTaskObserver
from dm_robotics.agentflow.subtask import SubTaskOption

from dm_robotics.agentflow.util import log_info
from dm_robotics.agentflow.util import log_termination_reason
