# AgentFlow

`AgentFlow` is a library for composing Reinforcement-Learning agents. The core
features that AgentFlow provides are:

1.  tools for slicing, transforming, and composing *specs*
2.  tools for encapsulating and composing RL-tasks.

## Why do we need another RL framework?

AgentFlow grew from the need to define RL problems on robots that operate only
on **subsets of the action-space**, e.g. cartesian-movement of the left arm, and
on **subtasks of an overall problem**, e.g. insertion, in a reach-insert-reset
task.
