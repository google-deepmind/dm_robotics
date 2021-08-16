# AgentFlow: A Modular Toolkit for Scalable RL Research

## Overview

`AgentFlow` is a library for composing Reinforcement-Learning agents. The core
features that AgentFlow provides are:

1.  tools for slicing, transforming, and composing *specs*
2.  tools for encapsulating and composing RL-tasks.

Unlike the standard RL setup, which assumes a single environment and an agent,
`AgentFlow` is designed for the single-embodiment, multiple-task regime. This
was motivated by the robotics use-case, which frequently requires training RL
modules for various skills, and then composing them (possibly with non-learned
controllers too).

Instead of having to implement a separate RL environment for each skill and
combine them ad hoc, with `AgentFlow` you can define one or more `SubTasks`
which *modify* a timestep from a single top-level environment, e.g. adding
observations and defining rewards, or isolating a particular sub-system of the
environment, such as a robot arm.

You then *compose* SubTasks with regular RL-agents to form modules, and use a
set of graph-building operators to define the flow of these modules over time
(hence the name `AgentFlow`).

The graph-building step is entirely optional, and is intended only for use-cases
that require something like a (possibly learnable, possibly stochastic)
state-machine.

### [Components](docs/components.md)
### [Control Flow](docs/control_flow.md)
### [Examples](docs/examples.md)
