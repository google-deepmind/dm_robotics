# AgentFlow Examples

[TOC]

## Modeling a simple insertion task.

```python

  # Define a subtask that exposes the desired RL-environment view on `base_task`
  my_subtask = MySubTask(env.observation_spec(), 'Insertion SubTask')

  # Define a regular RL agent against this task-spec.
  my_policy = MyPolicy(my_subtask.action_spec(),
                       my_subtask.observation_spec(), 'My Policy')

  # Compose the policy and subtask to form an Option module.
  learned_insert_option = subtask.SubTaskOption(
      my_subtask, my_policy, name='Learned Insertion')

  reach_option = MyOption(env.action_spec(), env.observation_spec(),
                          'Reach for Socket')
  scripted_reset = MyOption(env.action_spec(), env.observation_spec(),
                            'Scripted Reset')
  extract_option = MyOption(env.action_spec(), env.observation_spec(),
                            'Extract')
  recovery_option = MyOption(env.action_spec(),
                             env.observation_spec(), 'Recover')

  # Use some AgentFlow operators to embed the agent in a bigger agent.
  # First use Cond to op run learned-agent if sufficiently close.
  reach_or_insert_op = cond.Cond(
      cond=near_socket,
      true_branch=learned_insert_option,
      false_branch=reach_option,
      name='Reach or Insert')

  # Loop the insert-or-reach option 5 times.
  reach_and_insert_5x = loop_ops.Repeat(
      5, reach_or_insert_op, name='Retry Loop')

  loop_body = sequence.Sequence([
      scripted_reset,
      reach_and_insert_5x,
      cond.Cond(
          cond=last_option_successful,
          true_branch=extract_option,
          false_branch=recovery_option,
          name='post-insert')
  ])
  my_big_agent = loop_ops.While(lambda _: True, loop_body)
```

For more details see the [tutorial colab](https://github.com/deepmind/robotics/py/agentflow/).
