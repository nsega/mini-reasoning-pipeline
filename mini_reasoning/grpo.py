"""GRPO loss with group-relative advantage. Role 2 wiring (reward).

To be self-written by Naoki (port of the lab's module 04, self-written
and migration-eligible, including the group_size >= 2 ValueError
guard). Lab lessons to carry in:

- group_relative_advantages(rewards): per-group mean/std with epsilon;
  zero-variance groups return exact zeros (this is what lets updates
  run on all-same-reward groups safely).
- grpo_loss(logprobs, advantages): advantages detached inside; scalar
  for gradient DESCENT (descent raises positive-advantage logprobs).
- Trainer-level lessons (harness, not this module): clip grad to norm
  1.0 (pre-clip norms ran 260-400x the bound), accumulate backward per
  rollout (six live graphs OOM'd 18.7GB on MPS), generation under
  no_grad never inference_mode, and NO-KL DEFAULT: the signed-logratio
  penalty is the sole gradient on zero-advantage steps and collapsed
  the policy in 40 steps at this scale.
"""


def group_relative_advantages(rewards):
    raise NotImplementedError("capstone self-written implementation (Naoki)")


def grpo_loss(logprobs, advantages):
    raise NotImplementedError("capstone self-written implementation (Naoki)")
