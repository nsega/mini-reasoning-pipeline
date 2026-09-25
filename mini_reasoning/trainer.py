"""The GRPO training loop. Role 2 (reward).

grpo.py owns the group-relative advantages and the loss; this module is
the loop around them, and it carries the lab's harness lessons. The
decisions are recorded in docs/design-notes.md.

- Generation runs under no_grad, never inference_mode: an
  inference-mode tensor cannot re-enter autograd, and the rollouts are
  scored again with gradients right after.
- The reward is boxed-only (fallback=None), which is the strictness
  role 2 depends on. The verifier is the same one both evals use; only
  extraction strictness differs.
- Backward runs one rollout at a time and each call is scaled by 1/G,
  because grpo_loss averages only what it is handed. Six live graphs
  OOM'd 18.7GB on MPS, and the scaling is what keeps the accumulated
  step identical to one batched call.
- The gradient norm is clipped to 1.0; pre-clip norms ran 260-400x that
  bound, so the clip is doing the work and the raw norm is reported.
- NO KL BY DEFAULT: the signed-logratio penalty is the sole gradient on
  a zero-advantage step and collapsed the policy in 40 steps at this
  scale. There is no reference model here to compute one from.
- A flat group is skipped entirely rather than stepped on a zero
  gradient: Adam carries momentum, so a step taken on zeros still moves
  the weights, and an all-same-reward group must not move anything.
- A flat problem is also retired, and the budget counts updates rather
  than draws, so training makes further passes over the problems that
  still give gradient instead of spending draws on ones that do not.
"""
from dataclasses import dataclass

import torch

from mini_reasoning import grpo, verifier
from mini_reasoning.evaluate import PROMPT

MAX_GRAD_NORM = 1.0


@dataclass(frozen=True)
class Rollouts:
    """One problem's group of sampled continuations.

    Attributes:
        sequences: Shape (group, prompt + generated), the token ids.
        prompt_len: Where the generated tail starts.
        texts: The decoded continuation per row, for grading.
    """
    sequences: torch.Tensor
    prompt_len: int
    texts: list[str]


def sample_rollouts(model, tokenizer, problem: str, n: int,
                    max_new_tokens: int = 512, temperature: float = 0.7,
                    top_p: float = 0.95) -> Rollouts:
    """Generates one problem's group, off the gradient path.

    The prompt is the one the eval path uses, imported rather than
    repeated: a policy trained on one format and graded on another is
    measuring the format. The sampling settings are the eval's too,
    because the reward is boxed-only and the probe behind that choice
    found the base model boxing 1 of 3 at temperature 1.0 against 3 of
    3 at 0.7, so hotter rollouts would make most groups uniformly
    zero-reward, flat, and therefore no gradient at all.

    Args:
        model: The policy.
        tokenizer: Its tokenizer.
        problem: The problem statement.
        n: Rollouts to sample.
        max_new_tokens: Generation cap per rollout.
        temperature: Sampling temperature.
        top_p: Nucleus sampling cutoff.

    Returns:
        The group, with the token ids the backward pass rescores.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    inputs = tokenizer(PROMPT.format(problem=problem), return_tensors="pt")
    inputs = inputs.to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        sequences = model.generate(
            **inputs, do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id)
    texts = tokenizer.batch_decode(sequences[:, prompt_len:],
                                   skip_special_tokens=True)
    return Rollouts(sequences=sequences, prompt_len=prompt_len, texts=texts)


def rollout_rewards(texts, ground_truth: str) -> torch.Tensor:
    """Binary reward per rollout, graded boxed-only.

    Role 2's strictness is not open: the reward's type sets the training
    regime, so the lenient extraction the eval roles use is not passed
    here. An unparseable sample grades False rather than raising, so a
    malformed rollout cannot end a run mid-training.

    Args:
        texts: The decoded rollouts.
        ground_truth: The problem's answer.

    Returns:
        Shape (group,), 1.0 where the boxed answer is equivalent.
    """
    return torch.tensor(
        [float(verifier.verify_answer(
            verifier.extract_final_candidate(text), ground_truth))
         for text in texts])


def sequence_logprobs(model, rollouts: Rollouts) -> torch.Tensor:
    """One differentiable logprob per rollout, averaged over its tokens.

    The mean rather than the sum, so a long rollout does not outweigh a
    short one before the advantages are applied. That choice puts a
    further 1/T on the gradient, which grpo.py's docstring names as the
    harness's to make.

    Args:
        model: The policy, called with the full sequences.
        rollouts: The group to score.

    Returns:
        Shape (group,), with a graph back to the policy.
    """
    logits = model(rollouts.sequences).logits[:, rollouts.prompt_len - 1:-1]
    realised = rollouts.sequences[:, rollouts.prompt_len:]
    chosen = logits.gather(-1, realised.unsqueeze(-1)).squeeze(-1)
    return (chosen - logits.logsumexp(-1)).mean(-1)


def _one_rollout(rollouts: Rollouts, index: int) -> Rollouts:
    """The group of one this rollout is, for its own backward pass."""
    return Rollouts(sequences=rollouts.sequences[index:index + 1],
                    prompt_len=rollouts.prompt_len,
                    texts=rollouts.texts[index:index + 1])


def train_step(model, optimizer, problem, tokenizer, *, rollout,
               group_size: int, max_grad_norm: float = MAX_GRAD_NORM) -> dict:
    """Samples one problem, scores the group, and takes at most one step.

    Args:
        model: The policy being trained.
        optimizer: Its optimizer.
        problem: A mapping with "problem", "answer" and "unique_id".
        tokenizer: Passed through to the rollout function.
        rollout: Returns Rollouts for (model, tokenizer, problem, n).
            It is responsible for generating under no_grad.
        group_size: Rollouts sampled for this problem.
        max_grad_norm: The clip bound.

    Returns:
        What the step did: the mean reward, the pre-clip gradient norm,
        and whether the optimizer moved at all.

    Raises:
        ValueError: from grpo when the group holds fewer than two
            rollouts, or when a reward is not finite.
    """
    rollouts = rollout(model, tokenizer, problem["problem"], group_size)
    rewards = rollout_rewards(rollouts.texts, problem["answer"])
    advantages = grpo.group_relative_advantages(rewards)
    report = {"unique_id": problem["unique_id"],
              "mean_reward": rewards.mean().item(),
              "grad_norm": 0.0, "stepped": False}

    if not advantages.any():
        # Skipping the step, not merely the gradient: Adam's momentum
        # would otherwise move the weights on an all-zero gradient.
        return report

    optimizer.zero_grad(set_to_none=True)
    group = len(rollouts.texts)
    for index in range(group):
        logprob = sequence_logprobs(model, _one_rollout(rollouts, index))
        loss = grpo.grpo_loss(logprob, advantages[index:index + 1])
        (loss / group).backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    report["grad_norm"] = float(norm)
    report["stepped"] = True
    return report


def train(model, tokenizer, problems, *, steps: int, rollout,
          group_size: int = 7, lr: float = 1e-6,
          max_grad_norm: float = MAX_GRAD_NORM,
          max_draws: int | None = None, seed: int | None = None) -> dict:
    """Runs GRPO until `steps` updates, cycling the problems in order.

    A draw whose group is flat takes no step and retires its problem for
    the rest of the run, whether every rollout was wrong or every one
    was right: neither gives a gradient, and at this scale a problem
    flat once is almost always flat again. The budget counts updates
    rather than draws, so the loop makes further passes over the
    problems that still give gradient, until the first of three limits:
    `steps` updates, `max_draws` draws, or no live problem left.

    Each draw is seeded from `seed` plus its index, counting from 0, so
    which problems retire repeats, and a run that skipped the baseline
    eval trains exactly like one that ran it.

    Args:
        model: The policy to train, in place.
        tokenizer: Passed through to the rollout function.
        problems: The training pool, cycled in order.
        steps: Optimizer updates to reach.
        rollout: Returns Rollouts for (model, tokenizer, problem, n).
        group_size: Rollouts per problem.
        lr: Adam learning rate.
        max_grad_norm: The clip bound.
        max_draws: A runtime cap on draws; None means 3 * steps. It may
            sit below steps.
        seed: Seeds each draw as seed + draw; None leaves the global
            RNG alone.

    Returns:
        "steps", one report per draw in order (the name predates dynamic
        sampling, and stored runs depend on it), and "updates", "draws"
        and "stopped": which limit ended the run, "updates",
        "max_draws" or "pool_exhausted", checked in that order.

    Raises:
        ValueError: if steps or max_draws is below one, or problems is
            empty.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not problems:
        raise ValueError("train needs at least one problem")
    if max_draws is None:
        max_draws = 3 * steps
    if max_draws < 1:
        raise ValueError(f"max_draws must be >= 1, got {max_draws}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    pool = {p["unique_id"] for p in problems}
    history, retired = [], set()
    updates = draws = position = 0
    while updates < steps and draws < max_draws and retired != pool:
        problem = problems[position % len(problems)]
        position += 1
        if problem["unique_id"] in retired:
            continue
        if seed is not None:
            torch.manual_seed(seed + draws)
        report = train_step(model, optimizer, problem, tokenizer,
                            rollout=rollout, group_size=group_size,
                            max_grad_norm=max_grad_norm)
        history.append(report)
        draws += 1
        if report["stepped"]:
            updates += 1
        else:
            retired.add(problem["unique_id"])
    if updates == steps:
        stopped = "updates"
    elif draws == max_draws:
        stopped = "max_draws"
    else:
        stopped = "pool_exhausted"
    return {"steps": history, "updates": updates, "draws": draws,
            "stopped": stopped}
