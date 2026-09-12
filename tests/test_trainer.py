"""Contract tests for the GRPO trainer harness.

grpo.py owns the advantages and the loss; everything around them is
here, and the lab's harness lessons are what these pin: generation off
the gradient path, boxed-only reward, per-rollout backward scaled so it
matches one batched call, the clip, and no KL term.

The policy is a real (tiny) causal LM rather than a double, so the
gradients are real gradients. Generation is injected, since only that
step needs a trained model.
"""
import pytest
import torch
from torch import nn

from mini_reasoning import grpo
from mini_reasoning.trainer import (
    Rollouts, rollout_rewards, sequence_logprobs, train, train_step,
)

VOCAB, PROMPT_LEN = 32, 3


class TinyPolicy(nn.Module):
    """A real causal LM, small enough to differentiate in a test."""

    def __init__(self, vocab=VOCAB, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids):
        return type("Out", (), {"logits": self.head(self.embed(input_ids))})


def policy(seed=0):
    torch.manual_seed(seed)
    return TinyPolicy()


def rollouts_of(texts, width=4, seed=0):
    """Token ids nobody reads meaning into, paired with given texts."""
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (len(texts), PROMPT_LEN + width))
    return Rollouts(sequences=ids, prompt_len=PROMPT_LEN, texts=list(texts))


def rollout_of(texts):
    return lambda model, tokenizer, problem, n: rollouts_of(texts)


PROBLEM = {"unique_id": "p", "problem": "What is 6*7?", "answer": "42",
           "level": 1}
RIGHT = r"so \boxed{42}"
WRONG = r"so \boxed{41}"


def params(model):
    return [p.detach().clone() for p in model.parameters()]


def moved(before, model):
    return any(not torch.equal(b, a)
               for b, a in zip(before, model.parameters()))


class TestReward:
    """Role 2 is boxed-only: the reward's type sets the regime."""

    def test_a_boxed_right_answer_scores_one(self):
        got = rollout_rewards([RIGHT, WRONG], "42")
        assert got.tolist() == [1.0, 0.0]

    def test_a_bare_number_is_not_rewarded(self):
        """The eval roles rescue this one; the reward role must not, or
        the training signal stops being the one the stub promised."""
        assert rollout_rewards(["the answer is 42"], "42").tolist() == [0.0]

    def test_the_reward_is_float_not_bool(self):
        """group_relative_advantages casts, but a bool tensor reaching
        it at all means the harness handed over the raw verdicts."""
        assert rollout_rewards([RIGHT], "42").dtype.is_floating_point


class TestSequenceLogprobs:
    """One differentiable scalar per rollout, length-normalised."""

    def test_one_scalar_per_rollout(self):
        model = policy()
        got = sequence_logprobs(model, rollouts_of([RIGHT, WRONG]))
        assert got.shape == (2,)

    def test_it_carries_a_gradient_back_to_the_policy(self):
        model = policy()
        sequence_logprobs(model, rollouts_of([RIGHT])).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())

    def test_length_does_not_change_the_scale(self):
        """The mean over tokens, not the sum: a long rollout must not
        outweigh a short one before the advantages are even applied."""
        model = policy()
        short = sequence_logprobs(model, rollouts_of([RIGHT], width=2))
        long = sequence_logprobs(model, rollouts_of([RIGHT], width=16))
        assert abs(short.item()) < 10 and abs(long.item()) < 10


class TestAccumulation:
    """Per-rollout backward must equal one batched call."""

    def test_accumulated_gradient_matches_the_batched_one(self):
        """Six live graphs OOM'd 18.7GB on MPS, so the harness goes one
        rollout at a time and scales by 1/G. That scaling is what makes
        the two paths the same step."""
        rollouts = rollouts_of([RIGHT, WRONG, RIGHT])
        rewards = rollout_rewards(rollouts.texts, "42")
        advantages = grpo.group_relative_advantages(rewards)

        batched = policy()
        grpo.grpo_loss(sequence_logprobs(batched, rollouts),
                       advantages).backward()

        stepwise = policy()
        for index in range(len(rollouts.texts)):
            one = Rollouts(sequences=rollouts.sequences[index:index + 1],
                           prompt_len=rollouts.prompt_len,
                           texts=rollouts.texts[index:index + 1])
            loss = grpo.grpo_loss(sequence_logprobs(stepwise, one),
                                  advantages[index:index + 1])
            (loss / len(rollouts.texts)).backward()

        for want, got in zip(batched.parameters(), stepwise.parameters()):
            assert torch.allclose(want.grad, got.grad, atol=1e-6)


class TestFlatGroups:
    """An all-same-reward group must leave the policy exactly alone."""

    def test_a_flat_group_does_not_move_the_parameters(self):
        """Zero gradients are not enough: Adam carries momentum, so a
        step taken on them still moves the weights. The harness has to
        skip the step, not just arrive at a zero gradient."""
        model = policy()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        train_step(model, optimizer, PROBLEM, None,
                   rollout=rollout_of([RIGHT, WRONG]), group_size=2)
        before = params(model)
        report = train_step(model, optimizer, PROBLEM, None,
                            rollout=rollout_of([RIGHT, RIGHT]), group_size=2)
        assert report["stepped"] is False
        assert not moved(before, model)

    def test_a_split_group_does_move_them(self):
        model = policy()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        before = params(model)
        report = train_step(model, optimizer, PROBLEM, None,
                            rollout=rollout_of([RIGHT, WRONG]), group_size=2)
        assert report["stepped"] is True
        assert moved(before, model)


class TestClipping:
    """Pre-clip norms ran 260-400x the bound, so the clip is load-bearing."""

    def test_the_applied_gradient_norm_is_the_bound(self):
        model = policy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        report = train_step(model, optimizer, PROBLEM, None,
                            rollout=rollout_of([RIGHT, WRONG]), group_size=2,
                            max_grad_norm=1e-4)
        assert report["grad_norm"] > 1e-4
        applied = torch.cat([p.grad.flatten() for p in model.parameters()])
        assert applied.norm().item() == pytest.approx(1e-4, rel=1e-3)

    def test_the_report_carries_the_pre_clip_norm(self):
        model = policy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        report = train_step(model, optimizer, PROBLEM, None,
                            rollout=rollout_of([RIGHT, WRONG]), group_size=2)
        assert report["grad_norm"] > 0


class TestTrain:
    """The loop over steps, and what it reports."""

    def test_it_runs_the_requested_number_of_steps(self):
        model = policy()
        history = train(model, None, [PROBLEM], steps=3,
                        rollout=rollout_of([RIGHT, WRONG]), group_size=2,
                        lr=1e-3)
        assert len(history["steps"]) == 3

    def test_each_step_reports_its_reward_and_whether_it_stepped(self):
        model = policy()
        history = train(model, None, [PROBLEM], steps=1,
                        rollout=rollout_of([RIGHT, WRONG]), group_size=2,
                        lr=1e-3)
        report = history["steps"][0]
        assert report["mean_reward"] == pytest.approx(0.5)
        assert report["stepped"] is True
        assert report["unique_id"] == "p"

    def test_it_cycles_the_problems(self):
        other = dict(PROBLEM, unique_id="q")
        history = train(policy(), None, [PROBLEM, other], steps=3,
                        rollout=rollout_of([RIGHT, WRONG]), group_size=2,
                        lr=1e-3)
        assert [s["unique_id"] for s in history["steps"]] == ["p", "q", "p"]

    def test_zero_steps_raises(self):
        with pytest.raises(ValueError):
            train(policy(), None, [PROBLEM], steps=0,
                  rollout=rollout_of([RIGHT]), group_size=2)

    def test_a_group_of_one_raises(self):
        """grpo.group_relative_advantages guards this, and the flag that
        sets it reaches here, so the harness must not hide the error."""
        with pytest.raises(ValueError):
            train(policy(), None, [PROBLEM], steps=1,
                  rollout=rollout_of([RIGHT]), group_size=1)
