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

    def test_the_clip_bounds_what_reaches_the_policy(self):
        """Asserted on the weights, not on the gradient buffers: under
        plain SGD at lr 1.0 the parameter delta is the applied
        gradient, so this holds however the step manages its buffers."""
        model = policy()
        before = params(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        report = train_step(model, optimizer, PROBLEM, None,
                            rollout=rollout_of([RIGHT, WRONG]), group_size=2,
                            max_grad_norm=1e-4)
        assert report["grad_norm"] > 1e-4
        delta = torch.cat([(b - a.detach()).flatten()
                           for b, a in zip(before, model.parameters())])
        assert delta.norm().item() == pytest.approx(1e-4, rel=1e-3)

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


SPLIT, ALL_WRONG, ALL_RIGHT = [RIGHT, WRONG], [WRONG, WRONG], [RIGHT, RIGHT]


def problem(uid):
    return dict(PROBLEM, unique_id=uid, problem=f"problem {uid}")


def rollout_by(table):
    """Rollout texts looked up by problem statement."""
    return lambda model, tokenizer, statement, n: rollouts_of(table[statement])


def drawn(history):
    return [s["unique_id"] for s in history["steps"]]


class TestDynamicSampling:
    """steps counts updates, and a flat problem retires for the run."""

    def test_steps_counts_updates_not_draws(self):
        history = train(policy(), None, [problem("flat"), problem("live")],
                        steps=2,
                        rollout=rollout_by({"problem flat": ALL_WRONG,
                                            "problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert history["updates"] == 2 and history["draws"] == 3
        assert drawn(history) == ["flat", "live", "live"]

    def test_a_retired_problem_is_never_drawn_again(self):
        history = train(policy(), None, [problem("flat"), problem("live")],
                        steps=4,
                        rollout=rollout_by({"problem flat": ALL_WRONG,
                                            "problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert drawn(history).count("flat") == 1

    def test_an_all_right_group_retires_like_an_all_wrong_one(self):
        """All right is as gradient-free as all wrong."""
        history = train(policy(), None, [problem("solved"), problem("live")],
                        steps=3,
                        rollout=rollout_by({"problem solved": ALL_RIGHT,
                                            "problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert drawn(history).count("solved") == 1

    def test_a_problem_that_steps_then_goes_flat_retires(self):
        """The fourth update has to come from "live" alone, which only a
        retired "fading" allows; a third draw of "fading" would exhaust
        its outcomes and raise. The order is asserted whole because a
        draw-counting loop over two problems reaches "fading" exactly
        twice by coincidence, and a count alone would pass against it."""
        outcomes = iter([SPLIT, ALL_WRONG])

        def rollout(model, tokenizer, statement, n):
            if statement == "problem fading":
                return rollouts_of(next(outcomes))
            return rollouts_of(SPLIT)

        history = train(policy(), None, [problem("fading"), problem("live")],
                        steps=4, rollout=rollout, group_size=2, lr=1e-3)
        assert drawn(history) == ["fading", "live", "fading", "live", "live"]
        assert [s["stepped"] for s in history["steps"]
                if s["unique_id"] == "fading"] == [True, False]


class TestStopping:
    """Whichever limit comes first ends the run, and says which it was."""

    def test_reaching_the_target_stops_on_updates(self):
        history = train(policy(), None, [problem("live")], steps=2,
                        rollout=rollout_by({"problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert history["stopped"] == "updates" and history["draws"] == 2

    def test_the_draw_cap_stops_on_max_draws_even_below_steps(self):
        """A runtime cap, not a validation rule, so it may sit under
        steps."""
        history = train(policy(), None, [problem("live")], steps=5,
                        max_draws=2,
                        rollout=rollout_by({"problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert history["stopped"] == "max_draws"
        assert history["updates"] == 2 and history["draws"] == 2

    def test_a_pool_that_all_goes_flat_stops_on_pool_exhausted(self):
        history = train(policy(), None, [problem("a"), problem("b")],
                        steps=3,
                        rollout=rollout_by({"problem a": ALL_WRONG,
                                            "problem b": ALL_WRONG}),
                        group_size=2, lr=1e-3)
        assert history["stopped"] == "pool_exhausted"
        assert history["updates"] == 0 and history["draws"] == 2

    def test_max_draws_defaults_to_three_times_steps(self):
        pool = [problem(str(i)) for i in range(5)]
        history = train(policy(), None, pool, steps=1,
                        rollout=rollout_by({f"problem {i}": ALL_WRONG
                                            for i in range(5)}),
                        group_size=2, lr=1e-3)
        assert history["draws"] == 3 and history["stopped"] == "max_draws"

    def test_a_draw_cap_below_one_raises(self):
        with pytest.raises(ValueError, match="max_draws"):
            train(policy(), None, [PROBLEM], steps=1, max_draws=0,
                  rollout=rollout_of(SPLIT), group_size=2)


def random_rollout(model, tokenizer, statement, n):
    """Rewards drawn from torch's global RNG with nothing reseeding it,
    so only the trainer's own seeding can make two runs agree."""
    bits = torch.randint(0, 2, (n,)).tolist()
    ids = torch.randint(0, VOCAB, (n, PROMPT_LEN + 4))
    return Rollouts(sequences=ids, prompt_len=PROMPT_LEN,
                    texts=[RIGHT if bit else WRONG for bit in bits])


class TestSeeding:
    """Each draw is seeded from the run's seed plus its index."""

    def run(self, seed, disturbance):
        model = policy()
        torch.manual_seed(disturbance)  # whatever the process did first
        history = train(model, None, [problem(str(i)) for i in range(6)],
                        steps=4, rollout=random_rollout, group_size=4,
                        lr=1e-3, seed=seed)
        return [(s["unique_id"], s["mean_reward"], s["stepped"])
                for s in history["steps"]]

    def test_the_same_seed_repeats_the_run_whatever_came_before(self):
        """The --reuse-baseline gap: skipping the baseline eval leaves the
        RNG somewhere else, and training must not notice."""
        assert self.run(7, disturbance=1) == self.run(7, disturbance=2)

    def test_a_different_seed_draws_differently(self):
        assert self.run(7, disturbance=1) != self.run(8, disturbance=1)

    def test_draw_k_is_seeded_with_seed_plus_k(self):
        seen = []

        def recording(model, tokenizer, statement, n):
            seen.append(torch.initial_seed())
            return rollouts_of(SPLIT)

        train(policy(), None, [problem("live")], steps=3,
              rollout=recording, group_size=2, lr=1e-3, seed=40)
        assert seen == [40, 41, 42]
