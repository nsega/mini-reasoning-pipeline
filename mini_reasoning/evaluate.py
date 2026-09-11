"""Before/after evaluation on the frozen subset. Roles 1 and 3.

The same function runs the baseline eval before training and the
validation eval after it, on the same frozen subset, with the same
scorer and the same extraction fallback. Role 3 grades with the verifier
that trained the model in role 2: that circularity is the point, and the
README discusses it. The full record is in docs/design-notes.md.

- evaluate builds the material bundle the scorer protocol expects (text
  plus a logprob summary per sample) and stores it in the per-problem
  record, so a composite can be re-run offline over the record. Every
  lab discovery came from re-analysing raw records.
- Sampling is the one model-bound step and sits behind the sampler
  argument, so everything else is tested against real scorers and the
  real verifier. The default sampler wraps Hugging Face generation.
- Selection is the caller's argmax, checked for the all-rejected case
  first: a composite that rejects everything returns all -inf and says
  nothing more, and a bare argmax would silently pick index 0.
- Accuracy is reported raw and reweighted to MATH-500's level mix. The
  seed-42 subset skews hard (mean level 3.88 against 3.44 for the full
  set), so the raw number understates the model.
"""
from collections.abc import Callable, Iterable, Mapping, Sequence

import torch

from mini_reasoning import verifier
from mini_reasoning.scorers import REJECTED, Candidate, Scorer

MATH500_LEVEL_COUNTS = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}

Sampler = Callable[[object, object, str, int], Sequence[Candidate]]

_PROMPT = ("Problem: {problem}\n"
           "Solve the problem. Put the final answer in \\boxed{{}}.\n"
           "Solution:")


def sample_solutions(model, tokenizer, problem: str, n: int,
                     max_new_tokens: int = 512, temperature: float = 0.7,
                     top_p: float = 0.95) -> list[Candidate]:
    """Samples n solutions and summarises each by its mean token logprob.

    The prompt is plain text rather than the tokenizer's chat template:
    the pipeline's model is a base model, and on Qwen3-0.6B-Base the
    chat template made it regurgitate prompts (0 of 3 usable) where the
    plain prompt at temperature 0.7 boxed 3 of 3.

    The summary is taken from the raw logits rather than from the
    processed scores. The scores have already been divided by the
    temperature and truncated to the nucleus, so summarising them would
    rank candidates by the sampling distribution instead of by the
    model: on one 8-sample probe the two orderings differed.

    Args:
        model: A causal LM with generate and compute_transition_scores.
        tokenizer: Its tokenizer.
        problem: The problem statement.
        n: Number of independent samples to draw.
        max_new_tokens: Generation cap per sample.
        temperature: Sampling temperature; 1.0 boxed 1 of 3 on the same
            probe where 0.7 boxed 3 of 3.
        top_p: Nucleus sampling cutoff.

    Returns:
        One Candidate per sample: the decoded text and the mean logprob
        the model assigned to its generated tokens, padding excluded.
    """
    eos_ids = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]
    inputs = tokenizer(_PROMPT.format(problem=problem), return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            output_logits=True, return_dict_in_generate=True,
            pad_token_id=pad_id)
        token_logprobs = model.compute_transition_scores(
            out.sequences, out.logits, normalize_logits=True)
    generated = out.sequences[:, inputs["input_ids"].shape[1]:]
    summaries = sequence_logprob_summary(token_logprobs, generated, eos_ids)
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [Candidate(text=text, logprob_summary=float(summary))
            for text, summary in zip(texts, summaries)]


def _eos_ids(model, tokenizer) -> tuple[int, ...]:
    """The ids generation actually stops on, as a tuple.

    Taken from the generation config rather than the tokenizer: a chat
    checkpoint often stops on several ids, and one the tokenizer does
    not name would leave an early-finished row looking unfinished.
    """
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    return (eos,) if isinstance(eos, int) else tuple(eos)


def sequence_logprob_summary(token_logprobs: torch.Tensor,
                             generated: torch.Tensor,
                             eos_ids: int | Iterable[int]) -> torch.Tensor:
    """Mean token logprob per row, up to and including the first EOS.

    Positions after the first stop token are padding. They can carry
    -inf, so they are excluded by selection rather than by a
    multiplicative mask, which would turn -inf times zero into NaN. A
    row that never stopped averages every token.

    Args:
        token_logprobs: Shape (rows, steps), one logprob per generated
            token.
        generated: Shape (rows, steps), the generated token ids.
        eos_ids: The id generation stops on, or every such id.

    Returns:
        Shape (rows,), the per-row mean.
    """
    if isinstance(eos_ids, int):
        eos_ids = (eos_ids,)
    stops = torch.tensor(tuple(eos_ids), device=generated.device)
    is_eos = torch.isin(generated, stops)
    finished_before = is_eos.cumsum(-1) - is_eos.int()
    keep = finished_before == 0
    kept = token_logprobs.masked_fill(~keep, 0.0)
    return kept.sum(-1) / keep.sum(-1).clamp(min=1)


_REQUIRED_FIELDS = ("unique_id", "problem", "answer", "level")


def _validate(problems: Sequence[Mapping]) -> None:
    """Checks every problem before the sampling loop runs.

    The reweighting looks each level up in MATH-500's counts, so an
    unknown level would otherwise raise after a whole run of generation
    and return nothing.
    """
    for position, problem in enumerate(problems):
        missing = [f for f in _REQUIRED_FIELDS if f not in problem]
        if missing:
            raise ValueError(
                f"problem {position} is missing {', '.join(missing)}")
        if problem["level"] not in MATH500_LEVEL_COUNTS:
            raise ValueError(
                f"problem {position} has level {problem['level']!r}, not "
                f"one of {sorted(MATH500_LEVEL_COUNTS)}")


def _select(scores: Sequence[float]) -> int | None:
    """Index of the first maximum, or None when every score is REJECTED."""
    if all(score == REJECTED for score in scores):
        return None
    return max(range(len(scores)), key=scores.__getitem__)


def _level_reweighted_accuracy(records: Sequence[Mapping]) -> float:
    """Accuracy under MATH-500's level shares, renormalised over the
    levels present so a level missing from the subset drops out rather
    than zeroing the total."""
    correct: dict[int, int] = {}
    total: dict[int, int] = {}
    for record in records:
        level = record["level"]
        total[level] = total.get(level, 0) + 1
        correct[level] = correct.get(level, 0) + int(record["correct"])
    weight = sum(MATH500_LEVEL_COUNTS[level] for level in total)
    return sum(MATH500_LEVEL_COUNTS[level] / weight * correct[level]
               / total[level] for level in total)


def evaluate(model, tokenizer, problems: Sequence[Mapping], scorer: Scorer,
             *, n_samples: int = 7, fallback: str | None = None,
             sampler: Sampler = sample_solutions) -> dict:
    """Samples, selects and grades every problem, and stores the record.

    Args:
        model: Passed through to the sampler.
        tokenizer: Passed through to the sampler.
        problems: Mappings with "unique_id", "problem", "answer" and
            "level", as in the frozen MATH-500 subset.
        scorer: Any Scorer; selection is argmax over its scores.
        n_samples: Samples drawn per problem, the candidate-set size.
        fallback: Extraction fallback for grading. None is boxed-only,
            the strict default; both eval call sites pass "number".
        sampler: Returns the material bundle for one problem.

    Returns:
        A dict with "accuracy", "level_reweighted_accuracy" and
        "records", one record per problem in input order holding the
        samples, extracted candidates, scores, selected index and
        correctness.

    Raises:
        ValueError: if problems is empty, if any problem lacks a
            required field or carries a level outside MATH-500's five,
            or if the sampler returns other than n_samples samples.
    """
    if not problems:
        raise ValueError("evaluate needs at least one problem")
    _validate(problems)
    records = []
    for problem in problems:
        samples = list(sampler(model, tokenizer, problem["problem"],
                               n_samples))
        if len(samples) != n_samples:
            raise ValueError(
                f"sampler returned {len(samples)} samples for problem "
                f"{problem['unique_id']!r}, expected {n_samples}")
        candidates = [verifier.extract_final_candidate(s.text, fallback)
                      for s in samples]
        scores = list(scorer.score(samples))
        selected = _select(scores)
        correct = selected is not None and verifier.verify_answer(
            candidates[selected], problem["answer"])
        records.append({
            "unique_id": problem["unique_id"],
            "level": problem["level"],
            "answer": problem["answer"],
            "samples": [{"text": s.text, "logprob_summary": s.logprob_summary}
                        for s in samples],
            "candidates": candidates,
            "scores": scores,
            "selected": selected,
            "correct": bool(correct),
        })
    accuracy = sum(r["correct"] for r in records) / len(records)
    return {
        "accuracy": accuracy,
        "level_reweighted_accuracy": _level_reweighted_accuracy(records),
        "records": records,
    }
