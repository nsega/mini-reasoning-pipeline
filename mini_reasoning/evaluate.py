"""Before/after evaluation on the frozen subset. Role 3 (validation).

To be self-written by Naoki. Contract sketch:

- evaluate(model, tokenizer, problems, scorer) -> results dict
  Same frozen 50-problem subset before and after training; the scorer
  argument is the swappable interface (role 3 uses the same verifier
  that trained the model in role 2: that circularity is the point, and
  the README must discuss it honestly).
- Report raw AND level-reweighted accuracy (the seed-42 subset skews
  hard: mean level 3.88 vs 3.44 for full MATH-500).
- Store per-problem records (samples, candidates, scores): every lab
  discovery came from being able to reanalyze raw records offline.
"""


def evaluate(model, tokenizer, problems, scorer):
    raise NotImplementedError("capstone self-written implementation (Naoki)")
