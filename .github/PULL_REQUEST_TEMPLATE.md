Please open this PR as a draft while the design or the code is still moving, and mark it ready for review once the description and code are ready to be looked at.

Before submitting, please review:
- [Design notes](https://github.com/nsega/mini-reasoning-pipeline/blob/main/docs/design-notes.md): a change to a recorded decision updates its section
- [Provenance](https://github.com/nsega/mini-reasoning-pipeline/blob/main/README.md#provenance): what was written with AI assistance is disclosed there and in the commit trailers

Go through the following checklist
- [ ] Tests pass (`uv run --group dev pytest -q`)
- [ ] Added/updated tests, written first and watched failing
- [ ] Updated the design notes and the README (if applicable)
- [ ] Updated the Provenance section (if the authorship split changed)
- [ ] Committed runs still report byte-identical (`uv run python -m mini_reasoning.report docs/runs/<run>`)
- [ ] Included measured timings or memory (for changes affecting runtime), never estimates

---

## Spec and plan

<!-- Keep the lines that apply and delete the others. A change beyond a small fix is designed in docs/specs/ and planned in docs/plans/ before its code is written. -->

Spec: `docs/specs/<!-- date and topic -->-design.md`
Plan: `docs/plans/<!-- date and topic -->.md`
Fixes #<!-- Issue number -->

## Changes records or flags?

<!-- If this change alters a stored record (baseline.json, training.json, validation.json, report.json), a flag's meaning or a default, describe the impact and how existing runs are read. Otherwise, write "No". -->

## Summary (human written only)

<!-- What changes and why, in up to one paragraph. Point to the spec for the design discussion rather than repeating it. -->


## Verification

<!-- What was run and what it showed: the test count, and for a change that touches sampling, training or grading, a real-model run and whether its records matched. -->

## Agent details (if applicable)

<!-- Which model or agent assisted and what it wrote, and anything from its run worth keeping: its plan, its review findings, the rulings it made. -->
