# Iteration 1 Feedback

## Setup

- **Skill version**: initial draft (this iteration's input)
- **Test case**: eval-1 (single-experiment-rich) on `consistency_curve_first_digits_k10`
- **Validator**: independent general-purpose subagent, told to follow only the skill files in `.claude/skills/create-quiz/`
- **Result**: 8 questions, all 6 quantitative assertions passed, but 5 SKILL.md defects identified

## Defects found and fixes applied

### 1. External-reference policy contradicted reality

The skill's "Validation before completion" section said the only external script reference should be MathJax. The Jinja template injects both MathJax *and* polyfill.io (a hard dependency MathJax 3 declares). Every render would have failed this check as written.

**Fix:** updated `SKILL.md` validation step to allow both MathJax and polyfill.io. The unit test (`test_no_external_refs_except_mathjax`) was already correct — it whitelists both.

### 2. Worked example in `question_catalog.md` was self-contradictory

The multiple_choice example showed "Answer: D ... wait, that's already above 0.95 ... Actually the answer is C." It read like a debugging stream-of-consciousness, not a teaching example.

**Fix:** rewrote the example to give a clean answer + a separate paragraph explaining the wrong-answer trap.

### 3. Precision vs. tolerance conflated

`SKILL.md` §2 said "answer keys must reflect that precision (e.g. `0.987` ... not rounded)" right next to the question-catalog rule that `numerical_estimate` requires a tolerance. A first-time reader could plausibly conclude all numerical questions must use tolerance=0.

**Fix:** rewrote §2 to explicitly separate two principles: (a) source numbers must be transcribed verbatim (lossy transcription corrupts the answer key); (b) per-question `tolerance` is a separate, deliberate parameter that controls auto-grader strictness.

### 4. Image-read recommendation didn't flag the no-PNGs case

`SKILL.md` §3 says "aim for at least 1 image-read" without mentioning that the rule is conditional. The conditional ("only if PNGs exist; never synthesize") was buried in `parsing.md` and `question_catalog.md`. The validator's experiment had only `*.html` interactive plots, no PNGs — and would have synthesized one if reading SKILL.md alone.

**Fix:** moved the conditional ("only if `analysis/*.png` exists; if only HTML plots, drop image_read entirely") up into `SKILL.md` §3.

### 5. Ranking-on-degenerate-metric not anticipated

`question_catalog.md` recommends ranking when ≥3 conditions exist on a single metric, but doesn't mention what to do when all conditions tie (here, all 6 fine-tuned runs hit accuracy 1.000). The validator made the right judgment call (rank by training loss instead) but the skill should anticipate this.

**Fix:** added a clarification to the §3 ranking bullet in `SKILL.md`: only generate ranking when conditions actually differ on the chosen metric; otherwise pick a different metric or a different question type.

## Validator-flagged judgment calls (no fix needed; left as guidance)

- **Tolerance for intuition questions:** validator set 0.04 (loose) on a "predict the base accuracy from the chance-baseline hypothesis" question, intentionally letting "~0.10" answers pass. The catalog said "loose tolerance is wrong" but for this question type the looser bar is right pedagogy. The fixed §2 (precision vs. tolerance separation) covers this.
- **Workspace redirect:** validator was unsure whether `experiments[].directory` should reflect the real experiment path or the redirected output path. Clarified in `generation.md` — citation, not output location.

## Outcome

All 5 defects fixed in-iteration. Iteration 1 closes with the renderer test still green (10/10) and the smoke-test artifact (length_sweep_nth14/quiz/quiz.html) regenerable from the corrected skill.

The user has not personally reviewed iteration 1's outputs yet — they should open both quiz.html files (length_sweep_nth14 and consistency_curve_first_digits_k10) in a browser before iteration 2 to give substantive feedback that an automated assertion can't catch (question quality, prompt clarity, intuition-test difficulty).
