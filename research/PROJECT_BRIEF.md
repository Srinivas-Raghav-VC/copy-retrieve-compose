# Project Brief

## Working title
Mechanistic Interpretability of In-Context Transliteration Rescue and Degradation in Gemma-family Models

## Mission
Build a reviewer-defensible, architecture-aware, behavior-grounded account of how helpful in-context examples rescue transliteration, when that rescue degrades, and what internal computation implements both the rescue and the failure modes.

## Broad stance
This is a research project, not just a mechanistic interpretability project in a narrow sense.
Stay broad enough to notice when:
- the evaluation stack is the real bottleneck,
- the architecture constraints drive the phenomenon,
- the prompting regime matters more than expected,
- or the initial story is simply wrong.

## Main questions

### Behavioral
- When does zero-shot fail?
- When does helpful ICL rescue?
- Is the rescue language-dependent or script-dependent?
- Does degradation appear at higher example counts or higher prompt density?
- Is degradation a real behavioral phenomenon or partly a metric artifact?

### Evaluation
- What counts as usable transliteration quality?
- Where is exact match too strict?
- Where do acceptable variants matter?
- Where does first-token proxy track real quality?
- Where does it fail?

### Mechanistic
- What internal change distinguishes zero-shot failure from successful ICL?
- What changes again in degraded ICL?
- What roles are played by global layers, local layers, heads, MLPs, and residual directions?
- Is the mechanism compact, distributed, or both?

### Publication
- What is the one-sentence paper thesis?
- What would make the paper weak, decent, strong, or standout?
- What would a skeptical reviewer attack first?
- What result would make the project not worth writing up?

## Scope discipline
Do not let the project become:
- a bag of benchmarks,
- a bag of patching plots,
- a bag of proxy claims,
- or a bag of disconnected multilingual analyses.

Always ask:
What is the main paper idea?
What is the minimum set of experiments needed to defend it?
