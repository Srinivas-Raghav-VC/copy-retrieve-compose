# Thesis strategy memo: from a narrow transliteration benchmark to a stronger honors thesis

_Date: 2026-03-29_

## Executive recommendation

Yes, you can and probably should use the free VRAM for **better verification infrastructure**. But the right move is **not** to replace deterministic checks with a local judge as the new source of truth.

The stronger move is:

1. keep deterministic checks as the **hard anchor**,
2. add a **calibrated local judge stack** for richer labels and failure taxonomy,
3. reframe the thesis around a **bigger scientific question** than “does transliteration get better or worse?”, and
4. use transliteration as a **model organism for in-context learning algorithms** — especially the boundary between **copying, retrieval, and composition** across scale.

If the goal is a thesis or paper that feels more substantial than a modest benchmark note, the best honest version is something like:

> **How does in-context learning shift a model between copying, nearest-neighbor retrieval, and genuine composition in multilingual transliteration, and how do those regimes change with scale?**

That is broader, more mechanistic, and more interesting than a narrow performance story, while still staying grounded in what the current evidence actually supports.

---

## First-principles framing

### What is the object we actually care about?

Not “transliteration accuracy” by itself.

The more fundamental question is:

> When a language model sees many in-context examples of a string transformation task, **what algorithmic mode** does it enter?

Possible modes include:

- **source copying**
- **wrong-script fallback**
- **nearest-neighbor retrieval from the prompt bank**
- **partial composition**
- **full query-specific composition**

That is a more general scientific object than transliteration alone.

### Why transliteration is still a good domain

Transliteration is unusually good as a research microscope because it is:

- cheap to run,
- multilingual,
- tightly string-grounded,
- easy to audit with deterministic metrics,
- easy to classify failure modes manually,
- rich enough to separate **early token routing** from **later whole-word continuation**.

This means transliteration is not necessarily the final “story”; it can be the **clean experimental system** for studying ICL algorithms.

### What would make the thesis feel bigger?

A better thesis is not one with more models, more plots, or more benchmarks.

A better thesis is one that makes a stronger claim of the form:

- here is a **clean phenomenon**,
- here is a **phase structure** or decomposition of that phenomenon,
- here is a **validated taxonomy of failure modes**,
- here is at least one **bounded mechanistic explanation** with strong evidence.

That is the level that feels closer to serious interpretability work rather than “we ran some multilingual evals.”

---

## Multi-persona synthesis

### Persona 1: honors thesis advisor

This persona asks: what would make a committee think the work is intellectually serious?

Best answer:

- A thesis-quality question is **not** “is Gemma 3 1B bad at transliteration?”
- It **is** something like: “What algorithmic regimes does ICL induce in multilingual string transformation, and how does scale change those regimes?”
- The thesis should produce:
  - a clean behavioral phase map,
  - a validated failure taxonomy,
  - and one or two bounded mechanistic case studies.
- The thesis should not promise a full circuit atlas.
- It should promise a **careful narrowing from behavior to candidate mechanism**.

### Persona 2: mechanistic-interpretability skeptic

This persona asks: what would be irresponsible to claim?

Irresponsible now:

- claiming a single unified `1B` mechanism,
- claiming a full circuit,
- claiming that SAE/transcoder features are the truth rather than a lens,
- claiming a judge agrees therefore the phenomenon is real,
- claiming a broad multilingual law from only a few anchor languages.

Defensible now:

- `1B Hindi` shows a **substantial early routing failure** at high shot,
- `1B Telugu` shows a **mostly later retrieval/composition failure**,
- `4B Telugu` is a **clean positive anchor**,
- visibility matters but is **not the whole story**,
- different languages appear to land in **different algorithmic regimes**.

### Persona 3: evaluation engineer with spare VRAM

This persona asks: what should we actually do with local compute?

Best answer:

Use VRAM for a **verification stack**, not a vanity judge.

Specifically:

1. **Local transliteration correctness judge**
   - Given source romanized word, target-script candidate, and gold answer, classify:
     - exact correct
     - acceptable variant / phonetic equivalent
     - wrong but close
     - source copy
     - prompt-bank copy
     - wrong script

2. **Local error-taxonomy judge**
   - A second prompt specialized for labeling failure mode, not correctness.

3. **Manual calibration set**
   - Human-label a few hundred examples.
   - Report agreement and disagreements.

4. **Null-model checks**
   - Evaluate judge behavior on:
     - exact source copies,
     - nearest-neighbor wrong bank copies,
     - random wrong-script outputs,
     - small edit-distance but wrong outputs.

If the judge is not robust on those nulls, it should not be central in the paper.

### Persona 4: audience strategist

This persona asks: what would feel more satisfying to a broader technical audience while still being honest?

The answer is probably **not** “Gemma 3 1B has transliteration issues.”

A stronger public-facing frame would be closer to:

> Small models often use in-context examples by **copying or retrieving similar prompt answers**, while larger models more often convert those examples into **query-specific composition**.

That is a much more legible and interesting claim.

Even better if supported by:

- a clean phase diagram across scale and shot,
- one striking manual example set,
- one clean mechanistic case study showing **where** the model diverges.

---

## What the recent sources suggest

### Anthropic / Transformer Circuits line

The 2025 Anthropic work on **circuit tracing** and **AI biology** argues for a bottom-up “microscope” approach: use case studies to uncover mechanisms you would not have guessed ahead of time, validate them with interventions, and be explicit that these are usually **existence proofs** rather than universal laws.

Important lessons for this thesis:

- case studies are acceptable and valuable,
- interventions matter more than pretty visualizations,
- attention blind spots remain real,
- one should not pretend the current tool explains everything,
- “surprising structure” discovered bottom-up is a strength, not a weakness.

This matches the current project state well: we already have a bottom-up discovery that `1B Hindi` and `1B Telugu` fail differently.

### Ekdeep Lubana / algorithmic phases / faithful abstractions line

Ekdeep’s recent work points in two especially useful directions.

First, the ICL phase-diagram work suggests that ICL behavior should often be thought of as a **competition among multiple algorithmic strategies**, not as a single mechanism. That is almost exactly the right language for the current project.

Second, his work on SAEs and concept geometry warns that sparse dictionary methods impose assumptions and are **not universally faithful abstractions**. That argues strongly against overselling any SAE/transcoder result as the mechanism itself.

The right takeaway is:

- use SAEs / transcoders as **instruments**,
- validate them behaviorally and causally,
- and frame them as one lens among several.

### ARENA / evals / alignment-science line

The ARENA materials emphasize:

- tested scaffolds,
- threat-model-first eval design,
- skepticism about reward hacking,
- and using black-box and white-box methods together.

This strongly supports a thesis structure where:

- deterministic evaluation defines the hard spec,
- local judges add soft labels,
- and mechanistic analysis is downstream of clear behavioral anchors.

### AI Alignment Forum / SAE probing caution line

Recent discussion around SAE probing and data-centric interpretability reinforces a recurring caution:

- if the metric or explanation pipeline is not stress-tested against nulls, it can look meaningful while tracking artifacts.

Together with the paper **Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers** (2025, arXiv:2501.17727), this is a serious warning against treating automatic metrics or judge scores as decisive without null-model validation.

---

## Recommended thesis-level question

### Primary recommended question

> **How does in-context learning move Gemma 3 models between copying, retrieval, and composition in multilingual transliteration, and how do those regimes change with scale?**

Why this is strong:

- it subsumes the current behavioral findings,
- it makes transliteration scientifically useful rather than narrow,
- it naturally motivates both evaluation and mechanistic work,
- it leaves room for scale effects without assuming one “special” model,
- and it is honest about partial rather than total understanding.

### Strong candidate title ideas

- **From Copying to Composition: Scale-Dependent In-Context Transliteration in Gemma 3**
- **When In-Context Learning Becomes Retrieval: A Cross-Scale Study of Multilingual Transliteration**
- **Algorithmic Phases of In-Context Transliteration: Copying, Retrieval, and Composition Across Scale**

---

## Recommended contribution stack

A stronger paper or thesis would ideally have **three layers of contribution**.

### Layer 1: evaluation contribution

Build a stronger benchmark and verifier stack than plain exact-match alone.

Deliverables:

- deterministic metrics:
  - exact match,
  - script validity,
  - CER / akshara-aware CER,
  - source-copy and prompt-bank-copy detectors,
  - nearest-neighbor rank features.
- local judge metrics:
  - transliteration acceptability / variant equivalence,
  - failure-mode labeling.
- human audit subset:
  - agreement table,
  - disagreement categories.

This prevents the work from feeling like “exact-match or bust.”

### Layer 2: algorithmic phase contribution

This is probably the core paper contribution.

Possible phase map:

- **Phase A: source-copy / wrong-script fallback**
- **Phase B: bank-copy retrieval**
- **Phase C: partial composition**
- **Phase D: robust query-specific composition**

And maybe an orthogonal decomposition:

- **early routing success/failure**
- **late continuation success/failure**

This gives the thesis a conceptual backbone.

### Layer 3: mechanistic case-study contribution

Not a universal map, but a bounded pair of case studies:

- **1B Hindi**: early routing failure
- **1B Telugu**: later retrieval/composition failure
- **4B Telugu**: positive comparison case

This is enough for an honors thesis if the causal evidence is careful.

---

## Should you use a local judge?

### Yes — but in a disciplined role

A local judge is worth using if it is treated as a **secondary instrument**.

### Good roles for a local judge

- recognizing acceptable transliteration variants,
- assigning error taxonomy labels,
- detecting whether two outputs are phonetic near-equivalents,
- helping create a manually audited benchmark more quickly,
- triaging examples for human review.

### Bad roles for a local judge

- replacing deterministic metrics as the primary claim,
- replacing human audit entirely,
- deciding mechanistic truth,
- being used without null checks or calibration.

### Best design

Use a **two-tier evaluation protocol**:

1. **hard truth layer**
   - deterministic metrics and manually auditable functions,
2. **soft interpretation layer**
   - local judge for acceptability and error labels.

Report them separately.

### If you have the VRAM, add this

A very good use of local compute would be a **small ensemble verifier**:

- judge A: correctness / equivalence,
- judge B: failure taxonomy,
- judge C: optional pairwise comparison between candidate and gold.

Then measure disagreement across judges and against humans.

That becomes a substantive methods contribution rather than just “we used an LLM judge.”

---

## What would make the project feel “grander” while staying honest?

Here are the strongest honest upgrades.

### Option 1: transliteration as a model organism for ICL

Instead of selling transliteration itself as the final importance, sell it as a **clean system for understanding ICL algorithms**.

This is my favorite option.

### Option 2: a failure-atlas paper

Build a paper around a **fine-grained atlas of failure modes**:

- source copies,
- bank copies,
- script drift,
- first-token routing failures,
- near-miss continuations,
- scale transitions.

This is useful and easy to explain.

### Option 3: a benchmark + mechanism package

A paper that combines:

- a strong benchmark,
- a calibrated verifier stack,
- a phase diagram,
- and one mechanistic case study.

This is probably the best mix of ambition and feasibility.

### Option 4: an interactive “ICL microscope” demo

For public-facing interest, an interactive artifact could matter a lot.

Imagine a demo where a user inputs:

- a source word,
- a set of in-context examples,
- a model size,

and the tool shows:

- predicted output,
- whether it is source copy / bank copy / near miss / correct,
- nearest copied prompt example,
- first-token confidence gap,
- layer-range indicators from the mechanistic screens.

This would make the research much more legible to non-specialists.

---

## A stronger thesis/paper outline

### Thesis claim status ladder

#### Established

- There are clean cross-scale differences in multilingual transliteration behavior.
- `4B Telugu` is a robust positive anchor.
- `1B Hindi` and `1B Telugu` fail differently.
- Visibility alone does not explain the `1B` failures.

#### Supported but provisional

- `1B Hindi` shows substantial early routing failure.
- `1B Telugu` shows mostly later retrieval/composition failure.
- `1B Telugu` bank-copy errors often come from the nearest-neighbor region of the prompt bank.
- High-shot `1B Telugu` mixes query-conditioned nearest-neighbor retrieval with a more generic long-prompt bank-copy tendency.

#### Not yet established

- the exact circuits implementing those failures,
- whether the same decomposition generalizes across many more languages and seeds,
- whether SAE/transcoder features identify the true causal basis rather than a correlated proxy.

### Suggested chapter / paper structure

1. **Why transliteration is a clean microscope for ICL**
2. **Benchmark and verifier design**
3. **Cross-scale behavioral phase map**
4. **Failure taxonomy: source copy, bank copy, routing failure, near miss**
5. **Mechanistic case study A: 1B Hindi**
6. **Mechanistic case study B: 1B Telugu vs 4B Telugu**
7. **What this says about copying vs composition in ICL**
8. **Limitations and future work**

---

## Concrete next research program

### Phase 1: evaluation hardening

Do this before trying to make the story feel bigger.

- build the calibrated local judge stack,
- create a manual gold subset,
- measure judge agreement,
- add explicit bank-copy / source-copy detectors,
- create a small public error taxonomy table.

### Phase 2: paper-grade phase diagram

Run a focused grid that supports a real “copying → composition” story.

- models: `270M`, `1B`, `4B` (optionally larger if practical)
- languages: Hindi, Telugu, plus 2–3 breadth languages
- shot sizes: enough to show transitions, not everything
- outputs:
  - exact match,
  - CER,
  - first-token correctness,
  - bank-copy rate,
  - source-copy rate,
  - judge-labeled acceptability.

### Phase 3: bounded mechanistic work

- **Hindi path:** first-token routing localizer
- **Telugu path:** late continuation / retrieval localizer
- **4B Telugu:** positive control

Only then do targeted causal interventions.

### Phase 4: public-facing artifact

- interactive failure explorer,
- or polished visual memo / website,
- or reproducible benchmark package.

---

## What not to do

- Do not swap out deterministic evaluation for a judge and call it rigor.
- Do not claim a single `1B` mechanism.
- Do not claim SAE/transcoder findings are definitive without nulls and perturbations.
- Do not chase “grandness” by adding many tasks with weak depth.
- Do not optimize for Hacker News applause at the cost of a cleaner thesis.

---

## Final recommendation

If you want the strongest honest thesis, I would choose this north star:

> **Use multilingual transliteration as a model organism to study when ICL behaves like copying, retrieval, or composition, and how scale changes those regimes.**

Then structure the work around:

- a stronger verifier stack,
- a cross-scale phase diagram,
- and one or two bounded mechanistic case studies.

That is bigger than “a mediocre transliteration benchmark,” but still realistic, auditable, and thesis-worthy.

## Sources

### Current project artifacts
- file:///mnt/d/Research/Honors/outputs/loop2_failure_modes_2026-03-29.md
- file:///mnt/d/Research/Honors/outputs/loop2_telugu_retrieval_conditions_2026-03-29.json
- file:///mnt/d/Research/Honors/outputs/loop2_bank_copy_rank_2026-03-29.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/first_token_competition_v1/results/summary.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/token_visibility_v1/results/token_visibility_summary.csv
- file:///mnt/d/Research/Honors/research/results/autoresearch/mech_screen_v1/results/script_space_summary_1b_hindi.csv
- file:///mnt/d/Research/Honors/research/results/autoresearch/mech_screen_v1/results/script_space_summary_4b_telugu.csv
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/seed_aggregate.json

### Web / docs / courses
- https://www.lesswrong.com/posts/nQAN2vxv2ASjowMda/new-arena-material-8-exercise-sets-on-alignment-science-and
- https://learn.arena.education/chapter1_transformer_interp/
- https://learn.arena.education/chapter3_llm_evals/
- https://learn.arena.education/chapter4_alignment_science/
- https://github.com/callummcdougall/ARENA_3.0?tab=readme-ov-file
- https://www.anthropic.com/research/tracing-thoughts-language-model
- https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- https://ekdeepslubana.github.io/
- https://www.alignmentforum.org/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse
- https://www.alignmentforum.org/posts/osNKnwiJWHxDYvQTD/takeaways-from-our-recent-work-on-sae-probing

### Papers / alpha-backed references
- Exploring the Role of Transliteration in In-Context Learning for Low-resource Languages Written in Non-Latin Scripts (2024), arXiv:2407.02320, https://arxiv.org/abs/2407.02320
- In-Context Learning Creates Task Vectors (2023), arXiv:2310.15916, https://arxiv.org/abs/2310.15916
- Competition Dynamics Shape Algorithmic Phases of In-Context Learning (2024), arXiv:2412.01003, https://arxiv.org/abs/2412.01003
- In-Context Learning Strategies Emerge Rationally (2025), arXiv:2506.17859, https://arxiv.org/abs/2506.17859
- Transcoders Find Interpretable LLM Feature Circuits (2024), arXiv:2406.11944, https://arxiv.org/abs/2406.11944
- A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models (2024), arXiv:2407.02646, https://arxiv.org/abs/2407.02646
- Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers (2025), arXiv:2501.17727, https://arxiv.org/abs/2501.17727
- Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry (2025), arXiv:2503.01822, https://arxiv.org/abs/2503.01822
- Analyzing (In)Abilities of SAEs via Formal Languages (2024/2025), arXiv:2410.11767, https://arxiv.org/abs/2410.11767
