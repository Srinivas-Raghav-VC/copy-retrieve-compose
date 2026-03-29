# Mechanistic technique playbook and fast-direction memo

_Date: 2026-03-29_

## Short answer to the question

Yes — I did go through the main web sources you asked me to use, including the key ARENA materials, Anthropic interpretability pages, Ekdeep Lubana's site, and Alignment Forum / LessWrong context.

But until now I had **not** written a single dedicated Markdown file whose sole job was:

- what technique each source suggests,
- how to use it in *this* thesis,
- what to trust,
- what not to overclaim,
- and what the fastest defensible direction is.

This file is that memo.

---

## What I actually read and what it contributes

## 1. ARENA materials

### Sources
- ARENA Chapter 1: Transformer Interpretability
- ARENA Chapter 3: LLM Evals
- ARENA Chapter 4: Alignment Science
- ARENA 3.0 repo / curriculum context
- LessWrong ARENA announcement / framing

### What to take from it

ARENA does **not** mainly give us one magic mechanism trick. It gives us the right research discipline:

- define the question clearly,
- define the failure modes clearly,
- use strong evals and controls,
- audit outputs manually,
- separate black-box evidence from white-box evidence,
- do not overclaim from a weak metric.

### How to use it here

Use ARENA as the project spine:

`behavioral anchor`
`-> controls`
`-> manual audit`
`-> localization`
`-> causal test`
`-> bounded claim`

That is already the correct workflow for this thesis.

### What not to take from it

Do **not** treat ARENA as meaning “do many exercises / many methods = good paper.”

The lesson is rigor, not method collection.

---

## 2. Anthropic interpretability / tracing / AI biology work

### Sources
- Tracing the Thoughts of a Large Language Model
- On the Biology of a Large Language Model
- Attribution graph methods / circuit tracing pages

### What to take from it

Anthropic’s strongest lesson is:

- start from a clean behavior,
- use detailed case studies,
- localize mechanisms,
- validate with interventions,
- and be explicit that these are often **partial** explanations, not full universal maps.

### How to use it here

This supports exactly the case-study structure we now want:

`1B Hindi`
`-> early-routing failure case study`

`1B Telugu`
`-> later retrieval/composition failure case study`

`4B Telugu`
`-> positive-control case study`

### What not to overclaim

Do **not** say:
- “we found the transliteration circuit”
- “the attribution graph proves the full mechanism”
- “the feature graph is the truth”

The right claim style is:
- candidate mechanism,
- bounded causal role,
- positive and negative evidence both reported.

---

## 3. Ekdeep Lubana / ICL phase / abstraction-caution line

### Sources
- Ekdeep’s website
- Competition Dynamics Shape Algorithmic Phases of In-Context Learning (arXiv:2412.01003)
- In-Context Learning Strategies Emerge Rationally (arXiv:2506.17859)
- Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry (arXiv:2503.01822)
- Analyzing (In)Abilities of SAEs via Formal Languages (arXiv:2410.11767)

### What to take from it

Two crucial ideas:

1. **ICL can move between different algorithmic regimes**, not one mechanism.
2. **SAEs / sparse feature methods impose assumptions** and should not be treated as automatically faithful.

### How to use it here

This is probably the single best conceptual frame for the thesis:

`copying`
`-> prompt-bank retrieval`
`-> nearest-neighbor retrieval`
`-> composition`

That is much stronger than a plain “rescue/degradation” framing.

It also tells us how to use feature methods:

`feature tools = instruments`
`not`
`feature tools = proof`

### What not to do

Do **not** build the whole paper around SAE or transcoder findings unless the behavioral and causal evidence already stands on its own.

---

## 4. Alignment Forum / LessWrong cautionary material

### Sources
- Towards Data-Centric Interpretability with Sparse ...
- Takeaways from recent work on SAE probing
- related LessWrong / Alignment Forum discussion

### What to take from it

Main lesson:

- explanation pipelines can look good while tracking artifacts,
- automatic interpretability metrics can be misleading,
- null baselines and manual auditing matter.

### How to use it here

For this thesis:

`manual audit` is not optional.

`judge outputs` are not hard truth.

`feature results` are not causal results.

This is why we now require:
- manual audit packets,
- hard/soft metric separation,
- reviewer-style skepticism.

---

## 5. Transliteration + ICL paper

### Source
- Exploring the Role of Transliteration in In-Context Learning for Low-resource Languages Written in Non-Latin Scripts (arXiv:2407.02320)

### What to take from it

Transliteration really is a useful ICL setting, especially for non-Latin scripts.

### How to use it here

This paper helps justify transliteration as a serious research domain, but our thesis should go beyond performance:

`transliteration task`
`-> clean laboratory for ICL algorithms`

not just

`transliteration task`
`-> another multilingual benchmark`

---

## The actual technique stack for *this* thesis

Below is the prioritized playbook.

## Priority 1 — use immediately

### A. Behavioral decomposition

#### What it tests
Whether failure happens:
- at the first target token,
- or later during whole-word continuation.

#### Why it matters
This already split the story cleanly:
- `1B Hindi` = early-routing failure,
- `1B Telugu` = mostly later retrieval/composition failure.

#### How to use it
- compare `first_prob`, `first_entry_correct`, `exact_match`, `CER`
- keep condition contrasts: `zs`, `helpful`, `corrupt`, similarity-sorted variants

#### Trust level
High, because it is directly tied to behavior.

---

### B. Target-vs-competitor tracing

#### What it tests
At the critical decoding step, which wrong competitor beats the correct token?

#### Why it matters
This is the strongest next probe for `1B Hindi`.

#### How to use it here
For `1B Hindi`:
- trace correct target token vs best wrong competitor,
- record script/type of competitor,
- localize by layer and position,
- compare `zs`, `helpful`, `corrupt`.

#### Acceptance criterion
A useful result would show:
- where the correct token loses,
- to what kind of competitor,
- and whether intervention in the localized band changes that competition selectively.

---

### C. Continuation-localized gold-vs-bank analysis

#### What it tests
After the first token, when does the model drift toward a copied bank continuation instead of the gold continuation?

#### Why it matters
This is the central next probe for `1B Telugu`.

#### How to use it here
For copied-bank cases:
- compare gold continuation tokens vs copied-bank continuation tokens,
- trace position-by-position divergence,
- use `helpful`, `corrupt`, `similarity_desc`, `similarity_asc`, `reversed`.

#### Acceptance criterion
A good result localizes where retrieval-like continuation overtakes query-specific continuation.

---

### D. Activation patching / bounded ablation

#### What it tests
Whether the localized band or component has a causal effect on the observed failure mode.

#### Why it matters
This is what turns “interesting pattern” into “bounded mechanistic evidence.”

#### How to use it here
- only after localization,
- patch or ablate a narrow layer/position/component band,
- test predicted effect on the exact failure mode.

#### Acceptance criterion
The intervention should shift the targeted competition or continuation pattern in the predicted direction without just globally destroying performance.

---

## Priority 2 — use as supporting evidence

### E. Logit-lens trajectories

#### Use
Good for showing where the target token becomes more or less linearly available.

#### Role here
Supportive localization tool, especially for:
- `1B Hindi` first-token path,
- `1B Telugu` continuation path,
- `4B Telugu` positive comparison.

#### Limitation
Correlational by itself.

---

### F. Script-space maps

#### Use
Useful coarse localization of script emergence and target-script mass.

#### Role here
Good screening tool; already helped identify candidate layer bands.

#### Limitation
Not enough for a mechanistic claim on its own.

---

### G. Feature methods: transcoders / SAEs / CLT-like tools

#### Use
Hypothesis generation and maybe selective intervention targets.

#### Role here
Secondary or appendix-level unless causal evidence becomes strong.

#### Limitation
Should not be treated as the main proof of mechanism.

---

## Priority 3 — defer unless needed

### H. Full attribution graphs / broad circuit maps

#### Why defer
Too easy to overclaim, too expensive relative to current uncertainty, and not necessary for a good honors thesis.

#### Use later only if
The bounded case studies become very clean and the broader graph view would materially sharpen the thesis.

---

## Fast direction if we need to wrap this up quickly

If the main constraint is time and we need the direction to become very clear very fast, then the thesis should collapse to this:

## Final fast-path thesis

`4-language multi-seed behavioral map`
`-> establish the regime story`

`manual audit packets`
`-> show the regime story is real in raw outputs`

`1B Hindi mechanistic case study`
`-> early-routing failure`

`1B Telugu vs 4B Telugu mechanistic comparison`
`-> retrieval-heavy later failure vs more compositional continuation`

`final thesis claim`
`-> scale changes the algorithmic regime of ICL in multilingual transliteration`

That is the fast, clear, defensible core.

---

## What to de-prioritize if wrapping quickly

If we need speed, these become secondary or future-work material:

- full verifier-stack ambition beyond a minimal calibrated subset,
- large extra language expansions beyond the confirmatory four,
- broad feature-circuit mapping,
- many exotic intervention families,
- trying to prove a universal multilingual circuit.

In other words:

`do not finish everything`

Finish the parts that make the thesis clearly acceptable.

---

## The clearest possible project direction

Here is the arrow version.

`romanized source + in-context examples`
`-> model enters an ICL regime`
`-> regime is copying / retrieval / composition`
`-> regime depends on scale`
`-> 1B and 4B differ behaviorally`
`-> differences can be decomposed into early-routing vs later-continuation failures`
`-> those differences can be localized mechanistically`
`-> bounded causal tests support the mechanism story`
`-> thesis: scale changes how ICL is internally implemented in multilingual transliteration`

---

## The three exact mechanistic targets

## 1. `1B Hindi`

`high-shot helpful prompt`
`-> early routing degrades`
`-> target token loses to source-like / Latin-like / wrong competitors`

### Best next method
- target-vs-competitor layer/position trace
- then narrow patching/ablation

### Why this is good
It is the cleanest mech-ready failure case.

---

## 2. `1B Telugu`

`high-shot helpful prompt`
`-> first token often fixed`
`-> later continuation often becomes wrong prompt-bank target`

### Best next method
- continuation-localized gold-vs-bank trace
- then localized intervention on the late retrieval/composition band

### Why this is good
It is the most interesting retrieval-vs-composition case.

---

## 3. `4B Telugu`

`same task family`
`-> positive control`
`-> escapes the strong bank-copy regime much more often`

### Best next method
- run the same later-continuation probes as on `1B Telugu`
- compare where and how the computation stays aligned with the gold continuation

### Why this is good
It gives the thesis a mechanistic success case, not only failure cases.

---

## What would make the direction maximally clear

The direction becomes extremely clear if we commit to this exact sentence:

> **This thesis studies when ICL in Gemma 3 behaves like copying, retrieval, or composition in multilingual transliteration, using a four-language behavioral map and two bounded mechanistic case studies (`1B Hindi`, `1B Telugu`) plus one positive control (`4B Telugu`).**

That is already clear enough to drive the rest of the work.

---

## What I recommend now

If the goal is to wrap this up quickly and clearly, the next priorities should be:

1. finish and aggregate the 4-language multi-seed panel,
2. emit manual audit packets for every seed/cell,
3. run the `1B Hindi` mechanistic localizer,
4. run the `1B Telugu` vs `4B Telugu` continuation localizer,
5. freeze the thesis around those results.

That is the shortest path to a thesis that still has real mechanistic backing.

## Sources
- https://learn.arena.education/chapter1_transformer_interp/
- https://learn.arena.education/chapter3_llm_evals/
- https://learn.arena.education/chapter4_alignment_science/
- https://github.com/callummcdougall/ARENA_3.0?tab=readme-ov-file
- https://www.lesswrong.com/posts/nQAN2vxv2ASjowMda/new-arena-material-8-exercise-sets-on-alignment-science-and
- https://www.anthropic.com/research/tracing-thoughts-language-model
- https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- https://ekdeepslubana.github.io/
- https://www.alignmentforum.org/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse
- https://www.alignmentforum.org/posts/osNKnwiJWHxDYvQTD/takeaways-from-our-recent-work-on-sae-probing
- https://arxiv.org/abs/2407.02320
- https://arxiv.org/abs/2412.01003
- https://arxiv.org/abs/2506.17859
- https://arxiv.org/abs/2406.11944
- https://arxiv.org/abs/2501.17727
- https://arxiv.org/abs/2503.01822
- https://arxiv.org/abs/2410.11767
- file:///mnt/d/Research/Honors/outputs/thesis_strategy_grander_goal_2026-03-29.md
- file:///mnt/d/Research/Honors/outputs/loop2_failure_modes_2026-03-29.md
- file:///mnt/d/Research/Honors/research/spec.md
