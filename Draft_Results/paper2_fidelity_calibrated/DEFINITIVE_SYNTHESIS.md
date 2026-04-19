# Definitive Synthesis: ICL Transliteration Rescue in Gemma 3
## Connecting Our Findings to Ekdeep's Framework, Carlini's Standards, and the 2022–2026 Literature

---

## I. Our Work Through Ekdeep's Three-Level Lens

Ekdeep's AP293 talk argues that interpretability needs all three of Marr's levels — computational, algorithmic, implementational — and that the gap between current tools (causal abstraction at the implementational level, SAEs at the implementational level) is the ALGORITHMIC level. His "model systems" approach fills this gap by using simplified systems to derive algorithmic hypotheses that then guide implementational analysis.

Our work accidentally does something similar, but in reverse: we started at the implementational level (patching, head attribution), discovered unexpected results (attention alone fails, coupled circuits), and now need to connect upward to the algorithmic level to explain WHY.

### Computational Level: What is the model computing?

The model solves a conditional generation problem:
- **Input**: Instruction + optional ICL examples + Latin-script query word
- **Output**: The same word in the target script (Devanagari, Telugu, Bengali, etc.)

Zero-shot, the model CANNOT do this — it defaults to Latin copies or English explanations. With 4-16 ICL examples, it CAN. The computational question is: **what function does ICL enable that the model cannot compute alone?**

The answer has two parts (our FORMAT vs CONTENT decomposition):
1. **FORMAT**: ICL activates the correct output mode ("produce text in this script"). This is a task-selection function — choosing among the model's pretrained capabilities.
2. **CONTENT**: ICL provides specific character mappings. This is a rule-extraction function — deriving the input→output mapping from examples.

This maps directly onto Ekdeep's Memorizing (M) vs Generalizing (G) dichotomy:
- **FORMAT-only ICL ≈ Memorizing**: The model recognizes the task from surface features and retrieves a memorized behavior (output in script X). No generalization from example content needed.
- **CONTENT-specific ICL ≈ Generalizing**: The model extracts the mapping rule from examples and applies it to the new input. This requires posterior inference over the task.

### Algorithmic Level: How does the model compute it?

Ekdeep's key equation:

```
h_pred(s_i | context) = σ(η) · M(s_i | context) + (1 - σ(η)) · G(s_i | context)
```

where η(N, D) = γN^{1-α} ΔL(D) - ΔK(D)/β determines the weighting between M and G.

**Our content-specificity hierarchy IS the η gradient across languages:**

| Language | h/c ratio (L17) | η interpretation | Strategy |
|----------|----------------|-------------------|----------|
| Hindi    | 1.8×           | High η → strong G weight | Generalizing from ICL content |
| Telugu   | 2.2×           | High η at mid-layers, converges at output | G→M collapse at readout |
| Bengali  | 1.2×           | η ≈ 0 → boundary | Mixed M/G |
| Kannada  | 0.7×           | Negative η → M dominates | Memorizing (format only) |
| Gujarati | 1.0×           | η ≈ 0 → boundary | Format-triggered retrieval |

The crucial insight: **η is not just a function of training (N) and task diversity (D) as in Ekdeep's model systems. In a production model on a real task, η also depends on the MODEL'S PRETRAINED KNOWLEDGE of the specific language.** Hindi has high η because the model has seen enough Hindi transliteration data during pretraining that the ICL examples can activate and refine its existing knowledge (generalizing). Kannada has low/negative η because the model lacks this foundation, so ICL can only activate the coarser "output in this script" mode (memorizing).

This is a genuine extension of Ekdeep's framework: **the M/G transition is modulated by pretrained knowledge, not just training steps and task diversity.**

**Ekdeep's belief dynamics model also explains our density curve.** His paper 4 (Bigelow, Wurgaft et al.) shows that ICL = updating a posterior belief. Each example shifts the belief vector: z_i → z_i + m. The critical context length is N*(m) = [(a·m + b)/γ]^{1/(1-α)}.

Our density data:
- 8 examples: p = 0.270 (rising)
- 16 examples: p = 0.234 (plateau)
- 32 examples: p = 0.171 (declining — attention dilution)
- 48 examples: p = 0.132 (declining)
- 64 examples: p = 0.004 (cliff — window loss)

This is NOT a monotonically increasing belief curve. The decline happens because of a RESOURCE CONSTRAINT that Ekdeep's model systems don't capture: **finite attention budget**. In a simplified model system, more examples always help (until saturation). In Gemma 3, more examples HURT because:
1. Global layers' attention is diluted across more tokens (gradual decline: 0.270 → 0.132)
2. Local layers lose access when examples exceed the sliding window (catastrophic cliff: 0.132 → 0.004)

**This is the key architectural insight our paper contributes: the "computational constraint" in Ekdeep's η equation has a specific implementational form — the finite attention budget of global layers and the hard cutoff of local layer windows.**

### Implementational Level: What mechanisms implement it?

This is where our experimental data lives:

**The circuit:**
1. L5 (global): Script detection, format recognition. 91-99% attention on ICL examples.
2. L6-L10 (local): Process L5's deposit. Cannot see ICL directly.
3. **L11 (global): THE KEY LAYER. L11 H0 is the universal rescue head across all 5 languages.** 92% attention on ICL. Writes character-mapping content to residual stream.
4. L12-L16 (local): Amplification zone. L14 H0 strongest single-head effect (but local — reads from residual stream, doesn't see ICL directly). L12 MLP is DESTRUCTIVE (-0.165) — it's adapted to ICL-modified residual streams and produces garbage on ZS streams.
5. **L17 (global): Integration layer.** 16%/84% split between local and ICL attention. First clear logit lens separation between helpful and corrupt ICL. Content-specific signal emerges here.
6. L18-L22 (local): Refinement. L21 H1 important for Telugu/Kannada/Gujarati.
7. L23 (global): Consolidation. ~60% on ICL.
8. L24-L25 (local): Readout. L25 MLP converts accumulated residual signal to output logits.

**The critical experimental finding:** Attention-only patching (replacing all 26 layers' attention outputs from ICL into ZS) gives PE ≈ 0.000026. Effectively ZERO. But single-head attribution shows L11 H0, L14 H0 with large effects. This means:

**ICL rescue is an EMERGENT property of the coupled attention-MLP interaction, not localizable to either component alone.** Replacing one head works because the MLP adapts to the mixed signal. Replacing ALL heads simultaneously creates an activation pattern the MLP has never seen.

This is EXACTLY what Ekdeep's framework would predict: the algorithmic-level computation (belief update, M/G weighting) is distributed across the attention-MLP circuit. You can't point to one component and say "this does ICL." The mechanism is the coupled circuit.

---

## II. Where Standard Tools Fail (and Why This Matters)

### SAEs / Transcoders

Our Stage A failure (sparse transcoders at a single MLP layer gave near-zero effect) is a textbook case of what Ekdeep warns about: **SAEs assume the Linear Representation Hypothesis (v ≈ Dz, ||z||_0 < K), but the ICL rescue signal may not be a sparse feature.**

The rescue signal is a DIRECTION in residual stream space that accumulates across multiple global attention heads over multiple layers. It's not a single feature that fires; it's a distributed shift in the residual stream's geometry. This is closer to what Liv Gorton calls "angular superposition" — the ICL signal might be one of exponentially many nearly-orthogonal directions that each carry small amounts of information, which sum to the full rescue.

However, Gemma Scope 2 transcoders COULD still be useful — not for finding THE rescue feature, but for decomposing WHAT the MLP at each layer computes:
- L12 transcoder: What features does L12 MLP produce that are destructive? Hypothesis: features adapted to ICL-modified residual streams that conflict with ZS streams.
- L17 transcoder: What features carry the content-specific signal? Should show transliteration-relevant features for Hindi but not Kannada.
- L25 transcoder: What features convert the accumulated signal to the correct output token?

The cross-layer transcoders (available for 270M and 1B) could build attribution graphs showing how information flows from ICL examples through global heads through MLPs to the output. This wouldn't find a single "rescue feature" but would map the full circuit.

### Causal Abstraction

Our activation patching IS causal abstraction. The key finding is that the clean causal story ("attention extracts, MLP transforms") is too simple. The actual story is:
- Attention and MLP are tightly coupled
- Perturbing one head is tolerable; perturbing all attention is catastrophic
- The causal structure is not a clean DAG of separable components

This connects to Ekdeep's point: "Where does the hypothesis come from?" Our initial hypothesis (the signal is in MLP features at one layer) came from the standard MI playbook. It was WRONG. The correct hypothesis (coupled multi-layer circuit at global layers) required understanding the ARCHITECTURE (the 5:1 local/global pattern) and the ALGORITHM (evidence accumulation with finite attention budget).

---

## III. What Makes This Paper Potentially Important (Carlini's Criteria)

### 1. "Do something only you can do"

**Gemma 3's 5:1 local/global architecture is unique.** No other production model has this clean interleaving pattern. It creates a natural experiment:
- Global layers: CAN see ICL examples → ICL processing happens here
- Local layers: CANNOT see ICL examples (when prompt is long) → must work with residual stream deposits

This architectural feature means our findings about global-layer bottlenecks, attention dilution, and the sliding window cliff are SPECIFIC to Gemma 3. They cannot be replicated by studying GPT, LLaMA, or Mistral (which use full attention at all layers). Other researchers studying those models would never discover these phenomena.

At the same time, the PRINCIPLES we discover (finite attention budget as computational constraint, coupled attention-MLP circuits, M/G transition modulated by pretrained knowledge) are GENERAL and should apply to any model with architectural bottlenecks.

### 2. "Have focus" — ONE idea

**Our one idea:** ICL rescue in Gemma 3 is an emergent property of coupled attention-MLP circuits operating at global attention bottleneck layers, where the model's pretrained knowledge determines whether ICL provides genuine content (generalizing) or merely activates a format mode (memorizing).

Everything connects to this:
- The architecture creates the bottleneck (global layers only)
- The mechanism requires both attention and MLP (coupled circuits)
- The cross-language hierarchy reflects pretrained knowledge (M/G transition)
- The density degradation reflects finite attention budget (computational constraint)
- The failure of sparse tools reflects the distributed nature (not a single feature)

### 3. "Put in unreasonable effort"

What we need:
- **Consistent N=50** across ALL experiments (currently mixed N=10/20/30/50)
- **Bootstrap 95% CIs** on everything
- **5 languages** for every experiment (not just Hindi)
- **Cross-model scaling** (270M, 1B, 4B) with explicit predictions
- **Transcoder decomposition** at key layers
- **Position control** (helpful examples at end vs beginning)
- **Fit Ekdeep's η model** to our density/language data

### 4. "The maximal version" — no obvious improvements left undone

Currently missing experiments that a reviewer would immediately ask for:
1. **Joint attention+MLP intervention**: Patch both simultaneously at global layers. If this recovers rescue while attention-only doesn't, it PROVES the coupled circuit claim.
2. **Position experiment**: Put 16 helpful examples at the END of a 64-example prompt. Prediction: much less degradation because both global AND local layers see them.
3. **Transcoder analysis at L12/L17/L25**: Decompose what the MLP computes.
4. **Cross-model predictions**: Explicit, falsifiable predictions for 270M and 4B before running experiments.
5. **Function vector extraction**: Extract L11 H0's function vector and test if it alone can rescue.

### 5. "Pick ideas for impact" — is this important?

**Yes, for three reasons:**

1. **Architecture-aware interpretability is underexplored.** Most MI work treats models as generic transformer stacks. Gemma 3's local/global interleaving shows that architecture fundamentally shapes the mechanism. As more diverse architectures emerge (Mamba, RWKV, hybrid models), understanding HOW architecture constrains mechanism becomes critical.

2. **The "attention alone fails" finding challenges a common assumption.** Many MI papers implicitly assume attention heads are the primary carriers of ICL signal (induction heads, function vector heads). Our finding that attention-only patching gives PE ≈ 0 while individual heads show large effects reveals a deeper truth about coupled circuits.

3. **The M/G transition in production models connects theory to practice.** Ekdeep's M/G framework was developed on model systems (finite Markov chains). Showing it manifests in Gemma 3 on a real task (transliteration across 5 languages with varying resource levels) validates and extends the theory.

---

## IV. Cross-Model Scaling Predictions (Falsifiable)

Before running 270M and 4B experiments, we commit to these predictions:

### Gemma 3 270M Predictions:
1. **Fewer global layers** → mechanism even MORE concentrated at global bottleneck
2. **Less pretrained knowledge** → ALL languages should be FORMAT-only (M-dominated). Content-specificity ratio ≈ 1.0 across all languages.
3. **Lower zero-shot rank for target tokens** → model "knows" the scripts less well
4. **Steeper density degradation** → fewer global layers = smaller total attention budget = faster dilution
5. **Transcoder analysis**: L_global MLP features should be mostly script-detection (format), not character-mapping (content)

### Gemma 3 4B Predictions:
1. **More global layers OR wider global heads** → mechanism more distributed, less concentrated
2. **More pretrained knowledge** → Content-specificity extends to more languages. Bengali and possibly Gujarati should show h/c ratio > 1.5.
3. **Higher zero-shot rank** → model already "almost knows" the task
4. **Shallower density degradation** → more capacity = higher tolerance for dilution
5. **Possible zero-shot success for Hindi** → enough pretrained knowledge that ICL isn't needed
6. **The L11-equivalent head should be less dominant** → the mechanism is more distributed across more heads

### What confirms vs. disconfirms our theory:

**Confirming**: 270M shows format-only ICL (M-dominated) across all languages; 4B shows content-specific ICL (G-dominated) for more languages than 1B.

**Disconfirming**: If 270M shows content-specific ICL for Hindi, our "pretrained knowledge modulates M/G" claim is wrong. If 4B shows format-only ICL for all languages, the scaling prediction fails.

---

## V. Transcoder Experimental Plan

### What Gemma Scope 2 Gives Us:
- SAEs AND transcoders for every layer/sublayer of Gemma 3 270M, 1B, 4B, 12B, 27B
- Cross-layer transcoders for 270M and 1B (enables attribution graphs)
- Matryoshka training for variable-granularity features

### Experiment T1: MLP Decomposition at Key Layers (1B)

For each of L12, L17, L25:
1. Run the transcoder on the MLP's input (residual stream) for both ICL and ZS prompts
2. Compare which latent features fire in ICL vs ZS
3. For L12 (destructive): Which features fire in ICL that DON'T fire in ZS? These are the features adapted to ICL-modified residual streams.
4. For L17 (constructive, content-specific): Which features fire ONLY for helpful ICL and not corrupt? These carry the content signal.
5. For L25 (readout): Which features map to target-script output tokens?

**Success criterion**: L17 features should differ between Hindi (content-specific) and Kannada (format-only). L12 features should explain WHY transplanting ICL MLP activations to ZS is destructive.

### Experiment T2: Cross-Layer Attribution Graph (270M and 1B)

Using cross-layer transcoders:
1. Build the full attribution graph from ICL example tokens → L5 global attention → L11 global attention → L17 global attention → L23 global attention → output
2. Identify which features at each layer correspond to:
   - Script detection (format)
   - Character mapping (content)
   - Task instruction encoding
3. Compare 270M vs 1B: Does 1B have richer content features? Does 270M have only format features?

**Success criterion**: The attribution graph should show a clear flow from ICL tokens through global heads to script/character features, with the content path being richer in 1B than 270M.

### Experiment T3: Feature Steering (1B)

Once we identify the key features from T1:
1. Take the L17 content features that distinguish helpful from corrupt ICL for Hindi
2. Artificially activate these features in the ZS context (feature steering)
3. Does this rescue the model without any ICL examples?

**This would be the strongest possible evidence** that these features carry the ICL content signal. If feature steering at L17 with specific transcoder features can rescue transliteration, we have found the feature-level mechanism.

---

## VI. The Honest Assessment of Novelty

### What is genuinely novel in our work:

1. **Architecture-mechanism coupling**: First demonstration that the 5:1 local/global attention pattern creates a specific ICL mechanism with global layers as bottleneck, local layers as amplifiers, and a hard sliding window cliff.

2. **Coupled circuit non-separability**: The finding that attention-only patching gives PE ≈ 0 while individual heads show large effects. This reveals coupled attention-MLP circuits that cannot be decomposed. This challenges the standard MI playbook of component isolation.

3. **Content-specificity hierarchy as M/G transition**: First mechanistic evidence of Ekdeep's memorizing/generalizing transition in a production model, modulated by pretrained knowledge across languages.

4. **Attention dilution + window loss as separable degradation mechanisms**: Two distinct causes of performance decline with more examples, measurable and separable.

5. **Failure analysis of sparse tools**: Documenting WHY transcoders failed (wrong component — signal is in attention-mediated residual stream shifts, not MLP features) provides guidance for the field.

### What is confirmatory (validates but doesn't extend prior work):

1. **Global layers attend to ICL examples** — expected from the architecture, not a discovery
2. **MLPs contribute to ICL processing** — consistent with Merullo et al. 2025, Todd et al. 2024
3. **Head attribution identifies important heads** — standard MI technique applied to a new model
4. **More examples can hurt** — known from many-shot literature (Agarwal et al. 2024)

### What is NOT novel (we should not overclaim):

1. The basic idea that ICL involves belief updating (Ekdeep et al. 2025)
2. The idea that pretrained knowledge affects ICL strategy (Yin & Steinhardt 2025, Wurgaft et al. 2025)
3. The idea that attention and MLP work together (well established in the field)

### Our honest positioning:

We are NOT proposing a new theory of ICL. We are providing **the first detailed mechanistic case study of ICL in an architecturally heterogeneous model (interleaved local/global attention), revealing how architecture constrains mechanism, and showing that the M/G theoretical framework manifests concretely in production models through a content-specificity hierarchy modulated by pretrained knowledge.**

---

## VII. Paper Structure (One Idea, Maximal Execution)

### Title candidates:
- "Architecture Constrains Mechanism: How Interleaved Attention Shapes In-Context Learning in Gemma 3"
- "The Coupled Circuit: ICL Rescue Through Global Attention Bottlenecks in Gemma 3"
- "From Format to Content: A Mechanistic Account of ICL Transliteration Across Languages and Scales"

### Structure:

**§1 Introduction** (1.5 pages)
- The puzzle: Gemma 3 1B can't transliterate zero-shot but can with ICL
- The surprise: attention alone carries effectively zero rescue signal
- The explanation: coupled attention-MLP circuits at global architectural bottleneck layers
- The gradient: pretrained knowledge determines format-only vs. content-specific ICL

**§2 Background & Related Work** (1.5 pages)
- Marr's levels applied to MI (cite Ekdeep's framing)
- ICL mechanisms: induction heads → function vectors → belief dynamics
- M/G framework (Wurgaft, Lubana et al.)
- Gemma 3 architecture: 5:1 local/global interleaving

**§3 Experimental Setup** (1 page)
- Task: Latin→Script transliteration across 5 Indic languages
- Metrics: target token probability, patching effect, rescue fraction
- Methods: activation patching, head attribution, logit lens, transcoder decomposition
- N=50 for all experiments, bootstrap 95% CIs

**§4 Results** (4 pages)

§4.1 The Architecture Creates the Bottleneck
- 4 global layers carry 99% of rescue (multi-layer patching)
- Attention patterns confirm: global layers attend to ICL, local layers don't
- L11 H0 is the universal cross-language rescue head

§4.2 Attention Alone Is Not Sufficient
- All-attention patching: PE ≈ 0
- Individual heads: large effects (L14 H0, L11 H0)
- Resolution: coupled attention-MLP circuits
- Joint attention+MLP intervention at global layers recovers rescue

§4.3 The Content-Specificity Hierarchy
- Hindi: strong content ICL (1.8× h/c ratio at L17)
- Kannada: format-only ICL (0.7× h/c ratio)
- Correlation with pretrained language representation
- Connection to M/G framework

§4.4 Attention Dilution and the Sliding Window Cliff
- Two separable degradation mechanisms
- Within-window: gradual dilution (0.270 → 0.132)
- Beyond-window: catastrophic cliff (0.132 → 0.004)
- Attention per example measurements across densities

§4.5 Transcoder Decomposition at Key Layers
- L12 MLP: destructive features adapted to ICL context
- L17 MLP: content-specific features for Hindi but not Kannada
- L25 MLP: script-readout features
- [Cross-layer attribution graph if time permits]

**§5 Scaling Analysis** (1.5 pages)
- Predictions (stated before experiments)
- 270M: format-only, steeper degradation
- 4B: content-specific for more languages, shallower degradation
- What confirmed, what surprised

**§6 Discussion** (1 page)
- Architecture as a constraint on mechanism
- Implications for MI tooling (when sparse methods fail)
- Connection to Ekdeep's model systems approach
- Limitations

**§7 Conclusion** (0.5 page)

---

## VIII. Immediate Experimental Priority (Ordered)

### Must-have for paper (run at N=50, 5 languages):

1. **Joint A+M intervention at global layers** — Patch BOTH attention output AND MLP output from ICL→ZS at L5/L11/L17/L23. If this recovers rescue while attention-only doesn't, it PROVES coupled circuit.

2. **Re-run all experiments at consistent N=50** — Head attribution (currently N=20), MLP contribution (currently N=30 Hindi only), density (currently N=10)

3. **Position control experiment** — 16 helpful examples placed at END of 64-example prompt. Prediction: much less degradation.

4. **Transcoder decomposition at L12/L17/L25** — Using Gemma Scope 2 pre-trained transcoders.

5. **Cross-model: 270M and 4B** — Same experiments on smaller/larger Gemma 3 models.

### Nice-to-have (strengthens but not essential):

6. **Function vector extraction from L11 H0** — Can L11 H0's output alone rescue?
7. **Cross-layer attribution graphs (270M, 1B)** — Full circuit map
8. **Fit Ekdeep's η(N,D) model to our data** — Quantitative connection to theory
9. **Feature steering at L17** — Can we rescue without ICL by steering features?

---

## IX. What Carlini Would Say About Our Paper Right Now

**Strengths:**
- Clean architectural story that creates a natural experiment (5:1 local/global)
- Multiple converging lines of evidence (patching, attribution, logit lens, attention patterns)
- Cross-language hierarchy adds richness
- The "attention alone fails" finding is surprising and important
- Connection to established theoretical framework (Ekdeep's M/G)

**Weaknesses Carlini would identify:**
- "Your experiments are at inconsistent sample sizes. Run everything at N=50 before claiming anything." → FIX: re-run everything.
- "You claim coupled circuits but haven't done the joint A+M intervention that would prove it." → FIX: run joint intervention.
- "You mention transcoders but haven't used them. Either use them or remove the claim." → FIX: run transcoder experiments.
- "The scaling analysis is predictions without data. Run 270M and 4B or drop it." → FIX: run experiments.
- "The paper tries to do too many things. Pick ONE of: architecture story, coupled circuits, M/G transition, transcoder analysis. Not all four." → DECISION NEEDED.

**Carlini's likely verdict:** "Good problem, interesting findings, but not yet at the 'unreasonable effort' threshold. The data is incomplete and inconsistent. Run the missing experiments, get consistent numbers, and this could be a strong paper."

---

## X. The Kernel Blocker

None of the above experiments can run until the `core/`, `config/`, `rescue_research/` modules that the multiscale_suite depends on are restored or rewritten. This is the critical path blocker. Options:
1. **Restore from git history** — If these files existed before
2. **Rewrite minimally** — Only the functions the suite actually calls
3. **Bypass the suite** — Write standalone experiment scripts that don't depend on the suite infrastructure

Option 3 is fastest but creates technical debt. Option 1 is cleanest if the files exist. This needs resolution before ANY experiment runs.
