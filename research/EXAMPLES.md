# Worked Examples

## Example 1: Good behavioral-degradation analysis shape

### Question
Is the observed drop from N=16 to N=32 in Hindi a real degradation effect?

### Good structure
1. restate the question precisely
2. split it into:
   - does exact drop?
   - does acceptable drop?
   - does CER worsen?
   - does first-token stay strong while full output worsens?
   - are token lengths matched?
   - do helpful and corrupt both degrade?
   - do early examples leave local visibility?
3. list competing hypotheses
4. propose the minimum disambiguating experiment
5. state what result would support each hypothesis
6. update the journal

### Why this is good
It does not answer “yes/no” from one plot.
It decomposes the phenomenon.

## Example 2: Good mechanistic-planning shape

### Question
What is the cleanest first causal localization pass for ICL rescue?

### Good structure
1. clarify whether the target is computational, algorithmic, or implementational
2. state current behavioral facts
3. propose cheap screening:
   - layerwise deltas
   - helpful vs corrupt
   - language split
4. propose exact causal follow-up:
   - grouped global-layer attention patching
   - grouped local-layer patching
   - grouped MLP patching
5. name confounds
6. define what would count as real evidence vs suggestive evidence

### Why this is good
It avoids jumping directly from behavior to a “hero head” story.
