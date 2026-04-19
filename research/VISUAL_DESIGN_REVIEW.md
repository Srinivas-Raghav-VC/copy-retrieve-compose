# Visual Design Review — Honors Thesis Figures & Deck

**Date:** 2026-03-30  
**Scope:** `gemma_1b_icl_paper_v11.tex`, `plot_final_paper_figures.py`, all three PNG figures, and the TikZ phase-map diagram.

---

## 1. What the Current Figure System Does Well

### Color & theme
- The Okabe-Ito-inspired palette (`#d55e00` 1B / `#0072b2` 4B / `#009e73` green / `#cc79a7` magenta) is **genuinely colorblind-safe** and consistent across all three figures. This is publication-ready.
- The `seaborn-v0_8-whitegrid` base with the custom `_style_axis` treatment (hidden top/right spines, alpha-faded left/bottom) is clean and modern — better than most interp papers.
- 260 DPI export is good enough for print.

### Fig 1 — Behavioral regime summary
- The **dumbbell/lollipop** encoding (1B dot ↔ 4B dot with connector line) is the right chart type for paired comparison. It immediately shows that Hindi's gap is qualitatively different from Telugu's.
- The Indo-Aryan / Dravidian background spans (`tan` / `mint`) are a simple but effective grouping that previews the family-level story.
- The dual-panel layout (exact match left, CER right) avoids dual-axis confusion.

### Fig 3 — Telugu bottleneck
- Horizontal bar chart with inline value labels is the right encoding for "which intervention had the biggest delta."
- The necessity/sufficiency grouping with colored background bands directly mirrors the text's conceptual structure.
- Clean 1B / 4B side-by-side lets the reader compare without mental gymnastics.

### Fig 4 — Hindi practical patch
- The 2×2 grid logically separates the narrative: selection sweep → held-out eval → follow-up controls.
- Error bars with CI whiskers on Panel B are the right level of statistical honesty for a 60-item eval.

### Structural flow
The paper's visual arc is: *behavioral overview (Fig 1) → conceptual map (Fig 2) → Telugu mechanistic (Fig 3) → Hindi practical (Fig 4)*. That's a good narrative spine: scope → framework → evidence × 2.

---

## 2. What Is Visually Weak or Risky for a Final-Review Honors Deck

### 🔴 The TikZ phase map (Fig 2) — weakest asset, high examiner-visibility

**Problems as rendered in the PDF (page 4):**

1. **Text-box overlap.** The "Marathi is the key control" annotation box collides visually with the `1B Marathi` label. In the rendered PDF the box left edge sits directly on the y-axis arrow, obscuring the arrow.
2. **Purely qualitative scatter with no scale.** The axes have no tick marks, gridlines, or numeric labels. The figure caption says "interpretive rather than directly measured," but examiners will read this as "we eyeballed it." This is the single most vulnerable figure in the paper for a tough reviewer.
3. **Note boxes dominate the figure.** The two `note` rectangles take ~40% of the figure area. They're essentially caption text placed inside the figure, which breaks the figure/caption contract.
4. **No visual encoding beyond position.** 1B (orange) vs 4B (blue) is the only distinguishing mark. Script family (Devanagari vs Dravidian vs Bengali) has no marker shape or other encoding, even though it's the main analytical dimension.
5. **No directional narrative.** The figure doesn't show 1B→4B improvement trajectory per language. A reader has to mentally link "1B Hindi" to "4B Hindi" across the scatter.
6. **Circle sizing.** The `minimum size=7mm` circles are fine for the paper but will be tiny on a projected slide.

**Risk level:** This figure is the thesis's central conceptual claim in visual form. If it looks hand-drawn or ungrounded, it undermines the two-axis story.

### 🟡 Hindi practical patch (Fig 4) — information-dense but cramped

1. **Label clipping.** In the rendered PDF, the "sign flip" x-label on Panel C is partially cut off / overlapping with "ΔEM -0.02" annotation. The `rotation=14` isn't enough.
2. **Δgap annotations crowd the bars.** In Panel B, `Δgap 6.51` overlaps with the green bar top and the legend text ("Δ exact match" / "Δ CER improvement"). At 260 DPI in a single-column figure this becomes hard to read.
3. **Suptitle proximity.** `fig.suptitle(..., y=1.02)` in a `constrained_layout=True` figure means the super title sits very close to Panel A's title, creating a crowded header.
4. **Four panels with different y-axis scales.** Panel A and bottom-left share x-axis but not y-axis, Panel B and C have unrelated y-axes. The reader's eye needs a moment to orient — acceptable for a paper, but risky for a slide where you get ~10 seconds of attention.

### 🟡 Telugu bottleneck (Fig 3) — minor polish issues

1. **"Necessity" / "sufficiency" text labels** are placed at `xmin + 0.15` which puts them very close to the left edge. In the rendered PDF, "necessity" and "sufficiency" are low-contrast gray text partially behind the tan/mint background — easily missed.
2. **Bar color inconsistency.** The same blue (`COLORS["4b"]`) is used for "writer only" and "site donor" even though those are different intervention types. Color here carries model meaning (4B) in other figures but intervention-type meaning here. Could confuse a reader flipping back.

### 🟡 Missing figure types

1. **No task example figure.** The paper never shows a concrete input/output example (e.g., "Latin input: `namaste` → Hindi target: `नमस्ते` → model output: `namaste`"). For an honors thesis with examiners who may not know transliteration, this is a missed opportunity. A single-row example table or a schematic would ground the entire paper.
2. **No architecture/layer diagram.** The mechanistic sections reference L25 MLP, L26 layer_output, L34, sliding-window vs global layers — but there's no visual showing where these sites live inside the Gemma architecture. A simple vertical stack diagram with highlighted intervention sites would massively help a slide deck.
3. **No "what the model actually does wrong" qualitative panel.** Showing 3–4 concrete failure examples (Latin fallback, bank-copy, correct, near-miss) side by side would make the phase map immediately intuitive.

---

## 3. How to Improve the TikZ Two-Axis Phase Map

### Strategy: semi-quantitative scatter + trajectory arrows + region zones

The figure should be rebuilt to sit halfway between a hand-drawn conceptual sketch and a data-driven scatter. It doesn't need to be fully measured, but it needs enough visual structure to look grounded.

### Concrete implementation plan

#### A. Use semi-quantitative axes from actual probes
- **X-axis (late query-specific continuation):** use the continuation gap or sequence gap metric, normalized to [0, 1]. Telugu 1B has a known gap of ~−1.41, 4B Telugu is strongly positive, etc.
- **Y-axis (early target/task entry):** use the first-token target rate directly. Hindi 1B ≈ 0.47 helpful, Marathi 1B ≈ 0.75, Telugu 1B ≈ 0.90+, Bengali 1B ≈ 0.017, etc.
- Add **light gridlines** at 0.25 / 0.5 / 0.75. Label the axes with both the conceptual name and the operationalization (e.g., "early target entry (first-token target rate)").

#### B. Add region zones
Replace the empty background with three **labeled translucent regions**:
- **Bottom-left:** "Collapse" (red-tinted, very faint) — where both axes are weak
- **Top-left / bottom-right:** "Partial" (amber-tinted) — one axis works, the other doesn't
- **Top-right:** "Composition" (green-tinted) — both axes succeed

This directly maps to the paper's three regimes: collapse, bank-retrieval, and stable composition.

#### C. Add 1B→4B trajectory arrows
For each language, draw a **thin arrow** from the 1B point to the 4B point. This immediately shows:
- Hindi's dramatic upward-right jump
- Telugu's mainly rightward jump (late improvement)
- Bengali's large jump from bottom-left
- Marathi's smaller shift (already high-early in 1B)

The arrow makes the "scale helps both axes" claim visual.

#### D. Use marker shapes for script family
- **Circle:** Devanagari (Hindi, Marathi)
- **Square:** Dravidian (Telugu, Tamil)
- **Diamond/Triangle:** Eastern (Bengali)

This adds a third visual channel without clutter.

#### E. Move annotations to a compact legend or margin
Kill the two big annotation boxes. Replace with:
- A small legend (script-family shapes, 1B/4B colors, optional: region labels)
- If the Marathi control story is critical, use a **single callout line** (not a box) pointing to the Marathi dot, saying "same script, different late outcome"

#### F. Implementation choice: TikZ vs matplotlib
Two options:
1. **Keep TikZ** but rewrite with `pgfplots` scatter + `fill between` for regions + `\draw[->]` for trajectory arrows. This stays self-contained in the .tex.
2. **Move to matplotlib** (add to `plot_final_paper_figures.py`) and export as PNG like the other three figures. This gives exact coordinate control and matches the visual language of the other figures.

**Recommendation:** Option 2 (matplotlib). Reason: the other three figures are all matplotlib PNGs. A lone TikZ figure looks visually inconsistent (different font rendering, different grid style, different spacing). Unifying all figures in one plotting script also makes the pipeline reproducible.

#### G. Concrete code sketch (matplotlib)

```python
def plot_two_axis_phase_map() -> None:
    """Semi-quantitative 2-axis regime map with trajectory arrows."""
    # Data: (first_token_target_rate, continuation_gap_normalized)
    # Normalize late gap to [0,1] using observed range
    data = {
        "1B Hindi":   (0.47, 0.10),   "4B Hindi":   (0.95, 0.85),
        "1B Marathi": (0.75, 0.30),   "4B Marathi": (0.92, 0.88),
        "1B Telugu":  (0.90, 0.15),   "4B Telugu":  (0.95, 0.90),
        "1B Tamil":   (0.63, 0.12),   "4B Tamil":   (1.00, 0.80),
        "1B Bengali": (0.02, 0.05),   "4B Bengali": (0.58, 0.55),
    }
    # Region fills: axhspan/axvspan or Polygon patches
    # Trajectory arrows: ax.annotate with arrowprops
    # Marker shapes: dict mapping script family to marker
    # Single callout for Marathi
    ...
```

**Key values would be pulled from the actual JSON result files** (the aggregate seed JSON and the first-token audit JSONs already in the repo), not hardcoded.

---

## 4. Recommended Slide-by-Slide Visual Arc for Honors Defense Deck

### Design principles for the deck
- **One idea per slide.** Examiners process projected content differently from paper pages.
- **Bigger text, less clutter.** Minimum 20pt for labels, 24pt for titles.
- **Build-up animations** where the story is layered (e.g., show 1B dots first, then add 4B arrows).
- **Dark-on-light theme**, matching the paper palette. Do not introduce new colors.

### Slide sequence

| # | Slide title | Visual | Purpose |
|---|-------------|--------|---------|
| 1 | **Title slide** | Title, author, institution, date. Optionally a small transliteration example in the corner (Latin → Devanagari) | Anchor |
| 2 | **What is transliteration?** | **NEW FIGURE:** 3-column example panel: Latin input → gold target → model output. Show one good case (4B Telugu), one failure mode (1B Hindi staying Latin), one bank-copy case (1B Telugu). Use script-colored boxes. | Ground the examiner in the task |
| 3 | **Why this task for studying ICL?** | Bullet list (3 items max) + a small schematic showing "strict, legible, auditable failures" | Motivation |
| 4 | **Gemma 3 architecture sketch** | **NEW FIGURE:** Vertical layer stack showing sliding-window (gray) and global-attention (blue) layers, with red markers at L25 MLP and L26/L34 layer_output. Just enough to orient the examiner on "where things happen." | Architecture context |
| 5 | **Behavioral phase structure** | Fig 1 (behavioral regime summary), full-slide width. Animate: show 1B points first, pause, then add 4B points. | Core behavioral result |
| 6 | **The two-axis regime map** | **IMPROVED Fig 2** (semi-quantitative matplotlib version). Animate: show region zones first, add 1B dots, then draw 1B→4B arrows. | Central conceptual claim |
| 7 | **Phase-map table** | Table 1 from the paper, reformatted as a slide table with color-coded cells (red=weak, yellow=partial, green=strong) | Examiner reference |
| 8 | **Marathi: the key same-script control** | Callout version of the phase map zoomed to Hindi/Marathi cluster. Show the dissociation: high early, low late. One concrete example pair. | Strongest control argument |
| 9 | **Hindi mechanism: where the failure lives** | **NEW FIGURE:** Simple layer diagram with L25 MLP highlighted + logit competition bar showing target vs Latin at first token under helpful vs zero-shot. | Mech localization |
| 10 | **Hindi: the two-channel core** | Compact version of the channel selection result. Bar chart: chosen pair [5486, 2299] vs random pairs at explaining the margin. | Mech specifics |
| 11 | **Practical Hindi patch** | Fig 4, but **split across two slides**: (A) selection sweep + result bars on slide 11, (B) signed steering vs lesioning on slide 12. Less cramped. | Intervention result |
| 12 | **Hindi patch: steering > lesioning** | Panel C from Fig 4, enlarged. One takeaway: "calibrated signed shift is much stronger than simple ablation." | |
| 13 | **Telugu mechanism: a different bottleneck** | Fig 3 (Telugu bottleneck). Animate: show necessity panel first, then sufficiency. | Second mech case |
| 14 | **Telugu vs Hindi: contrastive practical result** | Table 2 (Telugu practical patch) side by side with the Hindi practical result. Highlight: "Hindi patch works; Telugu patch doesn't — this is informative." | Contrastive claim |
| 15 | **Out-of-core predictive check** | Table 3 (predictive check), reformatted as a slide table with prediction vs observation columns. | Validation |
| 16 | **What this means** | The phase map one more time (callback), with a one-sentence overlay: "Multilingual ICL in Gemma 3 fragments into behaviorally legible, sometimes mechanistically editable regimes." | Synthesis |
| 17 | **Limitations & open questions** | 3–4 bullets. Honest. | Scientific honesty |
| 18 | **Thank you / questions** | Phase map as subtle background. Contact info. | Close |

### Total: ~18 slides, ~20 minutes + 10 min Q&A.

### New figures needed for the deck (not currently in the repo)

1. **Task example panel** (slide 2): 3-column transliteration examples with failure mode labels. Could be a simple matplotlib text-box figure or a TikZ/Beamer box.
2. **Architecture layer diagram** (slide 4): vertical stack of ~26 layers for 1B, with intervention sites marked. Simple TikZ or matplotlib.
3. **Improved phase map** (slide 6): as described in Section 3 above.

---

## 5. Priority Action Items

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Rebuild TikZ phase map as semi-quantitative matplotlib figure with trajectory arrows and region zones | 2–3 hrs | Fixes the weakest and most examiner-visible figure |
| **P0** | Create task example panel (3 concrete input/output/failure examples) | 1 hr | Grounds examiners who don't know transliteration |
| **P1** | Create architecture layer diagram with intervention sites marked | 1–2 hrs | Orients mechanistic sections |
| **P1** | Fix Hindi patch figure label clipping and Δgap annotation crowding | 30 min | Polish |
| **P1** | Increase "necessity"/"sufficiency" label contrast in Telugu figure | 15 min | Polish |
| **P2** | Build Beamer slide deck using the arc above | 4–6 hrs | Defense prep |
| **P2** | Add a qualitative failure-mode panel (4 example outputs, labeled) | 1 hr | Strengthens examiner intuition |

---

## Summary

The paper's plotting infrastructure is solid — the palette, the theme, the figure types are all publication-quality. The **one critical weakness** is the TikZ phase map, which is the conceptual centerpiece of the thesis but looks ungrounded and cluttered. Rebuilding it as a semi-quantitative matplotlib scatter with trajectory arrows, region zones, and script-family marker shapes would transform it from the weakest figure into the strongest. The other two matplotlib figures need minor polish (label clipping, annotation overlap, color consistency). For the defense deck, the main gaps are a task-example grounding figure, an architecture diagram, and splitting the dense 4-panel figures across separate slides.
