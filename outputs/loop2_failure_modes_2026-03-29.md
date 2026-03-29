# Loop 2 failure-mode follow-up (first token vs continuation)

This memo decomposes the current Loop 2 behavior into two pieces using already-generated control artifacts:

1. **early target selection** — `mean_first_prob`, `mean_first_entry_correct`
2. **whole-word continuation** — `mean_exact_match`, `mean_akshara_cer`, plus copy-style error patterns

## Key takeaways

- **Established:** `1B × Hindi × n_icl=64` is already failing at the **first target token**. Helpful ICL is worse than zero-shot on first-token probability (0.481 vs 0.579) and first-entry correctness (0.500 vs 0.600).
- **Established:** `1B × Telugu × n_icl=64` is **not** primarily a first-token failure. Helpful ICL drives first-token probability from 0.002 to 0.869 and first-entry correctness from 0.000 to 0.900, yet exact match stays at 0.000.
- **Established:** `1B × Telugu × n_icl=64` often emits an **ICL target string** rather than the query-specific answer: 80.0% of helpful predictions are exact copies of one of the prompt-bank targets, and 88.9% of the `first-correct but exact-wrong` cases are bank copies.
- **Established:** `1B × Hindi × n_icl=64` often falls back to **Latin/source-like outputs**: 50.0% of helpful predictions are Latin-script, and 40.0% are exact source copies.
- **Supported but provisional:** `4B × Telugu × n_icl=64` is a clean positive anchor because it gets both stages right: first-token probability rises to 0.979, first-entry correctness to 0.967, and exact match to 0.300; its residual errors are mostly near-misses rather than prompt-bank copies (6.7%).
- **Supported but provisional:** `4B × Hindi × n_icl=64` is a strong-base-capability comparison cell, not a rescue-heavy cell: zero-shot is already strong (exact 0.367), and helpful ICL mostly sharpens first-token confidence rather than creating a new regime.

## Core panel table

| model | pair | n_icl | helpful first_prob | zs first_prob | Δ first_prob | helpful first_entry | zs first_entry | helpful exact | zs exact | helpful source-copy | helpful bank-copy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1b | hin | 8 | 0.430 | 0.629 | -0.199 | 0.600 | 0.767 | 0.100 | 0.067 | 0.0% | 3.3% |
| 1b | hin | 64 | 0.481 | 0.579 | -0.098 | 0.500 | 0.600 | 0.000 | 0.067 | 40.0% | 26.7% |
| 1b | tel | 8 | 0.278 | 0.002 | 0.275 | 0.233 | 0.000 | 0.000 | 0.000 | 0.0% | 10.0% |
| 1b | tel | 64 | 0.869 | 0.002 | 0.867 | 0.900 | 0.000 | 0.000 | 0.000 | 6.7% | 80.0% |
| 4b | hin | 64 | 0.871 | 0.837 | 0.034 | 0.933 | 0.900 | 0.367 | 0.367 | 0.0% | 3.3% |
| 4b | tel | 64 | 0.979 | 0.002 | 0.977 | 0.967 | 0.000 | 0.300 | 0.000 | 0.0% | 6.7% |

## Threshold check on `1B` (`n_icl = 48/56/64`)

| pair | n_icl | helpful first_prob | zs first_prob | helpful first_entry | zs first_entry | helpful exact | zs exact | helpful CER | zs CER | helpful bank-copy | helpful Latin-script |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hin | 48 | 0.488 | 0.601 | 0.433 | 0.667 | 0.033 | 0.067 | 4.561 | 0.709 | 6.7% | 43.3% |
| hin | 56 | 0.387 | 0.581 | 0.400 | 0.700 | 0.033 | 0.067 | 1.379 | 0.660 | 13.3% | 56.7% |
| hin | 64 | 0.481 | 0.579 | 0.500 | 0.600 | 0.000 | 0.067 | 1.343 | 0.705 | 26.7% | 50.0% |
| tel | 48 | 0.722 | 0.002 | 0.700 | 0.000 | 0.000 | 0.000 | 1.043 | 0.978 | 53.3% | 26.7% |
| tel | 56 | 0.807 | 0.002 | 0.800 | 0.000 | 0.000 | 0.000 | 0.919 | 0.985 | 70.0% | 20.0% |
| tel | 64 | 0.869 | 0.002 | 0.900 | 0.000 | 0.000 | 0.000 | 0.751 | 1.068 | 80.0% | 10.0% |

## Failure-mode interpretation

### 1B Hindi: early target-selection failure

Observed at `n_icl=64`: helpful ICL is worse than zero-shot on first-token probability (0.481 vs 0.579), worse on first-entry correctness (0.500 vs 0.600), and worse on exact match (0.000 vs 0.067).

Helpful outputs are often not even in the right script: 50.0% Latin-script predictions, 40.0% exact source copies.

This points to a **routing / target-selection problem before whole-word continuation**, not just a later suffix-composition issue.

Representative failures:

- `aadeshpaalak` → gold `आदेशपालक` but predicted `आदेशप्रताप` (bank copy: `True`)
- `aadhikarik` → gold `आधिकरिक` but predicted `आधिपतिक` (bank copy: `False`)
- `aabhyantarikaran` → gold `आभ्यंतरीकरण` but predicted `आभ्यन्तरिकृत` (bank copy: `True`)
- `aadhyatmakta` → gold `आध्यात्मकता` but predicted `आध्यदेश` (bank copy: `True`)
- `aadarahi` → gold `आदरही` but predicted `आदिशक्तिका` (bank copy: `True`)

### 1B Telugu: continuation / retrieval-composition failure

Observed at `n_icl=64`: helpful ICL greatly improves first-token probability (0.002 → 0.869) and first-entry correctness (0.000 → 0.900), but exact match stays at `0.000`.

The dominant error mode is prompt-bank copying: 80.0% of helpful predictions are exact copies of one of the 64 prompt-bank targets.

Among the items where the model gets the first entry correct but still misses the whole word, 88.9% are bank copies. This is much closer to **retrieving the wrong full answer from the prompt bank** than to failing to choose the target script.

Representative failures:

- `aadaallameeda` → gold `ఆడాళ్ళమీద` but predicted `ఆడాళ్లదీ` (bank copy: `True`)
- `aacharanaatmakathatho` → gold `ఆచరణాత్మకతతో` but predicted `ఆచరణాత్మకమైనదో` (bank copy: `True`)
- `aacharanaatmamgaa` → gold `ఆచరణాత్మంగా` but predicted `ఆచరించడం` (bank copy: `False`)
- `aacharinchabadedi` → gold `ఆచరించబడేది` but predicted `ఆచరింపదగిన` (bank copy: `True`)
- `aabharanamgaano` → gold `ఆభరణంగానో` but predicted `ఆభరణాల్లోనో` (bank copy: `True`)

### 4B Telugu: the clean anchor remains clean

`4B × Telugu × n_icl=64` improves both early and late metrics. Helpful exact match reaches 0.300, while helpful bank-copy rate is only 6.7%. Residual errors are usually near-misses rather than verbatim prompt-bank retrieval.

## Claim ledger

- **Established:** `1B × Hindi` high-shot fragility includes an early target-selection failure; helpful ICL often pushes the model toward Latin/source-like or wrong-bank outputs rather than toward the correct Devanagari target.
- **Established:** `1B × Telugu` high-shot fragility is mainly not a first-token problem; it is a continuation/retrieval-composition problem in which the model often emits a prompt-bank target string that matches the script and sometimes the first character, but not the query-specific transliteration.
- **Supported but provisional:** the `1B` story is therefore **not unitary** across languages. Hindi and Telugu appear to fail at different stages of the computation.
- **Supported but provisional:** visibility remains a contributing architectural factor, but these failure-mode differences show that visibility alone cannot explain the `1B` behavior.

## Recommended next bounded experiments

1. **1B Hindi — first-token competition audit.** Build a small teacher-forced probe that records the correct target token, the emitted top competitor token, and whether the competitor is Latin/source-like or a wrong Devanagari bank token. This directly tests the early-routing hypothesis.
2. **1B Telugu — prompt-bank copy audit with nearest-neighbor controls.** Measure how often the model outputs an in-context target exactly, and whether those copies correspond to the most orthographically similar prompt example rather than the correct answer. This directly tests the retrieval-composition hypothesis.
3. **Only after those two audits:** decide whether causal interventions should target early token selection (`1B Hindi`) or later query-specific continuation (`1B Telugu`).

## Sources

- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/raw/1b/aksharantar_hin_latin/nicl8/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/raw/1b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/raw/1b/aksharantar_tel_latin/nicl8/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/raw/1b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/raw/4b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/raw/4b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/raw/1b/aksharantar_hin_latin/nicl48/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/raw/1b/aksharantar_hin_latin/nicl56/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/raw/1b/aksharantar_hin_latin/nicl64/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/raw/1b/aksharantar_tel_latin/nicl48/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/raw/1b/aksharantar_tel_latin/nicl56/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/raw/1b/aksharantar_tel_latin/nicl64/neutral_filler_recency_controls.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/loop2_full/score.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/loop2_vm_controls/threshold_1b_seed42/score.json
- file:///mnt/d/Research/Honors/research/results/autoresearch/token_visibility_v1/results/token_visibility_summary.csv
