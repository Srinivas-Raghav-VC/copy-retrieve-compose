"""
Single source of truth for the rescue research pipeline.

All n_icl, n_select, n_eval, seeds, model, layer, topk are defined here.
No experiment should use different defaults — import from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Default run directory (relative to workspace root)
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = WORKSPACE_ROOT / "rescue_research" / "results"
DEFAULT_LOG_FILE = None  # stderr only unless set

# -----------------------------------------------------------------------------
# Data split (must satisfy: n_icl + n_select + n_eval <= total words)
# Use three-way whenever selection is involved; two-way only for pure eval.
# -----------------------------------------------------------------------------
# Runtime defaults stay small for quick local checks. Main-track target counts
# are encoded below and enforced by the full pipeline modules.
N_ICL: int = 5
N_SELECT: int = 50   # For layer/feature selection (must be disjoint from eval)
N_EVAL: int = 50     # Held-out evaluation set
SEEDS: List[int] = [42, 123, 456, 789, 1337]

# Main-track target counts (per pair, per seed)
# Submission profile (depth-balanced): 100 / 400 / 1000 / 200
TARGET_ICL_BANK: int = 100
TARGET_SELECTION: int = 400
TARGET_EVAL_OPEN: int = 1000
TARGET_EVAL_BLIND: int = 200
LOCKED_PAIRS: List[str] = ["hindi_telugu", "hindi_tamil", "english_arabic", "english_cyrillic"]

# Minimum words required to avoid fallback to two-way (which causes selection bias)
MIN_WORDS_FOR_THREE_WAY: int = N_ICL + N_SELECT + N_EVAL  # 105

# Target sample size (for power): aim for n_eval >= 100 per seed, ideally 200–500.
# With ~200 words you can do at most ~50 eval per seed; use Aksharantar or a larger
# word list for 100–500 eval.

# -----------------------------------------------------------------------------
# Model and layer
# -----------------------------------------------------------------------------
DEFAULT_MODEL: str = "1b"
DEFAULT_LAYER: int = 21   # Only used after layer_sweep_cv picks one; override by result
TOP_K_VALUES: List[int] = [25]
DEFAULT_TOPK: int = 25

# -----------------------------------------------------------------------------
# Primary outcome (pre-registered)
# -----------------------------------------------------------------------------
PRIMARY_OUTCOME_DESCRIPTION: str = (
    "At pre-registered topk (DEFAULT_TOPK), mean NLL improvement > 0 and Holm-adjusted "
    "p(NLL improvement vs corrupt_control) < alpha across tested topk values on held-out eval."
)
PRIMARY_ALPHA: float = 0.05

# -----------------------------------------------------------------------------
# Pipeline stages (order matters for --stage full)
# -----------------------------------------------------------------------------
STAGES_IN_ORDER: List[str] = [
    "baseline",
    "layer_sweep_cv",
    "comprehensive",
    "mediation",
]


@dataclass
class RunConfig:
    """Config for a single run; override defaults via constructor or CLI."""

    out_dir: Path = field(default_factory=lambda: DEFAULT_OUT_DIR)
    log_file: Path | None = None

    n_icl: int = N_ICL
    n_select: int = N_SELECT
    n_eval: int = N_EVAL
    seeds: List[int] = field(default_factory=lambda: list(SEEDS))

    model: str = DEFAULT_MODEL
    layer: int = DEFAULT_LAYER
    topk_values: List[int] = field(default_factory=lambda: list(TOP_K_VALUES))

    device: str = "cuda"
    pair: str = ""   # Empty = default Hindi–Telugu from reference config
    prepared_split_dir: str = ""  # Optional: data/processed/<pair> with split_seed_<seed>.json
    use_blind_eval: bool = False
    sweep_positions: bool = False
    decoupled_control: bool = True
    patch_style: str = "sparse"
    eval_generation: bool = False
    task: str = "transliteration"
    control_mode: str = "default"

    def ensure_out_dir(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir
