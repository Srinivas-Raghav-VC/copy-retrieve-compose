# AI Interpretability Research Session Optimization

Based on analysis of multiple research workflow frameworks.

## Current Setup

### Interpretability Core (INSTALLED)
|Tool | Purpose | Status |
|------|---------|--------|
| transformer-lens | Mechanistic interpretability | ✓ Skill installed |
| nnsight | Remote interpretability (70B+) | ✓ Skill installed |
| pyvene | Causal interventions | ✓ Skill installed |
| safelens | Sparse autoencoder training | ✓ Skill installed |

### Paper Research (INSTALLED)
| Tool | Purpose | Status |
|------|---------|--------|
| pdfmux | PDF extraction with confidence | ✓ Installed v1.3.0 |
| pdftotext | Fast text extraction | ✓ Installed |
| alphaxiv MCP | Paper lookup | ⚠ Needs OAuth fix |
| arxiv-mcp-server | ArXiv search | ✓ Enabled |

### Infrastructure (INSTALLED)
| Tool | Purpose | Status |
|------|---------|--------|
| wandb | Experiment tracking | ✓ Skill installed |
| tensorboard | Training visualization | ✓ Skill installed |
| opencode | This session | ✓ Running |

---

## Recommended Additions

### 1. Research Pipeline Skills (PRIORITY: HIGH)

**Source**: `Imbad0202/academic-research-skills` (906 stars)

This is the most comprehensive research skill suite. Install:

```bash
# Clone the repository
git clone https://github.com/Imbad0202/academic-research-skills ~/.claude/academic-research-skills

# Copy skills to your commands
mkdir -p ~/.claude/commands
cp -r ~/.claude/academic-research-skills/deep-research/SKILL.md ~/.claude/commands/deep-research.md
cp -r ~/.claude/academic-research-skills/academic-paper/SKILL.md ~/.claude/commands/academic-paper.md
cp -r ~/.claude/academic-research-skills/academic-paper-reviewer/SKILL.md ~/.claude/commands/academic-paper-reviewer.md
```

**Key Features**:
- 13-agent deep research pipeline with 7 modes (full, quick, socratic, systematic-review)
- Integrity verification agent (anti-hallucination)
- Socratic SCR Loop (prevents confirmation bias)
- Cross-skill data contracts for pipeline integrity

### 2. Paper Review Skill (PRIORITY: HIGH)

**Source**: `claesbackman/AI-research-feedback`

```bash
mkdir -p ~/.claude/commands
curl -o ~/.claude/commands/review-paper.md \
  https://raw.githubusercontent.com/claesbackman/AI-research-feedback/main/Paper-review/review-paper.md
```

**Key Features**:
- 6-agent parallel review (grammar, consistency, claims, math, figures, contribution)
- Journal-specific personas (AER, QJE, JPE, Econometrica, etc.)
- "Claim discipline" - ensures claims never exceed identification

### 3. Interpretability Python Packages

```bash
pip install --break-system-packages \
  captum \           # PyTorch interpretability
  umap-learn \       # Dimensionality reduction
  plotly \           # Interactive visualizations
  scikit-learn \     # Linear probes, PCA
  transformers \     # HuggingFace models
  einops \           # Tensor operations
  datasets           # Benchmark datasets
```

### 4. MINT Lab Workflow Pattern (PRIORITY: MEDIUM)

From the Philosophy of Computing article, the MINT Lab workflow:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PAPER FEED │ ──► │  PDF FETCH  │ ──► │   CORPUS    │
│ (Twitter/X, │     │ (pdfmux +   │     │ (Vector     │
│  ArXiv,     │     │  pdftotext) │     │  Store)     │
│  Substack)  │     │             │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐     ┌──────▼──────┐
                    │  REPORTS    │ ◄───│   SEARCH    │
                    │ (Agent-gen │     │ (Semantic) │
                    │  summaries)│     │             │
                    └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │CONVERSATION │
                    │ (Back-forth │
                    │  about      │
                    │  corpus)    │
                    └─────────────┘
```

**Implementation**:
- Use alphaxiv/arxiv MCP for paper discovery
- Use pdfmux for extraction
- Use sentence-transformers for vector embeddings
- Use a vector DB (chroma/qdrant/pinecone) for semantic search

### 5. DAAF Repository

**Status**: Repository not found publicly

The `DAF-Contribution-Community/daaf` repository doesn't exist publicly. If you have access to it:
- Share the correct URL
- Or provide the local path

---

## Interpretability-Specific Workflow

### For a New Interpretability Project:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INTERPRETABILITY WORKFLOW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. RESEARCH (deep-research systematic-review mode)                 │
│     → "Systematic review of attention visualization methods"        │
│     → PRISMA flow diagram, RoB assessment                          │
│     → Outputs: RQ Brief, Bibliography, Synthesis Report            │
│                                                                      │
│  2. INTEGRITY CHECK (Stage 2.5)                                     │
│     → Verify citations to Anthropic/OpenAI model papers             │
│     → Check statistical claims about metrics (AOPC, fidelity)      │
│     → Validate ground truth labels in datasets                      │
│                                                                      │
│  3. EXPERIMENT (transformer-lens / nnsight)                        │
│     → Run mechanistic interpretability analysis                     │
│     → Log to wandb/tensorboard                                      │
│     → Use SAE for feature analysis                                   │
│                                                                      │
│  4. WRITE (academic-paper full mode)                                │
│     → 12-agent paper writing with visualizations                    │
│     → LaTeX output with apa7 class                                   │
│                                                                      │
│  5. REVIEW (academic-paper-reviewer)                                 │
│     → 5-person review with Devil's Advocate                          │
│     → Claim discipline check on interpretability claims             │
│                                                                      │
│  6. REVISE + FINALIZE                                                │
│     → Two rounds of revision                                         │
│     → Publication-ready PDF                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation Summary

### Essential (Install Now)
```bash
# 1. Research pipeline skills
git clone https://github.com/Imbad0202/academic-research-skills ~/.claude/academic-research-skills
mkdir -p ~/.claude/commands
cp ~/.claude/academic-research-skills/deep-research/SKILL.md ~/.claude/commands/deep-research.md

# 2. Paper review skill
curl -o ~/.claude/commands/review-paper.md \
  https://raw.githubusercontent.com/claesbackman/AI-research-feedback/main/Paper-review/review-paper.md

# 3. Python packages
pip install --break-system-packages captum umap-learn plotly scikit-learn
```

### Recommended (Install Later)
- chroma or qdrant for vector storage
- sentence-transformers for embeddings
- MINTY toolkit pattern for automated paper feeds

---

## Key Insights from Analysis

### From academic-research-skills:
1. **Integrity Verification Agent** - Anti-hallucination system with 5-phase verification
2. **Socratic SCR Loop** - Collects predictions BEFORE showing evidence
3. **Devil's Advocate (3 checkpoints)** - Challenges interpretability claims
4. **Cross-skill Data Contracts** - Ensures pipeline integrity

### From AI-research-feedback:
1. **6-Agent Parallel Review** - Specialized reviewers for different aspects
2. **Claim Discipline** - Flags causal language without identification
3. **Journal Personas** - Adapts review standards to target venue

### From Philosophy of Computing (MINT Lab):
1. **Paper Feed Automation** - Agents monitor sources and curate
2. **Corpus Building** - Vector stores enable semantic search
3. **Agent Reports** - Quick summaries of fast-developing topics

---

## Files Referenced

- `Imbad0202/academic-research-skills` - Comprehensive research pipeline (906 stars)
- `claesbackman/AI-research-feedback` - Multi-agent paper review
- `DAF-Contribution-Community/daaf` - NOT FOUND (private or misnamed)
- Philosophy of Computing Newsletter - MINT Lab workflow patterns
- `mint-philosophy/minty-agent-toolkit` - Agent infrastructure setup