# AI Interpretability Research Session - Complete Context Reference

**Purpose:** This document provides deep context on every tool, skill, and resource installed in this session. Future sessions should read this to understand **what** was installed, **why** it matters, **when** to use it, and **how** it fits into the broader research workflow.

---

## Table of Contents

1. [Research Philosophy & Workflow](#research-philosophy--workflow)
2. [Paper Discovery & Analysis](#paper-discovery--analysis)
3. [Literature Review & Research Pipeline](#literature-review--research-pipeline)
4. [Paper Writing & Review](#paper-writing--review)
5. [Mechanistic Interpretability Tools](#mechanistic-interpretability-tools)
6. [Rigorous Analysis Framework](#rigorous-analysis-framework)
7. [Paper Corpus Management](#paper-corpus-management)
8. [PDF Processing](#pdf-processing)
9. [Prompt Engineering](#prompt-engineering)
10. [Authentication & Access](#authentication--access)
11. [How Everything Fits Together](#how-everything-fits-together)
12. [Quick Reference Card](#quick-reference-card)

---

## Research Philosophy & Workflow

### The Problem This Session Solves

**Before this session:** You had to manually search arXiv, download PDFs, extract text, read papers, take notes, and synthesize findings. This is slow, error-prone, and doesn't scale to staying current with rapid AI research.

**After this session:** You have an integrated pipeline that can:
1. **Discover** papers via semantic search (AlphaXiv) or keyword search (arXiv)
2. **Fetch** papers automatically with PDF extraction
3. **Analyze** papers with structured AI-generated summaries
4. **Organize** a searchable corpus of papers
5. **Conduct** systematic literature reviews with multi-agent pipelines
6. **Write** publication-ready papers with citation management
7. **Execute** interpretability experiments with standardized tools

### The Three-Layer Architecture

```
OUTPUT LAYER: Write & Publish
├── academic-paper (12 agents)
├── academic-paper-reviewer (5+ agents)└── ml-paper-writing (formatting)▲
ANALYSIS LAYER: Understand & Synthesize
├── deep-research (13 agents)
├── interpretability-research (5 phases)
└── DAAF (12 agents)
▲
DISCOVERY LAYER: Find & Fetch papers
├── AlphaXiv (semantic search)
├── arXiv (keyword search)
├── awesome-mechanistic-interpretability (curated list)
├── pdfmux (+OCR)
└── pdftotext (fast)
```

---

## Paper Discovery & Analysis

### AlphaXiv MCP

**What it is:** AlphaXiv is an AI-enhanced arXiv interface that provides structured paper summaries, semantic search, and content retrieval. The MCP (Model Context Protocol) server integrates it directly into OpenCode.

**Why it matters:**
- Raw arXiv papers are PDFs - hard to query programmatically
- AlphaXiv converts papers to structured, queryable format
- Semantic search finds conceptually similar papers, not just keyword matches

**When to use:**
- **Semantic search:** When you want papers about a concept (e.g., "how do transformers learn in-context?")
- **Paper summary:** When you need a quick overview before reading full text
- **Paper comparison:** When comparing multiple papers on similar topics
- **GitHub integration:** When a paper has code and you want to see it

**How it works:**
```
User query: "Find papers about sparse autoencoders"
    ↓
alphaxiv_embedding_similarity_search("Sparse autoencoders for decomposing model activations")
    ↓
Returns: Top papers ranked by semantic similarity + popularity
    ↓
User selects paper → alphaxiv_get_paper_content(url)
    ↓
Returns: Structured summary (background, methods, findings, limitations)
```

**Key Tools:**
| Tool | Purpose |
|------|---------|
| `alphaxiv_embedding_similarity_search(query)` | Semantic/conceptual search |
| `alphaxiv_full_text_papers_search(query)` | Keyword search |
| `alphaxiv_get_paper_content(url)` | Get structured summary |
| `alphaxiv_answer_pdf_queries(urls, queries)` | Ask questions about papers |
| `alphaxiv_read_files_from_github_repository(url, path)` | See paper's code |

**Trigger:** `/alphaxiv-paper-lookup` for quick interface

---

### arXiv Tools

**What it is:** Direct arXiv API integration for searching and downloading papers.

**Why it matters:**
- More precise control over search (categories, date ranges, authors)
- Download papers for local processing
- AlphaXiv may not have every paper; arXiv tools cover everything

**When to use:**
- **Category-specific search:** When you need papers from specific arXiv categories (cs.LG, cs.AI, cs.CL, etc.)
- **Author search:** When tracking specific researchers
- **Date-filtered search:** When you need papers from a specific time period
- **Download for corpus:** When building a local paper collection

**Key Tools:**
| Tool | Purpose |
|------|---------|
| `arxiv_search_papers(query, categories, date_from, date_to)` | Search with filters |
| `arxiv_download_paper("2307.12307")` | Download by ID |
| `arxiv_list_papers()` | List downloaded papers |
| `arxiv_read_paper("2307.12307")` | Read downloaded paper |

**Common Categories:**
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CL` - Computation and Language (NLP)
- `cs.CV` - Computer Vision
- `cs.RO` - Robotics

---

### awesome-mechanistic-interpretability

**What it is:** A curated collection of 637+ papers on mechanistic interpretability, organized by topic.

**URL:** https://github.com/gauravfs-14/awesome-mechanistic-interpretability

**Why it matters:**
- Human-curated quality (not algorithm-ranked)
- Covers the full breadth of the interpretability field
- Includes code repositories, tutorials, benchmarks
- Automatically updated with new arXiv papers
- Includes survey paper: "Bridging the Black Box: A Survey on Mechanistic Interpretability in AI"

**What's included:**
- Sparse Autoencoders (SAE) papers
- Activation Patching & Causal Tracing
- Circuit Discovery & Analysis
- Feature Visualization
- Model Editing & Steering
- Interpretability Benchmarks (MIB, etc.)
- 600+ more papers

**When to use:**
- **Starting a project:** Browse categories to understand the field's structure
- **Finding foundational papers:** The curated list includes classics
- **Staying current:** New papers added automatically from arXiv
- **Finding code:** Linked repositories for key papers

---

## Literature Review & Research Pipeline

### deep-research Skill

**What it is:** A 13-agent pipeline for conducting rigorous academic research on any topic.

**Why it matters:**
- Accounts for different agent biases through specialized roles
- Includes adversarial review (devil's advocate)
- PRISMA-compliant systematic review mode
- APA 7.0 formatted output
- Bilingual support (English + Traditional Chinese)

**The 13 Agents:**
| # | Agent | What it does | Why it matters |
|---|-------|--------------|----------------|
| 1 | research_question_agent | FINER-scored research questions | Prevents unfocused research |
| 2 | research_architect_agent | Methodology blueprint | Ensures scientific rigor |
| 3 | bibliography_agent | Systematic literature search | Comprehensive coverage |
| 4 | source_verification_agent | Fact-checking, CoI flagging | Prevents bad sources |
| 5 | synthesis_agent | Cross-source integration | Deep understanding |
| 6 | report_compiler_agent | APA 7.0 report drafting | Professional output |
| 7 | editor_in_chief_agent | Q1 journal editorial review | Quality control |
| 8 | devils_advocate_agent | Challenges assumptions | Prevents confirmation bias |
| 9 | ethics_review_agent | AI research ethics | Ethical rigor |
| 10 | socratic_mentor_agent | Guides thinking | Helps novice researchers |
| 11 | risk_of_bias_agent | RoB 2 / ROBINS-I assessment | Systematic reviews |
| 12 | meta_analysis_agent | Meta-analysis | Quantitative synthesis |
| 13 | monitoring_agent | Literature alerts | Stay current |

**Modes:**
| Mode | Use Case |
|------|----------|
| `full` | Complete research pipeline |
| `quick` | 30-minute brief |
| `review` | Evaluate a specific paper |
| `lit-review` | Literature review only |
| `fact-check` | Verify specific claims |
| `socratic` | Guided thinking (unclear topics) |
| `systematic-review` | PRISMA-compliant systematic review |

**Trigger:** `/deep-research`

---

### MINT Lab Workflow

**What it is:** A research infrastructure philosophy from the MINT Lab (ANU) for managing paper corpora.

**Source:** philosophyofcomputing.substack.com

**Why it matters:**
- Solves "InfoGlut" - too many papers, too little time
- Agent reads papers for you, surfaces what's relevant
- Creates a live, evolving corpus (NotebookLM-like but unlimited)
- Semantic search over entire corpus

**The Pipeline:**
```
PAPER FEED → PDF FETCH → CORPUS → REPORTS → CONVERSATION

┌─────────────────┐
│ PAPER FEED      │ arXiv subs, Twitter lists, RSS feeds
└────────┬────────┘▼
┌─────────────────┐
│ PDF FETCH       │ Agent tracks down PDFs (~100% success)
└────────┬────────┘▼
┌─────────────────┐
│ CORPUS BUILDING │ Markdown + embeddings + 41-question analysis
└────────┬────────┘▼
┌─────────────────┐
│ AGENTIC SEARCH  │ Query naturally: "What discusses superposition?"
└────────┬────────┘▼
┌─────────────────┐
│ REPORTS         │ Daily/weekly digests, deep dives
└─────────────────┘
```

**Key Insight:** "I'm on social media because I'm having a conversation; not because I'm frantically trying to follow the latest research—because Minty does that for me."

---

## Paper Writing & Review

### academic-paper Skill

**What it is:** A 12-agent pipeline for writing publication-ready academic papers.

**Why it matters:**
- Structure-first approach (ensures logical flow)
- Citation management (APA7, Chicago, MLA, IEEE, Vancouver)
- Peer review built into the workflow
- Multi-format output (LaTeX, DOCX, PDF, Markdown)
- Bilingual abstract support (zh-TW + EN)

**The 12 Stages:**
1. Configuration Interview - Paper type, discipline, citation format
2. Literature Search - Systematic search strategy
3. Architecture Design - Paper structure, word count allocation
4. Argumentation Construction - Claim-evidence chains, logical flow
5. Full-Text Drafting - Section-by-section drafting
6. Citation Compliance - Format checking, bilingual abstract
7. Peer Review - Five-dimension scoring
8. Output Formatting - LaTeX/DOCX/PDF/Markdown

**Supported Structures:**
- IMRaD (Introduction, Methods, Results, Discussion)
- Literature review
- Theoretical paper
- Case study
- Policy brief
- Conference paper

**Trigger:** `/academic-paper`

---

### academic-paper-reviewer Skill

**What it is:** Multi-perspective peer review simulation with 5 reviewers.

**Why it matters:**
- Catches issues you'd miss in self-review
- Different reviewer personas bring different expertise
- Devil's advocate finds weaknesses
- Constructive feedback, not just criticism

**The 5 Reviewers:**
| Reviewer | Focus |
|----------|-------|
| Editor-in-Chief | Overall quality, novelty, fit |
| Peer Reviewer 1 | Methodology |
| Peer Reviewer 2 | Domain expertise |
| Peer Reviewer 3 | General quality |
| Devil's Advocate | Challenges assumptions |

**Modes:**
| Mode | Use Case |
|------|----------|
| `full` | Complete review |
| `re-review` | Verification after revision |
| `quick` | Rapid assessment |
| `methodology` | Methods focus |
| `socratic` | Guided review dialogue |

**Trigger:** `/academic-paper-reviewer`

---

## Mechanistic Interpretability Tools

### interpretability-research Skill

**What it is:** A complete 5-phase workflow for mechanistic interpretability research.

**Why it matters:**
- Integrates all interpretability tools in one workflow
- Covers methodology design specific to interpretability
- Includes implementation guidance for TransformerLens, SAEs
- End-to-end: idea → paper

**The 5 Phases:**
| Phase | Focus | Activities |
|-------|-------|------------|
| 1. Literature Review | Understand field | Systematic search, synthesis, gap identification |
| 2. Methodology Design | Plan experiment | Research questions, hypotheses, method selection |
| 3. Implementation | Run analysis | TransformerLens/NNsight/SAE integration |
| 4. Analysis | Interpret results | Statistics, effect sizes, visualizations |
| 5. Write-up | Publish | Paper formatting, citations |

**Trigger:** `/interpretability-research`

---

### TransformerLens

**What it is:** Library for mechanistic interpretability of transformers via HookPoints.

**Why it matters:**
- Core tool for circuit analysis
- Access any activation in the model
- Patch activations to test causal hypotheses
- Works with GPT-2, LLaMA, etc.

**What you can do:**
- **Activation caching:** Store activations during forward pass
- **Activation patching:** Replace activations to test causal effects
- **Head analysis:** Study individual attention heads
- **Circuit discovery:** Find subgraphs implementing behaviors

**Skill:** `/transformer-lens-interpretability`

**When to use:**
- Circuit analysis (IOI, induction heads)
- Attention head importance
- Layer-wise decomposition
- Testing mechanistic hypotheses

---

### NNsight

**What it is:** Remote interpretability on large models (70B+) via NDIF.

**Why it matters:**
- Run interpretability on large models without local GPU
- Access to model internals remotely
- Results cached for reproducibility

**When to use:**
- Interpreting large models (70B+)
- No local GPU access
- Collaborative experiments

**Skill:** `/nnsight-remote-interpretability`

---

### PyVene

**What it is:** Declarative intervention framework for causal experiments.

**Why it matters:**
- Specify interventions declaratively
- Composable interventions
- Supports interchange intervention training

**What you can do:**
- Activation patching
- Causal tracing
- Interchange interventions
- Training optimal interventions

**Skill:** `/pyvene-interventions`

---

### SAE Lens

**What it is:** Library for training and analyzing Sparse Autoencoders.

**Why it matters:**
- Decompose activations into interpretable features
- Study superposition
- Monosemantic features vs polysemantic neurons

**What you can do:**
- Train SAEs on any model's activations
- Feature analysis (what each feature encodes)
- Feature visualization
- Steering model behavior

**Skill:** `/sparse-autoencoder-training`

---

### captum & umap-learn

**captum** (installing):
- PyTorch's interpretability library from Meta
- Integrated Gradients, GradCAM, DeepLIFT, Feature Ablation
- Standardized attribution methods

**umap-learn** (installing):
- Dimensionality reduction for visualization
- Visualize activation clusters
- See feature manifolds
- Compare representations

**When to use:**
- Activation visualization
- Feature space exploration
- Debugging training dynamics

---

## Rigorous Analysis Framework

### DAAF Framework

**What it is:** Data Analyst Augmentation Framework - rigorous, transparent, reproducible data analysis.

**Location:** `~/.claude/daaf/`

**Why it matters:**
- **Transparent:** All operations saved as files with changelog
- **Scalable:** Structured expertise injection via skills
- **Rigorous:** Adversarial code review, self-checking
- **Reproducible:** Every step documented and runnable

**Philosophy:**
> "LLM research assistants will never be perfect and can never be trusted as a matter of course. But with strict guardrails and auditability, they can be immensely valuable for critically-minded researchers."

**12 Agents:**
| Agent | Role |
|-------|------|
| `code-reviewer` | Adversarial code inspection |
| `data-ingest` | Profile and integrate data |
| `data-planner` | Plan analysis workflows |
| `data-verifier` | Verify data integrity |
| `debugger` | Debug analysis issues |
| `integration-checker` | Integration testing |
| `notebook-assembler` | Create reproducible notebooks |
| `plan-checker` | Verify research plans |
| `report-writer` | Generate reports |
| `research-executor` | Execute research plans |
| `research-synthesizer` | Synthesize findings |
| `source-researcher` | Research data sources |

**Quality Hooks:**
- `bash-safety.sh` - Blocks dangerous commands
- `audit-log.sh` - Tracks all operations
- `output-scanner.sh` - Scans outputs for issues

---

## Paper Corpus Management

### Minty Agent Toolkit

**What it is:** Complete implementation reference from MINT Lab for paper corpus systems.

**Location:** `~/.claude/minty-agent-toolkit/`

**What's included:**
- `claude-config/` - Configuration files
- `daemon-framework/` - Background daemon patterns
- `skills/` - Research skills
- `SETUP_GUIDE.md` - 337KB replication manual

**Key Patterns:**
1. Agent as colleague (not just tool)
2. Action over reporting (solve, don't describe)
3. Proactive management (anticipate issues)
4. Iron Laws (non-negotiable constraints)
5. Delegation first (use subagents)

---

## PDF Processing

### pdfmux

**What it is:** PDF extraction with OCR fallback and confidence scoring.

**Location:** `~/.local/bin/pdfmux`

**When to use:**
- Scanned PDFs (not born-digital)
- Mixed text/image PDFs
- When you need confidence scores

**Example:**
```bash
pdfmux extract paper.pdf -o paper.txt --ocr-fallback
```

---

### pdftotext

**What it is:** Fast PDF text extraction (5.2MB portable binary).

**Location:** `~/.local/bin/pdftotext`

**When to use:**
- Born-digital PDFs
- Quick text extraction
- When speed matters

**Example:**
```bash
pdftotext paper.pdf paper.txt
```

---

## Prompt Engineering

### JFP CLI (Jeffrey's Prompts)

**What it is:** 21 free prompts for agentic coding workflows.

**Location:** `~/.local/bin/jfp`

**Installed as Skills:**

| Skill | Purpose |
|-------|---------|
| `/jeffrey-prompts/idea-wizard` | Generate 30 ideas → evaluate → distill to 5 best |
| `/jeffrey-prompts/readme-reviser` | Update documentation for recent changes |

---

## Authentication & Access

### Anthropic OAuth

**Location:** `~/.local/share/opencode/auth.json`

**Status:** ✓ Synced and working

**Token expiry:** ~8 hours from session start

---

### AlphaXiv MCP

**Location:** Configured in `~/.config/opencode/opencode.json`

**Status:** ✓ Enabled

**OAuth Issue:** WSL callback URL may need manual fix (replace `127.0.0.1` with WSL IP)

---

## How Everything Fits Together

### Workflow 1: New Interpretability Project

```
1. /interpretability-research → Start 5-phase workflow

2. Phase 1: Literature Review
   → alphaxiv_search + arxiv_search + deep-research

3. Phase 2: Methodology Design
   → Design experiments using TransformerLens/NNsight/SAE Lens

4. Phase 3: Implementation
   → Run analysis using skill-specific guidance
   → captum for attributions, umap-learn for visualization

5. Phase 4: Analysis
   → Interpret results, statistical testing

6. Phase 5: Write-up
   → /academic-paper for publication formatting
   → /academic-paper-reviewer for peer review
```

### Workflow 2: Quick Paper Analysis

```
1. Find paper
   → alphaxiv_embedding_similarity_search("topic")
   → OR arxiv_search_papers("keywords", categories=["cs.LG"])

2. Get summary
   → alphaxiv_get_paper_content(url)

3. Ask questions
   → alphaxiv_answer_pdf_queries([url], ["Main contribution?", "Limitations?"])

4. See code (if available)
   → alphaxiv_read_files_from_github_repository(url, "/")

5. Full text (if needed)
   → arxiv_download_paper("2307.12307")
   → arxiv_read_paper("2307.12307")
```

### Workflow 3: Stay Current with Research

```
1. Paper Feed → arxiv_search_papers by category with date_from

2. PDF Fetch → arxiv_download_paper for each new paper

3. Quick Analysis → alphaxiv_get_paper_content for summary

4. Weekly Digest → /deep-research in quick mode

5. Deep Dive → /deep-research full mode for comprehensive review
```

---

## Quick Reference Card

| I want to... | Use this |
|--------------|----------|
| Find papers by concept | `alphaxiv_embedding_similarity_search(query)` |
| Find papers by keyword | `arxiv_search_papers(query, categories)` |
| Get paper summary | `alphaxiv_get_paper_content(url)` |
| Ask questions about paper | `alphaxiv_answer_pdf_queries([url], questions)` |
| Download paper | `arxiv_download_paper("2307.12307")` |
| Conduct literature review | `/deep-research` |
| Write a paper | `/academic-paper` |
| Review a paper | `/academic-paper-reviewer` |
| Start interpretability project | `/interpretability-research` |
| Analyze circuits | `/transformer-lens-interpretability` |
| Train SAEs | `/sparse-autoencoder-training` |
| Visualize activations | captum + umap-learn |
| Rigorous data analysis | Use DAAF framework |
| Build paper corpus | Use Minty toolkit patterns |

---

## File Locations

| Component | Location |
|-----------|----------|
| pdfmux | `~/.local/bin/pdfmux` |
| pdftotext | `~/.local/bin/pdftotext` |
| JFP CLI | `~/.local/bin/jfp` |
| Anthropic auth | `~/.local/share/opencode/auth.json` |
| OpenCode config | `~/.config/opencode/opencode.json` |
| Skills | `~/.config/opencode/skills/` |
| DAAF | `~/.claude/daaf/` |
| Minty toolkit | `~/.claude/minty-agent-toolkit/` |
| academic-research-skills | `~/.claude/academic-research-skills/` |

---

**End of Context Reference**

*Read this document to understand every tool's purpose, when to use it, and how it fits together.*
---

## Context Engineering Skills

### Agent Skills for Context Engineering

**Source:** https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering

**What it is:** A comprehensive collection of13 skills for context engineering - the discipline of managing the language model's context window effectively.

**Why it matters:**
- Context windows are constrained by attention mechanics, not just token capacity
- Models exhibit predictable degradation: "lost-in-the-middle", U-shaped attention curves
- Effective context engineering = smallest high-signal tokens for maximum outcomes
- Cited in academic research: "Meta Context Engineering via Agentic Skill Evolution" (Peking University, 2026)

---

### The 13 Skills

#### Foundational Skills

**context-fundamentals**
- **What:** Understanding context anatomy, why it matters, and how it works in agent systems
- **When:** Starting any agent project, debugging context issues
- **Key Concepts:** Context window, attention budget, signal-to-noise ratio

**context-degradation**
- **What:** Recognizing patterns of context failure
- **When:** Debugging agent failures, diagnosing performance issues
- **Key Patterns:**
  - Lost-in-middle: Information in center gets less attention
  - Context poisoning: Errors compound
  - Context distraction: Irrelevant info overwhelms relevant
  - Context clash: Conflicting instructions

**context-compression**
- **What:** Designing compression strategies for long-running sessions
- **When:** Sessions exhausting memory, need to reduce token usage
- **Key Insight:** Optimize tokens-per-task, not tokens-per-request

---

#### Architectural Skills

**multi-agent-patterns**
- **What:** Master orchestrator, peer-to-peer, and hierarchical multi-agent architectures
- **When:** Building multi-agent systems, designing supervisor patterns
- **Key Patterns:**
  - Supervisor/orchestrator: Centralized control
  - Peer-to-peer swarm: Flexible handoffs
  - Hierarchical: Complex task decomposition
- **Critical Insight:** Sub-agents exist to isolate context, not simulate org roles

**memory-systems**
- **What:** Designing short-term, long-term, and graph-based memory architectures
- **When:** Implementing agent memory, building knowledge graphs, tracking entities
- **Key Patterns:**
  - Vector RAG: Semantic retrieval, loses relationships
  - Knowledge graphs: Preserves structure, more engineering
  - File-system-as-memory: Just-in-time context loading

**tool-design**
- **What:** Building tools that agents can use effectively
- **When:** Designing agent tools, implementing MCP tools, reducing tool complexity
- **Key Principles:**
  - Consolidation: Single comprehensive > multiple narrow tools
  - Return contextual info in errors
  - Support response format options for token efficiency
  - Clear namespacing

**filesystem-context**
- **What:** Using filesystems for dynamic context discovery, tool output offloading, plan persistence
- **When:** Offloading context to files, dynamic context discovery, agent scratchpad
- **Key Patterns:**
  - Scratch pads for tool output offloading
  - Plan persistence for long-horizon tasks
  - Sub-agent communication via shared files
  - Dynamic skill loading

**hosted-agents**
- **What:** Building background coding agents with sandboxed VMs
- **When:** Creating hosted coding agents, sandboxed execution, multiplayer agents
- **Key Patterns:**
  - Pre-built environment images
  - Warm sandbox pools for instant starts
  - Filesystem snapshots for session persistence
  - Multiplayer support for collaborative sessions

---

#### Operational Skills

**context-optimization**
- **What:** Applying compaction, masking, and caching strategies
- **When:** Optimizing context, reducing token costs, implementing KV-cache
- **Key Techniques:**
  - Compaction: Summarize context near limits
  - Observation masking: Replace verbose outputs with references
  - Prefix caching: Reuse KV blocks across requests
  - Strategic context partitioning: Split work across sub-agents

**evaluation**
- **What:** Building evaluation frameworks for agent systems
- **When:** Evaluating agent performance, building test frameworks, measuring quality
- **Key Patterns:**
  - LLM-as-judge for scalability
  - Human evaluation for edge cases
  - End-state evaluation for persistent state

**advanced-evaluation**
- **What:** Mastering LLM-as-a-Judge techniques
- **When:** Implementing LLM-as-judge, comparing model outputs, mitigating bias
- **Key Techniques:**
  - Direct scoring with weighted criteria
  - Pairwise comparison with position bias mitigation
  - Rubric generation
  - Bias mitigation strategies

---

#### Development Methodology

**project-development**
- **What:** Designing and building LLM projects from ideation through deployment
- **When:** Starting LLM project, designing batch pipeline, evaluating task-model fit
- **Key Concepts:**
  - Task-model fit analysis: Validate through manual prototyping
  - Staged, idempotent architectures: acquire → prepare → process → parse → render
  - File system state management for debugging and caching
  - Structured output design with explicit format specifications

---

#### Cognitive Architecture

**bdi-mental-states**
- **What:** Transform external RDF context into agent mental states (beliefs, desires, intentions)
- **When:** Building cognitive agents, implementing BDI architecture, explainability
- **Key Concepts:**
  - Beliefs: Agent's knowledge about the world
  - Desires: Agent's goals and objectives
  - Intentions: Agent's committed plans
  - BDI ontology patterns for deliberative reasoning

---

### Using These Skills

**Skill Triggers:**

| Skill | Triggers On |
|-------|-------------|
| `context-fundamentals` | "understand context", "explain context windows", "design agent architecture" |
| `context-degradation` | "diagnose context problems", "fix lost-in-middle", "debug agent failures" |
| `context-compression` | "compress context", "summarize conversation", "reduce token usage" |
| `context-optimization` | "optimize context", "reduce token costs", "implement KV-cache" |
| `multi-agent-patterns` | "design multi-agent system", "implement supervisor pattern" |
| `memory-systems` | "implement agent memory", "build knowledge graph", "track entities" |
| `tool-design` | "design agent tools", "reduce tool complexity", "implement MCP tools" |
| `filesystem-context` | "offload context to files", "dynamic context discovery", "agent scratch pad" |
| `hosted-agents` | "build background agent", "create hosted coding agent", "sandboxed execution" |
| `evaluation` | "evaluate agent performance", "build test framework", "measure quality" |
| `advanced-evaluation` | "implement LLM-as-judge", "compare model outputs", "mitigate bias" |
| `project-development` | "start LLM project", "design batch pipeline", "evaluate task-model fit" |
| `bdi-mental-states` | "model agent mental states", "implement BDI architecture", "build cognitive agent" |

**Location:** `~/.config/opencode/skills/context-engineering-collection/` (main) and individual skills in their own directories

---

## Repository Locations Summary

| Repository | Location | Purpose |
|------------|----------|---------|
| academic-research-skills | `~/.claude/academic-research-skills/` | 32-agent research pipeline |
| DAAF | `~/.claude/daaf/` | Rigorous data analysis framework |
| Minty Agent Toolkit | `~/.claude/minty-agent-toolkit/` | Paper corpus management |
| Agent Skills for Context Engineering | `~/.claude/agent-skills-context-eng/` | Context engineering skills |

---

**End of Context Reference**

*Last updated: March 22, 2026*
