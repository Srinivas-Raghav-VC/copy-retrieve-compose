# v3 Research Setup

This repository is configured as a small research operating system for OpenCode + Oh My OpenCode.

## Where files go

Keep these files in the project root:
- `AGENTS.md`
- `opencode.json`
- `README_SETUP.md`

Keep these under `research/`:
- project brief
- verification rules
- paper/tool policy
- examples
- journal
- retrospective
- planner prompt
- task template
- module-local guidance
- agent prompt files

Keep Oh My OpenCode project config under:
- `.opencode/oh-my-opencode.json`

Keep project-local skills under:
- `.opencode/skills/<skill-name>/SKILL.md`

## Recommended workflow

### 1. Planning
Use `research-plan` or `sisyphus` in planning mode first.
Paste `research/FINAL_PLANNER_PROMPT.xml` or ask it to read that file.
Goal:
- interview
- frame
- build `spec.md`
- critique
- revise
- stop

### 2. Build
Switch to `sisyphus` or `build`.
Execute one bounded phase at a time.
After each phase:
- run objective checks
- update the journal
- write a retrospective note
- continue only if success criteria are met

### 3. Subagents
Use:
- `@repo-scout` for fast repo scanning
- `@literature-scout` for papers/docs
- `@skeptic-reviewer` for challenge
- `@refactor-police` for code cleanup audits
- `@figure-critic` for paper visual planning

## What changed in v3

v3 adds:
- explicit model-routing guidance
- skill-based reuse
- module-local guidance
- examples and retrospectives as first-class project memory
- stronger code-quality contract
- cleaner separation between planning, execution, and review

## Important note

The config intentionally avoids hardcoding fragile provider-specific model IDs for every subagent.
If you want to pin models per agent later, read `research/MODEL_ROUTING.md` and update `opencode.json`.
