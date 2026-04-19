# Code Quality Contract

This project should stay clean enough for both humans and agents to reason about it.

## Rules

### File size
Prefer small files.
If a file becomes hard to read on one screen, split it unless there is a strong reason not to.

### Directory sprawl
Avoid dumping many unrelated files into one directory.
Create subdirectories when there is a meaningful boundary.

### Typing
Prefer explicit typed contracts.
Avoid `any`-style escape hatches unless there is a documented reason.

### Duplication
When logic is duplicated, collapse it.
One source of truth is better than many nearly-the-same versions.

### Interfaces
Use narrow interfaces.
Prefer wrappers and adapters to leaky direct usage.

### Testing
Prefer objective checks over “looks good”.
Tests, validation scripts, and reproducibility checks are part of code quality.

### Refactoring
A meaningful fraction of the work should be deletion, simplification, and compression.
Do not only add code.

## Agent rule
Before large edits, ask:
- can this be simpler?
- can this be smaller?
- can this be split more clearly?
- is there duplicated logic?
- what test proves it works?
