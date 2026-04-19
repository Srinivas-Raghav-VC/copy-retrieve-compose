# Model Routing

## Principle
Use the best model for the job, not one model for everything.

## Recommended routing for this repo

### Strong planner / final writer
Use your strongest high-reliability planning/writing model when:
- framing the paper thesis
- building or revising spec.md
- deciding what is publishable
- writing claims that need precision
- merging conflicting evidence

In your current workflow this is typically Claude Opus while quota lasts.

### Fast scout / scanner
Use a fast, cheaper, high-throughput model when:
- scanning the repo
- finding likely files
- summarizing error clusters
- triaging TypeScript / lint / config problems
- generating quick architecture inventories
- performing broad searches that will later be checked by a stronger model

In your current workflow this is a good place to use MiniMax M2.5 or M2.5-highspeed if available.

### Deep execution / long-horizon worker
Use a long-horizon model when:
- doing deep multi-step implementation
- running large refactors
- handling big migration-like tasks
- executing long plans with minimal hand-holding
- working after strong planner quota is exhausted

In your current workflow this is a good place for GLM-5 if it is connected and behaving well.

## Routing rules
- Do not let a fast scout make final scientific claims.
- Do not waste the strongest model on low-value file listing.
- Do not let a long-horizon executor define the paper thesis alone.
- When a subagent returns something important, have the main agent verify it.

## Practical use
- Use `@repo-scout` and `@literature-scout` for breadth.
- Use the primary planner for the spec and publication framing.
- Use build or sisyphus for the implementation loop.
