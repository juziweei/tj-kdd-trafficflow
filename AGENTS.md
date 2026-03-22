# AGENTS.md

Repository-wide operating policy for agentic development.

## Mission
Build a competition-first traffic forecasting system with research-grade reproducibility.

## Priority Order
1. Correctness and anti-leakage guarantees.
2. MAPE improvement on time-based validation.
3. Reproducibility (config, seed, run artifacts).
4. Code quality and maintainability.

## Required Workflow
1. Before coding, add a new session entry in `docs/vibe_coding_protocol.md`.
2. Define scope and expected metric impact.
3. Implement minimal working version first.
4. Run at least one time-based validation check.
5. Update the same session with results, risks, and next step.
6. Commit only after the session is updated.

## Hard Constraints
- No future-data leakage in features, labels, or validation.
- No random split for model selection; use rolling/blocked time split.
- Every experiment must have a `run_id`.
- Submission file must pass schema checks.

## Directory Conventions
- `configs/`: all run-time configs
- `src/`: production pipeline code
- `scripts/`: reproducible entrypoints
- `outputs/`: run artifacts, submissions, figures
- `docs/`: protocol, design, experiment notes

## Done Criteria for Any Modeling Task
- Reproducible command documented
- Validation metric reported
- Error slice analysis at least by tollgate-direction or horizon
- Session log updated in `docs/vibe_coding_protocol.md`
