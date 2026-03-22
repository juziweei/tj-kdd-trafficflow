---
name: "vibe-governance"
description: "Enforce session logging, anti-leakage checks, and reproducible experiment reporting for this repository."
---

# Vibe Governance Skill

## When to use
- Any task that changes data pipeline, features, models, evaluation, or submission.
- Any task that adds/changes experiment scripts or configs.

## Mandatory steps
1. Create or update a session entry in `docs/vibe_coding_protocol.md` before edits.
2. State scope, expected metric impact, and validation plan.
3. Run at least one time-based validation check if model/data logic changed.
4. Record run id, metric, and key findings in the same session entry.
5. Mark session status as `Done` or `Blocked` with clear next step.

## Guardrails
- Reject random-split evaluation for model selection.
- Reject any feature using future timestamps.
- Reject code changes without protocol log updates.

## Required output in final response
- Files changed
- Validation performed
- Updated session id in protocol log
- Next immediate action
