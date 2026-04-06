---
name: "git-clean-publish"
description: "Publish code/docs to Git while excluding tests and result artifacts (outputs), with repeatable checks."
---

# Git Clean Publish Skill

## When to use
- User asks to "upload to git" / "push" with repository cleanliness requirements.
- Team policy requires excluding test files and experiment result artifacts.
- Before releasing public-facing updates that should contain only source/config/docs.

## Required target policy
- Include: source code, configs, docs, scripts (non-test), governance files.
- Exclude: `tests/`, `scripts/test_*.py`, `scripts/test_*.sh`, `outputs/**` artifacts.

## Mandatory steps
1. Confirm `.gitignore` has required patterns:
   - `tests/`
   - `scripts/test_*.py`
   - `scripts/test_*.sh`
   - `outputs/*` (with only allowed placeholders if explicitly needed)
2. Run preflight checks:
   - `git status --short --branch`
   - `git check-ignore -v tests/test_bootstrap.py outputs/submissions/example.csv scripts/test_nash.py`
   - `git ls-files tests 'outputs/**'`
3. If tracked forbidden files exist, untrack without deleting local content:
   - `git rm --cached <path>`
4. Stage and verify:
   - `git add -A`
   - `git diff --cached --name-status | rg 'tests/|outputs/|scripts/test_'`
   - Accept only deletions for forbidden tracked files.
5. Commit with clear scope:
   - `git commit -m "<type>: <summary>"`
6. Push and verify:
   - `git push origin <branch>`
   - `git status --short --branch` should be clean/synced.

## Recommended commit message templates
- `chore: publish docs and scripts with clean git policy`
- `chore: sync project updates and enforce no-test-no-output upload rule`

## Failure handling
- If `git add/commit` fails due sandbox index lock, rerun with elevated permission.
- If push fails due proxy/network, retry in direct network mode (`env -u ALL_PROXY ...`).
- If unexpected tracked result files reappear, rerun step 2 and clean via `git rm --cached`.

## Expected final report
- Commit hash and branch.
- Pushed remote ref.
- Exact files included.
- Confirmation that tests/results are excluded by policy checks.
