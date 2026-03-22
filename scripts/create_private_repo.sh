#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "[ERROR] GITHUB_TOKEN is required (needs repo scope)."
  exit 1
fi

if [[ -z "${GH_USER:-}" ]]; then
  echo "[ERROR] GH_USER is required, e.g. GH_USER=yourname"
  exit 1
fi

if [[ -z "${GH_REPO:-}" ]]; then
  echo "[ERROR] GH_REPO is required, e.g. GH_REPO=kdd2017-trafficflow"
  exit 1
fi

API="https://api.github.com/user/repos"
JSON=$(cat <<JSON
{"name":"${GH_REPO}","private":true,"description":"Competition-first, research-grade traffic flow forecasting"}
JSON
)

HTTP_CODE=$(curl -sS -o /tmp/create_repo_resp.json -w "%{http_code}" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "${JSON}" \
  "${API}")

if [[ "${HTTP_CODE}" != "201" && "${HTTP_CODE}" != "422" ]]; then
  echo "[ERROR] GitHub API failed (HTTP ${HTTP_CODE})."
  cat /tmp/create_repo_resp.json
  exit 1
fi

REMOTE_URL="git@github.com:${GH_USER}/${GH_REPO}.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi

git push -u origin main

echo "[OK] Remote configured and pushed: ${REMOTE_URL}"
