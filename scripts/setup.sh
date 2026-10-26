#!/usr/bin/env bash
# One-command setup for RAGAdvanced. Safe to re-run: existing .env values are kept.
#
#   scripts/setup.sh               full setup: venv, both .env files, LiteLLM gateway, virtual key
#   scripts/setup.sh --no-gateway  offline setup: venv + .env only (local minilm embedder, no answers)
#
# Non-interactive: export VERTEX_PROJECT (and optionally VERTEX_LOCATION) before running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY="$ROOT/infra/litellm"
COMPOSE=(docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml)
WITH_GATEWAY=1
[[ "${1:-}" == "--no-gateway" ]] && WITH_GATEWAY=0

step() { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
fail() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "'$1' not found. $2"; }

# Read KEY's value from an env file ("" if missing or commented out).
env_get() { sed -n "s/^$2=//p" "$1" | tail -n 1; }
# Set KEY=VALUE in an env file, replacing an existing or commented-out line, else appending.
env_set() {
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
path, key, value = sys.argv[1:4]
text = open(path).read()
pattern = re.compile(rf"^#?\s*{re.escape(key)}=.*$", re.M)
line = f"{key}={value}"
text = pattern.sub(lambda _: line, text, count=1) if pattern.search(text) else text.rstrip("\n") + f"\n{line}\n"
open(path, "w").write(text)
PY
}

need uv "Install: https://docs.astral.sh/uv/"
need python3 "Install Python 3."

step "Python environment (.venv, Python 3.13)"
cd "$ROOT"
[[ -d .venv ]] || uv venv --python 3.13 .venv
uv pip install --quiet --python .venv/bin/python -r requirements.txt
echo "  dependencies installed"

step "App settings (.env)"
if [[ -f .env ]]; then echo "  keeping existing .env"; else cp .env.example .env; echo "  created .env from .env.example"; fi

if [[ $WITH_GATEWAY -eq 0 ]]; then
  step "Done (offline mode)"
  echo "  Try: .venv/bin/python demo/ingest_all.py && .venv/bin/python demo/compare.py \"How long is an access token valid?\""
  exit 0
fi

need docker "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
need openssl "Install OpenSSL."
need curl "Install curl."

step "Gateway settings (infra/litellm/.env)"
umask 077
if [[ ! -f "$GATEWAY/.env" ]]; then
  cp "$GATEWAY/.env.example" "$GATEWAY/.env"
  echo "  created infra/litellm/.env from .env.example"
fi
for key in LITELLM_MASTER_KEY LITELLM_SALT_KEY; do
  if [[ -z "$(env_get "$GATEWAY/.env" "$key")" ]]; then
    env_set "$GATEWAY/.env" "$key" "sk-$(openssl rand -hex 32)"
    echo "  generated $key"
  fi
done
if [[ -z "$(env_get "$GATEWAY/.env" VERTEX_PROJECT)" ]]; then
  project="${VERTEX_PROJECT:-}"
  if [[ -z "$project" ]]; then
    [[ -t 0 ]] || fail "VERTEX_PROJECT is not set. Export it or run interactively."
    read -rp "  GCP project ID (Vertex AI API enabled): " project
  fi
  [[ -n "$project" ]] || fail "A GCP project ID is required for the gateway (or use --no-gateway)."
  env_set "$GATEWAY/.env" VERTEX_PROJECT "$project"
fi
if [[ -n "${VERTEX_LOCATION:-}" ]]; then env_set "$GATEWAY/.env" VERTEX_LOCATION "$VERTEX_LOCATION"; fi
echo "  project: $(env_get "$GATEWAY/.env" VERTEX_PROJECT), region: $(env_get "$GATEWAY/.env" VERTEX_LOCATION)"

adc="$(env_get "$GATEWAY/.env" GCLOUD_ADC_PATH)"
adc="${adc:-$HOME/.config/gcloud/application_default_credentials.json}"
[[ -f "$adc" ]] || fail "No gcloud credentials at $adc. Run:
    gcloud auth application-default login
    gcloud auth application-default set-quota-project $(env_get "$GATEWAY/.env" VERTEX_PROJECT)"

step "Start LiteLLM gateway (Docker)"
docker info >/dev/null 2>&1 || fail "Docker is not running. Start Docker Desktop and re-run."
(cd "$GATEWAY" && "${COMPOSE[@]}" up -d)
port="$(env_get "$GATEWAY/.env" LITELLM_PORT)"; port="${port:-4000}"
for _ in $(seq 60); do
  curl -sf "http://127.0.0.1:$port/health/liveliness" >/dev/null && break
  sleep 2
done
curl -sf "http://127.0.0.1:$port/health/liveliness" >/dev/null \
  || fail "Gateway did not become healthy. Logs: (cd infra/litellm && ${COMPOSE[*]} logs litellm)"
echo "  gateway healthy on http://localhost:$port"

step "App key (scoped LiteLLM virtual key)"
[[ -n "$(env_get .env LITELLM_BASE_URL)" ]] || env_set .env LITELLM_BASE_URL "http://localhost:$port"
if [[ -n "$(env_get .env LITELLM_API_KEY)" ]]; then
  echo "  keeping existing LITELLM_API_KEY"
else
  models="$(sed -nE 's/^[[:space:]]*-[[:space:]]*model_name:[[:space:]]*([^[:space:]#]+).*/\1/p' "$GATEWAY/config/config.yaml")"
  models_json="$(printf '%s\n' $models | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().split()))')"
  master="$(env_get "$GATEWAY/.env" LITELLM_MASTER_KEY)"
  key="$(curl -sf -H "Authorization: Bearer $master" -H "Content-Type: application/json" \
           "http://127.0.0.1:$port/key/generate" \
           -d "{\"key_alias\":\"rag-advanced-local-$(date +%s)\",\"models\":$models_json}" \
         | python3 -c 'import json,sys; print(json.load(sys.stdin)["key"])')" \
    || fail "Could not mint a virtual key. Check the gateway logs."
  env_set .env LITELLM_API_KEY "$key"
  echo "  minted virtual key for: $models"
fi

step "Smoke test (embedding via gateway)"
.venv/bin/python - <<'PY'
from vectorstore.embedder import get_embedder
print("  text-embedding-005 dims:", len(get_embedder("text-embedding-005").embed_query("hello")))
PY

step "Done"
cat <<EOF
  Ingest and compare:  .venv/bin/python demo/ingest_all.py --embedders minilm text-embedding-005
                       .venv/bin/python demo/compare.py "How long is an access token valid?" --embedder text-embedding-005
  Agent:               .venv/bin/adk web agent   → http://localhost:8000, pick "rag_agent"
EOF
