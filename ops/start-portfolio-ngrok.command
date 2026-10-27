#!/bin/bash
# One command: Clinic demo + HEMS + local API proxy + one public ngrok tunnel.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/portfolio-ngrok.json"
CHECK_ONLY=0
LOCAL_ONLY=0
START_BACKENDS=1
PUBLIC_URL_OVERRIDE=""
PROXY_PID=""
NGROK_PID=""
AWAKE_PID=""
OWN_LOCK=0

usage() {
  cat <<'HELP'
Usage: bash start-portfolio-ngrok.command [options]

Default: start existing Clinic demo and HEMS images, proxy, then one ngrok tunnel.
  --check              Validate settings and check health; start nothing.
  --local-only         Start backends and proxy without publishing an ngrok URL.
  --no-start-backends  Use backends that are already healthy.
  --url HTTPS_URL      Explicitly use your account's assigned ngrok URL.
  --config FILE        Read another portfolio-ngrok.json configuration.
  -h, --help           Show this help.

Ctrl+C stops this launcher's tunnel and proxy. Backend containers keep running.
Settings: portfolio-ngrok.json. Instructions: ops/README.md.
HELP
}
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value."; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --local-only) LOCAL_ONLY=1; shift ;;
    --no-start-backends) START_BACKENDS=0; shift ;;
    --url) need_value "$@"; PUBLIC_URL_OVERRIDE="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1. Use --help." ;;
  esac
done

NODE_BIN="${PORTFOLIO_NODE_BIN:-}"
if [[ -z "$NODE_BIN" ]]; then
  for candidate in "$(command -v node || true)" /opt/homebrew/bin/node /usr/local/bin/node "$HOME"/.nvm/versions/node/*/bin/node; do
    if [[ -x "$candidate" ]] && "$candidate" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)' >/dev/null 2>&1; then
      NODE_BIN="$candidate"
      break
    fi
  done
fi
[[ -n "$NODE_BIN" && -x "$NODE_BIN" ]] || fail 'Node.js 18+ is required. Install a supported Node.js release or set PORTFOLIO_NODE_BIN.'
"$NODE_BIN" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 18 ? 0 : 1)' || fail 'Use Node.js 18 or newer.'
for executable in curl python3; do
  command -v "$executable" >/dev/null 2>&1 || fail "$executable is missing."
done
[[ -f "$SCRIPT_DIR/portfolio-proxy.cjs" ]] || fail 'Keep portfolio-proxy.cjs beside this launcher.'
"$NODE_BIN" -e 'require(process.argv[1]).loadConfig(process.argv[2]);' "$SCRIPT_DIR/portfolio-proxy.cjs" "$CONFIG" || fail 'Invalid configuration. See the error above.'

VALUES=()
while IFS= read -r value; do VALUES+=("$value"); done < <(
  "$NODE_BIN" -e 'const c=require(process.argv[1]).loadConfig(process.argv[2]); for(const k of ["clinicDemoDir","hemsProjectDir","proxyPort","clinicPort","hemsPort","publicUrl"]) console.log(c[k]);' "$SCRIPT_DIR/portfolio-proxy.cjs" "$CONFIG"
)
CLINIC_DIR="${VALUES[0]}"
HEMS_DIR="${VALUES[1]}"
PROXY_PORT="${VALUES[2]}"
CLINIC_PORT="${VALUES[3]}"
HEMS_PORT="${VALUES[4]}"
PUBLIC_URL="${PUBLIC_URL_OVERRIDE:-${VALUES[5]}}"
if [[ -n "$PUBLIC_URL" ]]; then
  [[ "$PUBLIC_URL" =~ ^https://[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z][A-Za-z0-9-]*/?$ ]] || fail 'Use an HTTPS hostname without path, credentials or port for --url.'
  PUBLIC_URL="${PUBLIC_URL%/}"
fi
[[ -f "$CLINIC_DIR/compose.yaml" ]] || fail "Missing demo compose.yaml in $CLINIC_DIR."
[[ -f "$HEMS_DIR/docker-compose.yml" ]] || fail "Missing docker-compose.yml in $HEMS_DIR."
if [[ "$LOCAL_ONLY" -eq 0 || "$CHECK_ONLY" -eq 1 ]]; then
  command -v ngrok >/dev/null 2>&1 || fail 'Install ngrok first: brew install ngrok'
  ngrok config check || fail 'Set your ngrok authtoken locally, then retry. See ops/README.md.'
fi

healthy() {
  curl --noproxy '*' --fail --silent --connect-timeout 2 --max-time 4 "$1" 2>/dev/null |
    python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("status")=="ok" else 1)' 2>/dev/null
}
docker_ready() { docker info >/dev/null 2>&1; }
printf '\nClinic demo: http://127.0.0.1:%s/api/health\nHEMS:        http://127.0.0.1:%s/health\nProxy:       http://127.0.0.1:%s\n' "$CLINIC_PORT" "$HEMS_PORT" "$PROXY_PORT"
printf 'Node: %s\n' "$NODE_BIN"
"$NODE_BIN" -e 'console.log("Allowed website origins:",require(process.argv[1]).loadConfig(process.argv[2]).allowedOrigins.join(", "));' "$SCRIPT_DIR/portfolio-proxy.cjs" "$CONFIG"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  status=0
  if healthy "http://127.0.0.1:$CLINIC_PORT/api/health"; then printf 'Clinic: ready\n'; else printf 'Clinic: not reachable\n'; status=1; fi
  if healthy "http://127.0.0.1:$HEMS_PORT/health"; then printf 'HEMS: ready\n'; else printf 'HEMS: not reachable\n'; status=1; fi
  if healthy "http://127.0.0.1:$PROXY_PORT/_proxy/health"; then printf 'Proxy: responding\n'; else printf 'Proxy: not running yet\n'; fi
  printf 'Check finished. Nothing started. ngrok account credentials have not been verified online.\n'
  exit "$status"
fi

RUN_DIR="${PORTFOLIO_RUN_DIR:-$SCRIPT_DIR/../work/portfolio-ngrok}"
umask 077
mkdir -p "$RUN_DIR"
LOCK_DIR="$RUN_DIR/launcher.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  previous_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$previous_pid" 2>/dev/null; then
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || fail "Cannot remove stale lock $LOCK_DIR."
    mkdir "$LOCK_DIR" || fail 'Another launcher is starting.'
  else
    fail 'The combined launcher is already running or starting. Stop it in its existing Terminal window first.'
  fi
fi
OWN_LOCK=1
printf '%s\n' "$$" > "$LOCK_DIR/pid"
cleanup() {
  trap - EXIT INT TERM
  for child in "$NGROK_PID" "$PROXY_PID" "$AWAKE_PID"; do
    if [[ -n "$child" ]]; then kill "$child" 2>/dev/null || true; fi
  done
  for child in "$NGROK_PID" "$PROXY_PID" "$AWAKE_PID"; do
    if [[ -n "$child" ]]; then wait "$child" 2>/dev/null || true; fi
  done
  if [[ "$OWN_LOCK" -eq 1 ]]; then rm -f "$LOCK_DIR/pid"; rmdir "$LOCK_DIR" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$START_BACKENDS" -eq 1 ]]; then
  command -v docker >/dev/null 2>&1 || fail 'Install Docker Desktop first.'
  if ! docker_ready; then
    printf '\nOpening Docker Desktop and waiting up to 90 seconds...\n'
    open -a Docker || fail 'Open Docker Desktop manually, then retry.'
    deadline=$((SECONDS + 90))
    until docker_ready; do
      [[ "$SECONDS" -lt "$deadline" ]] || fail 'Docker is not ready. Finish its initial setup and retry.'
      sleep 2
    done
  fi
  docker compose version >/dev/null 2>&1 || fail 'Docker Compose is unavailable.'
  printf '\nStarting Clinic demo and its isolated database...\n'
  (cd "$CLINIC_DIR" && CLINIC_PORT="$CLINIC_PORT" docker compose up -d --no-build clinic-demo) || fail 'Clinic demo startup failed. See ops/README.md.'
  printf '\nStarting HEMS and MLflow...\n'
  (cd "$HEMS_DIR" && docker compose up -d --no-build api) || fail 'HEMS startup failed. See ops/README.md.'
fi

printf '\nWaiting up to 180 seconds for both backend health endpoints...\n'
deadline=$((SECONDS + 180))
until healthy "http://127.0.0.1:$CLINIC_PORT/api/health" && healthy "http://127.0.0.1:$HEMS_PORT/health"; do
  if [[ "$START_BACKENDS" -eq 0 ]]; then fail 'Both backends must be healthy when using --no-start-backends.'; fi
  [[ "$SECONDS" -lt "$deadline" ]] || fail 'Backend health check failed. Inspect the backend logs described in ops/README.md.'
  sleep 2
done

printf '\nStarting the API proxy...\n'
"$NODE_BIN" "$SCRIPT_DIR/portfolio-proxy.cjs" "$CONFIG" > "$RUN_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
deadline=$((SECONDS + 15))
until healthy "http://127.0.0.1:$PROXY_PORT/_proxy/health"; do
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then cat "$RUN_DIR/proxy.log"; fail 'Proxy did not start.'; fi
  [[ "$SECONDS" -lt "$deadline" ]] || fail "Proxy is not ready. See $RUN_DIR/proxy.log."
  sleep 1
done
sleep 0.2
if ! kill -0 "$PROXY_PID" 2>/dev/null; then cat "$RUN_DIR/proxy.log"; fail 'Proxy could not claim its port; stop the process already using that port.'; fi
healthy "http://127.0.0.1:$PROXY_PORT/clinic/api/health" || fail 'Clinic route through proxy failed.'
healthy "http://127.0.0.1:$PROXY_PORT/hems/health" || fail 'HEMS route through proxy failed.'

if command -v caffeinate >/dev/null 2>&1; then caffeinate -i -w "$$" & AWAKE_PID=$!; fi
printf '\nBoth backends are ready. API paths:\n  /clinic/api/...\n  /hems/...\n'
printf 'Static frontends remain on GitHub Pages; these are API routes.\n'
printf 'Ollama must be running separately for local AI.\n'
printf 'Keep this window open. Ctrl+C stops proxy and tunnel; containers keep running.\n'

if [[ "$LOCAL_ONLY" -eq 1 ]]; then
  printf '\nLocal-only mode: no public ngrok tunnel.\n'
  wait "$PROXY_PID"
else
  printf '\nStarting one ngrok tunnel. Stop any old standalone HEMS/Clinic tunnel using this domain first.\n'
  if [[ -n "$PUBLIC_URL" ]]; then
    printf 'Clinic API base: %s/clinic\nHEMS API base:   %s/hems\n' "$PUBLIC_URL" "$PUBLIC_URL"
  else
    printf 'Use the HTTPS Forwarding URL printed by ngrok, adding /clinic or /hems for the frontend API base.\n'
  fi
  ngrok_args=(http "http://127.0.0.1:$PROXY_PORT")
  if [[ -n "$PUBLIC_URL" ]]; then ngrok_args+=(--url "$PUBLIC_URL"); fi
  ngrok "${ngrok_args[@]}" &
  NGROK_PID=$!
  # A proxy failure must also stop the public tunnel instead of leaving a broken URL.
  while kill -0 "$NGROK_PID" 2>/dev/null; do
    kill -0 "$PROXY_PID" 2>/dev/null || fail "Proxy stopped. See $RUN_DIR/proxy.log."
    sleep 1
  done
  wait "$NGROK_PID"
fi
