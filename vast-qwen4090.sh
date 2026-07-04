#!/usr/bin/env bash
set -euo pipefail

OFFER_ID="${OFFER_ID:-}"
DISK_GB="${DISK_GB:-80}"
IMAGE="${IMAGE:-vastai/llama-cpp:b9628-cuda-12.9}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SSH_PUB_KEY="${SSH_PUB_KEY:-$HOME/.ssh/id_rsa.pub}"
MODEL="${MODEL:-unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M}"
CONTEXT="${CONTEXT:-65536}"
CONTAINER_PORT="${CONTAINER_PORT:-18000}"
OPENCODE_CONFIG="${OPENCODE_CONFIG:-$HOME/.config/opencode/opencode.json}"
OPENCODE_PROVIDER_ID="${OPENCODE_PROVIDER_ID:-vast}"
OPENCODE_MODEL_ID="${OPENCODE_MODEL_ID:-qwen3.6-35b-a3b}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.98}"
MIN_CUDA="${MIN_CUDA:-12.8}"
MIN_DIRECT_PORTS="${MIN_DIRECT_PORTS:-2}"
SEARCH_LIMIT="${SEARCH_LIMIT:-80}"
EXCLUDE_GEOS="${EXCLUDE_GEOS:-CN}"

LLAMA_DIR="/opt/llama.cpp/cuda-12.8"
SESSION="llama"

PRIMARY_ARGS="./llama-server -hf ${MODEL} --host 0.0.0.0 --port ${CONTAINER_PORT} -ngl 99 -c ${CONTEXT} --jinja -fa on --cache-type-k q8_0 --cache-type-v q8_0"
FALLBACK_ARGS="./llama-server -hf ${MODEL} --host 0.0.0.0 --port ${CONTAINER_PORT} -ngl 99 -c ${CONTEXT} --jinja -fa on --cache-type-k q4_0 --cache-type-v q4_0 -b 1024 -ub 256"

SSH_PERMS_CMD="mkdir -p /root/.ssh; touch /root/.ssh/authorized_keys; chown root:root /root /root/.ssh /root/.ssh/authorized_keys; chmod 700 /root /root/.ssh; chmod 600 /root/.ssh/authorized_keys; (for i in \$(seq 1 600); do chown root:root /root /root/.ssh /root/.ssh/authorized_keys 2>/dev/null || true; chmod 700 /root /root/.ssh 2>/dev/null || true; chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true; sleep 1; done) &"
ONSTART_CMD="${SSH_PERMS_CMD} cd ${LLAMA_DIR} && export LD_LIBRARY_PATH=${LLAMA_DIR}:\$LD_LIBRARY_PATH && tmux new -d -s ${SESSION} '${PRIMARY_ARGS} 2>&1 | tee /root/llama.log'"

usage() {
  cat <<EOF
Usage:
  $0 create             Create the Vast instance and print SSH/API details
  $0 choose-offer       Print the cheapest currently suitable RTX 4090 offer
  $0 status INSTANCE_ID Print instance status, SSH URL, and API URL
  $0 logs INSTANCE_ID   Tail llama.cpp logs over SSH
  $0 restart INSTANCE_ID
                         Restart llama.cpp with fallback q4 KV cache settings
  $0 update-onstart INSTANCE_ID
                         Update stored onstart without restarting
  $0 wire-opencode INSTANCE_ID
                         Update global opencode config to use this instance
  $0 sync-opencode       Sync global opencode config from active Vast instances
  $0 cleanup-inactive    Destroy Vast instances that are not running
  $0 repair INSTANCE_ID  Reapply fixed onstart, stop/start, reattach SSH key
  $0 destroy INSTANCE_ID Destroy the instance

Config via env vars:
  OFFER_ID=${OFFER_ID:-auto}
  DISK_GB=${DISK_GB}
  IMAGE=${IMAGE}
  MODEL=${MODEL}
  CONTEXT=${CONTEXT}
  SSH_KEY=${SSH_KEY}
  SSH_PUB_KEY=${SSH_PUB_KEY}
  OPENCODE_CONFIG=${OPENCODE_CONFIG}
  OPENCODE_PROVIDER_ID=${OPENCODE_PROVIDER_ID}
  OPENCODE_MODEL_ID=${OPENCODE_MODEL_ID}
  MIN_RELIABILITY=${MIN_RELIABILITY}
  MIN_CUDA=${MIN_CUDA}
  MIN_DIRECT_PORTS=${MIN_DIRECT_PORTS}
  EXCLUDE_GEOS=${EXCLUDE_GEOS}

By default, create searches live offers and chooses the cheapest suitable
real 24GB RTX 4090. Set OFFER_ID only to force a specific offer:
  OFFER_ID=40016630 $0 create
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

parse_new_contract() {
  sed -nE "s/.*['\"]new_contract['\"]:[[:space:]]*([0-9]+).*/\1/p" | tail -1
}

offer_query() {
  local query
  query="num_gpus=1 gpu_name=RTX_4090 gpu_ram>=24 gpu_ram<30000 cuda_vers>=${MIN_CUDA} direct_port_count>=${MIN_DIRECT_PORTS} reliability>=${MIN_RELIABILITY} verified=true rentable=true"
  if [[ -n "$EXCLUDE_GEOS" ]]; then
    query="${query} geolocation notin [${EXCLUDE_GEOS}]"
  fi
  printf "%s" "$query"
}

choose_offer() {
  need_cmd vastai
  need_cmd python3

  local query
  query="$(offer_query)"
  echo "Searching RTX 4090 offers:" >&2
  echo "  $query" >&2

  vastai search offers "$query" \
    --storage "$DISK_GB" \
    --limit "$SEARCH_LIMIT" \
    --raw \
    -o "dph,reliability-" |
    python3 -c '
import json, sys

offers = json.load(sys.stdin)
if not offers:
    print("No suitable RTX 4090 offers found.", file=sys.stderr)
    sys.exit(2)

def total_cost(o):
    return float(o.get("dph_total") or o.get("search", {}).get("totalHour") or 999999)

def score_key(o):
    # Cheapest first; tie-break on reliability, PCIe bandwidth, internet up/down.
    return (
        total_cost(o),
        -float(o.get("reliability") or 0),
        -float(o.get("pcie_bw") or 0),
        -float(o.get("inet_down") or 0),
    )

selected = sorted(offers, key=score_key)[0]

def value(o, key, default=""):
    value = o.get(key, default)
    return default if value is None else value

print("Selected offer:", file=sys.stderr)
print("  id:          %s" % value(selected, "id"), file=sys.stderr)
print("  gpu:         %s (%s MB)" % (value(selected, "gpu_name"), value(selected, "gpu_ram")), file=sys.stderr)
print("  cost:        $%.4f/hr with requested disk" % total_cost(selected), file=sys.stderr)
print("  cuda:        %s" % value(selected, "cuda_max_good"), file=sys.stderr)
print("  driver:      %s" % value(selected, "driver_version"), file=sys.stderr)
print("  reliability: %s" % value(selected, "reliability"), file=sys.stderr)
print("  location:    %s" % value(selected, "geolocation"), file=sys.stderr)
print("  direct ports:%s" % value(selected, "direct_port_count"), file=sys.stderr)
print("  pcie bw:     %s" % value(selected, "pcie_bw"), file=sys.stderr)
print(value(selected, "id"))
'
}

json_field() {
  local expr="$1"
  python3 -c '
import json, sys
data = json.load(sys.stdin)
expr = sys.argv[1]
cur = data
try:
    for part in expr.split("."):
        if part.endswith("]"):
            name, idx = part[:-1].split("[")
            cur = cur[name][int(idx)]
        else:
            cur = cur[part]
except (KeyError, IndexError, TypeError):
    sys.exit(1)
print(cur)
' "$expr"
}

instance_json() {
  vastai show instance "$1" --raw
}

public_ip() {
  instance_json "$1" | json_field "public_ipaddr"
}

host_port() {
  instance_json "$1" | json_field "ports.${CONTAINER_PORT}/tcp[0].HostPort"
}

wait_for_api_port() {
  local instance_id="$1"
  local i
  for i in $(seq 1 60); do
    if host_port "$instance_id" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for published port ${CONTAINER_PORT}/tcp on instance $instance_id." >&2
  return 1
}

write_onstart_file() {
  local path="$1"
  printf "%s\n" "$ONSTART_CMD" > "$path"
}

ssh_url() {
  vastai ssh-url "$1"
}

ssh_command_from_url() {
  local url host port
  url="$(ssh_url "$1")"
  host="${url#ssh://root@}"
  port="${host##*:}"
  host="${host%:*}"
  printf "ssh -i %q -o IdentitiesOnly=yes -p %q root@%q" "$SSH_KEY" "$port" "$host"
}

remote_run() {
  local instance_id="$1"
  shift
  local url host port
  url="$(ssh_url "$instance_id")"
  host="${url#ssh://root@}"
  port="${host##*:}"
  host="${host%:*}"
  ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" "$@"
}

wait_for_instance() {
  local instance_id="$1"
  local i
  for i in $(seq 1 60); do
    if vastai show instance "$instance_id" --raw >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for instance $instance_id to appear." >&2
  return 1
}

wait_for_ssh_url() {
  local instance_id="$1"
  local i
  for i in $(seq 1 60); do
    if ssh_url "$instance_id" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for SSH URL for instance $instance_id." >&2
  return 1
}

print_details() {
  local instance_id="$1"
  local ip port
  ip="$(public_ip "$instance_id")"
  port="$(host_port "$instance_id")"

  echo
  echo "Instance: $instance_id"
  echo "SSH URL:  $(ssh_url "$instance_id")"
  echo "SSH cmd:  $(ssh_command_from_url "$instance_id")"
  echo "Health:   http://${ip}:${port}/health"
  echo "API base: http://${ip}:${port}/v1"
  echo
  echo "Test:"
  echo "  curl http://${ip}:${port}/health"
  echo
  echo "Logs:"
  echo "  $0 logs $instance_id"
  echo
  echo "Fallback restart if it OOMs:"
  echo "  $0 restart $instance_id"
}

create_instance() {
  need_cmd vastai
  need_cmd python3

  if [[ ! -f "$SSH_PUB_KEY" ]]; then
    echo "Missing SSH public key: $SSH_PUB_KEY" >&2
    exit 1
  fi

  local selected_offer_id
  if [[ -n "$OFFER_ID" ]]; then
    selected_offer_id="$OFFER_ID"
    echo "Using forced offer $selected_offer_id from OFFER_ID."
  else
    selected_offer_id="$(choose_offer)"
  fi

  echo "Creating Vast instance from offer $selected_offer_id..."
  local output instance_id
  output="$(
    vastai create instance "$selected_offer_id" \
      --image "$IMAGE" \
      --disk "$DISK_GB" \
      --ssh \
      --direct \
      --env "-p ${CONTAINER_PORT}:${CONTAINER_PORT}" \
      --onstart-cmd "$ONSTART_CMD"
  )"
  echo "$output"

  instance_id="$(printf "%s\n" "$output" | parse_new_contract)"
  if [[ -z "$instance_id" ]]; then
    echo "Could not parse new instance id from Vast output." >&2
    exit 1
  fi

  echo "Waiting for instance $instance_id..."
  wait_for_instance "$instance_id"

  echo "Attaching SSH key..."
  vastai attach ssh "$instance_id" "$SSH_PUB_KEY" || true

  echo "Waiting for SSH URL..."
  wait_for_ssh_url "$instance_id"

  echo "Waiting for published API port..."
  wait_for_api_port "$instance_id"

  print_details "$instance_id"
}

status_instance() {
  local instance_id="$1"
  need_cmd vastai
  need_cmd python3
  wait_for_api_port "$instance_id"
  print_details "$instance_id"
}

logs_instance() {
  local instance_id="$1"
  remote_run "$instance_id" "tail -f /root/llama.log"
}

restart_fallback() {
  local instance_id="$1"
  local cmd
  cmd="tmux kill-session -t ${SESSION} 2>/dev/null || true; cd ${LLAMA_DIR} && export LD_LIBRARY_PATH=${LLAMA_DIR}:\$LD_LIBRARY_PATH && tmux new -d -s ${SESSION} '${FALLBACK_ARGS} 2>&1 | tee /root/llama.log'"
  remote_run "$instance_id" "$cmd"
  echo "Restarted llama.cpp with fallback KV cache settings."
  echo "Logs: $0 logs $instance_id"
}

repair_instance() {
  local instance_id="$1"
  local update_output
  need_cmd vastai
  need_cmd python3

  echo "Updating onstart for instance $instance_id..."
  update_output="$(vastai update instance "$instance_id" --image "$IMAGE" --onstart "$ONSTART_CMD" 2>&1)"
  echo "$update_output"
  if grep -qi "failed with error\\|invalid args\\|error 400" <<<"$update_output"; then
    echo "Vast rejected the onstart update. Destroy and recreate this fresh instance instead:" >&2
    echo "  $0 destroy $instance_id" >&2
    echo "  $0 create" >&2
    exit 1
  fi

  echo "Stopping instance $instance_id..."
  vastai stop instance "$instance_id"

  echo "Starting instance $instance_id..."
  vastai start instance "$instance_id"

  echo "Waiting for instance $instance_id..."
  wait_for_instance "$instance_id"

  echo "Attaching SSH key..."
  vastai attach ssh "$instance_id" "$SSH_PUB_KEY" || true

  echo "Waiting for SSH URL..."
  wait_for_ssh_url "$instance_id"

  echo "Waiting for published API port..."
  wait_for_api_port "$instance_id"

  print_details "$instance_id"
}

update_onstart_only() {
  local instance_id="$1"
  local update_output
  need_cmd vastai

  update_output="$(vastai update instance "$instance_id" --image "$IMAGE" --onstart "$ONSTART_CMD" 2>&1)"
  echo "$update_output"
  if grep -qi "failed with error\\|invalid args\\|error 400" <<<"$update_output"; then
    echo "Vast rejected the onstart update." >&2
    exit 1
  fi
}

wire_opencode() {
  local instance_id="$1"
  local ip port base_url
  need_cmd vastai
  need_cmd python3
  need_cmd node

  ip="$(public_ip "$instance_id")"
  port="$(host_port "$instance_id")"
  base_url="http://${ip}:${port}/v1"

  mkdir -p "$(dirname "$OPENCODE_CONFIG")"
  if [[ ! -f "$OPENCODE_CONFIG" ]]; then
    printf '{\n  "$schema": "https://opencode.ai/config.json"\n}\n' > "$OPENCODE_CONFIG"
  fi

  OPENCODE_CONFIG="$OPENCODE_CONFIG" \
  OPENCODE_PROVIDER_ID="$OPENCODE_PROVIDER_ID" \
  OPENCODE_MODEL_ID="$OPENCODE_MODEL_ID" \
  OPENCODE_BASE_URL="$base_url" \
  MODEL="$MODEL" \
  CONTEXT="$CONTEXT" \
  node <<'NODE'
const fs = require("fs")
const file = process.env.OPENCODE_CONFIG
const providerID = process.env.OPENCODE_PROVIDER_ID
const modelID = process.env.OPENCODE_MODEL_ID
const baseURL = process.env.OPENCODE_BASE_URL
const upstreamModel = process.env.MODEL
const context = Number(process.env.CONTEXT || 65536)

const data = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {}
data.$schema ??= "https://opencode.ai/config.json"
data.provider ??= {}
data.provider[providerID] = {
  name: "Vast Qwen 4090",
  npm: "@ai-sdk/openai-compatible",
  models: {
    [modelID]: {
      id: upstreamModel,
      name: "Qwen3.6 35B A3B (Vast 4090)",
      temperature: true,
      tool_call: true,
      limit: {
        context,
        output: 8192,
      },
    },
  },
  options: {
    apiKey: "sk-no-key-required",
    baseURL,
    timeout: false,
    headerTimeout: false,
    chunkTimeout: 600000,
  },
}
data.model = `${providerID}/${modelID}`
fs.writeFileSync(file, JSON.stringify(data, null, 2) + "\n")
console.log(`Updated ${file}`)
console.log(`Default model: ${data.model}`)
console.log(`Base URL: ${baseURL}`)
NODE
}

sync_opencode() {
  need_cmd vastai
  need_cmd node

  local instances_json
  instances_json="$(vastai show instances-v1 --raw)"

  mkdir -p "$(dirname "$OPENCODE_CONFIG")"
  if [[ ! -f "$OPENCODE_CONFIG" ]]; then
    printf '{\n  "$schema": "https://opencode.ai/config.json"\n}\n' > "$OPENCODE_CONFIG"
  fi

  OPENCODE_CONFIG="$OPENCODE_CONFIG" \
  OPENCODE_MODEL_ID="$OPENCODE_MODEL_ID" \
  MODEL="$MODEL" \
  CONTEXT="$CONTEXT" \
  CONTAINER_PORT="$CONTAINER_PORT" \
  INSTANCES_JSON="$instances_json" \
  node <<'NODE'
const fs = require("fs")

const file = process.env.OPENCODE_CONFIG
const modelID = process.env.OPENCODE_MODEL_ID
const upstreamModel = process.env.MODEL
const context = Number(process.env.CONTEXT || 65536)
const containerPort = process.env.CONTAINER_PORT || "18000"
const portKey = `${containerPort}/tcp`

const parsed = JSON.parse(process.env.INSTANCES_JSON)
const instances = Array.isArray(parsed) ? parsed : parsed.instances || []

const active = instances
  .filter((inst) => {
    const state = inst.actual_status || inst.cur_state
    const intended = inst.intended_status
    const ports = inst.ports || {}
    return state === "running" &&
      intended !== "stopped" &&
      inst.public_ipaddr &&
      Array.isArray(ports[portKey]) &&
      ports[portKey][0]?.HostPort
  })
  .sort((a, b) => Number(a.id) - Number(b.id))

const data = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {}
data.$schema ??= "https://opencode.ai/config.json"
data.provider ??= {}

const managedProvider = /^vast(?:-\d+)?$/
const removed = []
for (const key of Object.keys(data.provider)) {
  if (managedProvider.test(key)) {
    delete data.provider[key]
    removed.push(key)
  }
}

const synced = []
for (const inst of active) {
  const providerID = `vast-${inst.id}`
  const baseURL = `http://${inst.public_ipaddr}:${inst.ports[portKey][0].HostPort}/v1`
  data.provider[providerID] = {
    name: `Vast Qwen 4090 (${inst.id})`,
    npm: "@ai-sdk/openai-compatible",
    models: {
      [modelID]: {
        id: upstreamModel,
        name: "Qwen3.6 35B A3B (Vast 4090)",
        temperature: true,
        tool_call: true,
        limit: {
          context,
          output: 8192,
        },
      },
    },
    options: {
      apiKey: "sk-no-key-required",
      baseURL,
      timeout: false,
      headerTimeout: false,
      chunkTimeout: 600000,
    },
  }
  synced.push({ providerID, model: `${providerID}/${modelID}`, baseURL })
}

if (synced.length > 0) {
  data.model = synced[0].model
} else if (typeof data.model === "string" && /^vast(?:-\d+)?\//.test(data.model)) {
  delete data.model
}

fs.writeFileSync(file, JSON.stringify(data, null, 2) + "\n")

console.log(`Updated ${file}`)
console.log(`Removed managed Vast providers: ${removed.length ? removed.join(", ") : "none"}`)
if (synced.length === 0) {
  console.log("Synced active Vast providers: none")
} else {
  console.log("Synced active Vast providers:")
  for (const item of synced) {
    console.log(`  ${item.model} -> ${item.baseURL}`)
  }
  console.log(`Default model: ${data.model}`)
}
NODE
}

cleanup_inactive() {
  need_cmd vastai
  need_cmd python3

  local ids
  ids="$(
    vastai show instances-v1 --raw | python3 -c '
import json, sys
data = json.load(sys.stdin)
for item in data.get("instances", []):
    state = item.get("actual_status") or item.get("cur_state")
    intended = item.get("intended_status")
    if state != "running" and intended != "running":
        print(item["id"])
'
  )"

  if [[ -z "$ids" ]]; then
    echo "No inactive Vast instances found."
    return 0
  fi

  while IFS= read -r instance_id; do
    [[ -n "$instance_id" ]] || continue
    echo "Destroying inactive instance $instance_id..."
    vastai destroy instance "$instance_id" -y
  done <<<"$ids"
}

destroy_instance() {
  local instance_id="$1"
  vastai destroy instance "$instance_id" -y
}

main() {
  local action="${1:-}"
  case "$action" in
    create)
      create_instance
      ;;
    choose-offer)
      choose_offer
      ;;
    status)
      [[ $# -eq 2 ]] || { usage; exit 1; }
      status_instance "$2"
      ;;
    logs)
      [[ $# -eq 2 ]] || { usage; exit 1; }
      logs_instance "$2"
      ;;
    restart)
      [[ $# -eq 2 ]] || { usage; exit 1; }
      restart_fallback "$2"
      ;;
    update-onstart)
      [[ $# -eq 2 ]] || { usage; exit 1; }
      update_onstart_only "$2"
      ;;
    wire-opencode)
      [[ $# -eq 2 ]] || { usage; exit 1; }
      wire_opencode "$2"
      ;;
    sync-opencode)
      sync_opencode
      ;;
    cleanup-inactive)
      cleanup_inactive
      ;;
    repair)
      [[ $# -eq 2 ]] || { usage; exit 1; }
      repair_instance "$2"
      ;;
    destroy)
      [[ $# -eq 2 ]] || { usage; exit 1; }
      destroy_instance "$2"
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "Unknown action: $action" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
