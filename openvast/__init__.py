#!/usr/bin/env python3
"""
openvast — a muted terminal dashboard to manage vast.ai GPU instances running
llama.cpp models, keeping opencode wired to whichever instances are healthy.

Features
  - List running / paused GPU instances with model, status, health, $/hr, uptime
  - Launch a model on a GPU: pick a model, then pick from GPU offers that FIT it
    (offers are filtered so only GPUs with enough VRAM are shown)
  - Pause (stop) / resume (start) / delete (destroy) an instance
  - Cost monitor: current $/hr plus spend over last 24h / 7d / 30d
  - Live start progress + health (provisioning -> loading -> healthy)
  - When an instance becomes healthy it is auto-added to opencode;
    when paused or deleted it is auto-removed from opencode.

Usage
  openvast              # launch the TUI  (or: python3 -m openvast)
  openvast --selftest   # exercise the vast/opencode backend, no UI
  openvast --models     # print the model registry

Keys (in the TUI): n launch · p pause · r resume · d delete · Enter details
                   l logs/progress · q quit   (table auto-refreshes every minute)

Requires: vastai CLI (authenticated), python3, textual, pyyaml, opencode, and an
SSH key at ~/.ssh/id_rsa(.pub) for the log viewer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

IMAGE = os.environ.get("VAST_IMAGE", "vastai/llama-cpp:b9628-cuda-12.9")
SSH_KEY = os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "id_rsa"))
SSH_PUB_KEY = os.environ.get("SSH_PUB_KEY", str(Path.home() / ".ssh" / "id_rsa.pub"))
OPENCODE_CONFIG = Path(
    os.environ.get("OPENCODE_CONFIG", str(Path.home() / ".config" / "opencode" / "opencode.json"))
)
MIN_RELIABILITY = float(os.environ.get("MIN_RELIABILITY", "0.98"))
MIN_CUDA = os.environ.get("MIN_CUDA", "12.8")
# Require directly-reachable ports, else the model API is unreachable from your
# machine (health never succeeds -> instance is stuck "loading" forever).
MIN_DIRECT_PORTS = int(os.environ.get("MIN_DIRECT_PORTS", "2"))
EXCLUDE_GEOS = os.environ.get("EXCLUDE_GEOS", "CN")
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "60"))

LLAMA_DIR = "/opt/llama.cpp/cuda-12.8"
SESSION = "llama"

# --------------------------------------------------------------------------- #
# Model registry (loaded from models.yaml)
#
# min_vram_gb drives the "only show GPUs that fit" filter. extra_args are the
# llama-server flags appended after the common ones. Edit models.yaml to add
# your own models — no code change needed.
# --------------------------------------------------------------------------- #

def _resolve_models_file() -> Path:
    """Find models.yaml across dev and installed layouts.

    Precedence: $VAST_MODELS_FILE, ./models.yaml (cwd override),
    ~/.config/openvast/models.yaml (user override), then the shipped default
    inside the package.
    """
    env = os.environ.get("VAST_MODELS_FILE")
    if env:
        return Path(env)
    for cand in (
        Path.cwd() / "models.yaml",
        Path.home() / ".config" / "openvast" / "models.yaml",
        Path(__file__).with_name("models.yaml"),
    ):
        if cand.is_file():
            return cand
    return Path(__file__).with_name("models.yaml")


MODELS_FILE = _resolve_models_file()

DEFAULT_EXTRA_ARGS = "--jinja -fa on --cache-type-k q8_0 --cache-type-v q8_0 --metrics"


@dataclass
class Model:
    key: str
    name: str
    hf: str                      # -hf repo:quant for llama-server
    min_vram_gb: int
    disk_gb: int
    context: int
    port: int = 18000
    image: str = IMAGE
    extra_args: str = DEFAULT_EXTRA_ARGS
    output_limit: int = 8192
    tool_call: bool = True
    reasoning: bool = True


# Built-in fallback used only if models.yaml is missing/unreadable.
_FALLBACK = {
    "default_model": "qwen3.6-35b-a3b",
    "models": [
        {"key": "qwen3.6-35b-a3b", "name": "Qwen3.6 35B A3B (Q4)",
         "hf": "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
         "min_vram_gb": 24, "disk_gb": 80, "context": 65536},
    ],
}


def _model_from_dict(d: dict) -> Model:
    return Model(
        key=str(d["key"]),
        name=str(d.get("name", d["key"])),
        hf=str(d["hf"]),
        min_vram_gb=int(d["min_vram_gb"]),
        disk_gb=int(d.get("disk_gb", 80)),
        context=int(d.get("context", 65536)),
        port=int(d.get("port", 18000)),
        image=str(d.get("image") or IMAGE),
        extra_args=str(d.get("extra_args", DEFAULT_EXTRA_ARGS)),
        output_limit=int(d.get("output_limit", 8192)),
        tool_call=bool(d.get("tool_call", True)),
        reasoning=bool(d.get("reasoning", True)),
    )


def load_models() -> tuple[dict[str, "Model"], str]:
    """Load the model registry from models.yaml (falls back to a built-in)."""
    data = None
    try:
        import yaml  # noqa: PLC0415
        if MODELS_FILE.is_file():
            data = yaml.safe_load(MODELS_FILE.read_text())
        else:
            print(f"warning: {MODELS_FILE} not found; using built-in model", file=sys.stderr)
    except ModuleNotFoundError:
        print("warning: PyYAML not installed (`pip install pyyaml`); using built-in model",
              file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not parse {MODELS_FILE}: {exc}; using built-in model", file=sys.stderr)

    if not data:
        data = _FALLBACK
    defaults = data.get("defaults") or {}
    entries = data.get("models") or []
    models: dict[str, Model] = {}
    for entry in entries:
        merged = {**defaults, **entry}
        try:
            m = _model_from_dict(merged)
        except (KeyError, ValueError, TypeError) as exc:
            print(f"warning: skipping bad model entry {entry!r}: {exc}", file=sys.stderr)
            continue
        models[m.key] = m
    if not models:
        models = {m.key: m for m in [_model_from_dict(_FALLBACK["models"][0])]}
    default_key = data.get("default_model")
    if default_key not in models:
        default_key = next(iter(models))
    return models, default_key


MODELS, DEFAULT_MODEL_KEY = load_models()


def model_for_label(label: str | None) -> Model:
    """Map an instance label back to a Model (fall back to the default)."""
    if label and label in MODELS:
        return MODELS[label]
    return MODELS[DEFAULT_MODEL_KEY]


# --------------------------------------------------------------------------- #
# vast CLI backend
# --------------------------------------------------------------------------- #


class VastError(RuntimeError):
    pass


def _run(args: list[str], timeout: int = 60) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise VastError(f"command not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VastError(f"timeout running: {' '.join(args)}") from exc
    if proc.returncode != 0:
        raise VastError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
    return proc.stdout


def _run_json(args: list[str], timeout: int = 60):
    out = _run(args, timeout=timeout).strip()
    if not out:
        return None
    return json.loads(out)


@dataclass
class Instance:
    id: int
    label: str | None
    gpu_name: str
    num_gpus: int
    gpu_ram_mb: int
    dph: float
    actual_status: str
    cur_state: str
    intended_status: str
    public_ip: str | None
    ports: dict
    start_date: float | None
    status_msg: str | None
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def model(self) -> Model:
        return model_for_label(self.label)

    def host_port(self) -> int | None:
        entry = (self.ports or {}).get(f"{self.model.port}/tcp")
        if isinstance(entry, list) and entry and entry[0].get("HostPort"):
            try:
                return int(entry[0]["HostPort"])
            except (TypeError, ValueError):
                return None
        return None

    def base_url(self) -> str | None:
        port = self.host_port()
        if self.public_ip and port:
            return f"http://{self.public_ip}:{port}/v1"
        return None

    def uptime_str(self) -> str:
        if not self.start_date:
            return "-"
        secs = max(0, int(time.time() - self.start_date))
        h, rem = divmod(secs, 3600)
        m, _ = divmod(rem, 60)
        if h >= 24:
            return f"{h // 24}d{h % 24}h"
        return f"{h}h{m:02d}m"

    @property
    def is_running(self) -> bool:
        return (self.actual_status == "running" or self.cur_state == "running") \
            and self.intended_status != "stopped"

    @property
    def is_paused(self) -> bool:
        return (
            self.intended_status == "stopped"
            or self.actual_status in {"exited", "stopped"}
            or self.cur_state == "stopped"
        )


def list_instances() -> list[Instance]:
    data = _run_json(["vastai", "show", "instances-v1", "--raw"], timeout=30)
    items = data.get("instances", []) if isinstance(data, dict) else (data or [])
    out = []
    for it in items:
        out.append(
            Instance(
                id=int(it["id"]),
                label=it.get("label"),
                gpu_name=it.get("gpu_name") or "?",
                num_gpus=int(it.get("num_gpus") or 1),
                gpu_ram_mb=int(it.get("gpu_ram") or 0),
                dph=float(it.get("dph_total") or 0.0),
                actual_status=it.get("actual_status") or "",
                cur_state=it.get("cur_state") or "",
                intended_status=it.get("intended_status") or "",
                public_ip=it.get("public_ipaddr"),
                ports=it.get("ports") or {},
                start_date=float(it["start_date"]) if it.get("start_date") else None,
                status_msg=it.get("status_msg"),
                raw=it,
            )
        )
    out.sort(key=lambda i: i.id)
    return out


def probe(inst: Instance, timeout: float = 3.0) -> tuple[bool, str]:
    """Probe /health, returning (healthy, stage).

    stage: "ok" (200), "loading" (llama.cpp returns 503 while loading weights),
    or "" (connection refused / timeout -> container booting or model still
    downloading, server not yet bound).
    """
    base = inst.base_url()
    if not base:
        return (False, "")
    url = base[:-3] + "/health" if base.endswith("/v1") else base + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            if resp.status == 200 and '"ok"' in body:
                return (True, "ok")
            return (False, "loading")
    except urllib.error.HTTPError:
        return (False, "loading")   # 503 "loading model"
    except Exception:
        return (False, "")          # refused / timeout


def health_ok(inst: Instance, timeout: float = 3.0) -> bool:
    return probe(inst, timeout)[0]


# llama.cpp /metrics (Prometheus) gauge names -> our short keys.
# predicted_tokens_seconds = generation throughput (decode tok/s)
# prompt_tokens_seconds    = prompt-processing throughput = rate the KV cache
#                            is filled (prefill / "kv cache tokens per second")
_METRIC_KEYS = {
    "llamacpp:predicted_tokens_seconds": "tok_s",
    "llamacpp:prompt_tokens_seconds": "kv_s",
}


def fetch_metrics(inst: Instance, timeout: float = 3.0) -> dict | None:
    """Return {"tok_s": float, "kv_s": float} from the instance's /metrics,
    or None if unreachable / metrics disabled (server launched without --metrics)."""
    base = inst.base_url()
    if not base:
        return None
    url = base[:-3] + "/metrics" if base.endswith("/v1") else base + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            body = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line[0] == "#" or " " not in line:
            continue
        name, _, value = line.partition(" ")
        key = _METRIC_KEYS.get(name)
        if key:
            try:
                out[key] = float(value)
            except ValueError:
                pass
    return out or None


_ERROR_MARKERS = (
    "failed", "error", "oci runtime", "exit status", "cannot ", "retries exceeded",
)
# Docker image-pull progress markers (host fetching the container image).
_PULL_MARKERS = (
    "download complete", "pulling", "extracting", "pull complete",
    "downloading", "verifying", "waiting",
)


def derive_status(inst: Instance, healthy: bool, stage: str = "") -> str:
    """Human-readable start progress / status."""
    # surface host/container failures first, even if vast then marks it exited,
    # so a broken instance shows "error" instead of "paused"/"starting" forever.
    msg = (inst.status_msg or "").lower()
    if msg and any(w in msg for w in _ERROR_MARKERS):
        return "error"
    if inst.is_paused:
        return "paused"
    if healthy:
        return "healthy"
    if not inst.is_running:
        return (inst.status_msg or inst.actual_status or "provisioning").lower()[:18]
    if stage == "loading":
        return "loading"        # server up, loading weights / warming up
    if inst.host_port() is None:
        # before the port is published the host is usually pulling the image
        if any(w in msg for w in _PULL_MARKERS):
            return "pulling"    # host downloading the docker image
        return "starting"
    return "downloading"        # port mapped, server not bound yet (dl / boot)


def dur(secs: float) -> str:
    secs = int(max(0, secs))
    m, s = divmod(secs, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def min_vram_mb(model: Model) -> int:
    # A "24GB" card actually reports ~24564 MB, so allow ~1GB under nominal.
    return model.min_vram_gb * 1024 - 1024


def gb(mb: int) -> int:
    return round(mb / 1024)


def offer_query(model: Model) -> str:
    # NOTE: in vast's search DSL gpu_ram is in GB (result JSON reports MB).
    q = (
        f"num_gpus=1 gpu_ram>={model.min_vram_gb} cuda_vers>={MIN_CUDA} "
        f"direct_port_count>={MIN_DIRECT_PORTS} "
        f"reliability>={MIN_RELIABILITY} verified=true rentable=true"
    )
    if EXCLUDE_GEOS:
        q += f" geolocation notin [{EXCLUDE_GEOS}]"
    return q


@dataclass
class Offer:
    id: int
    gpu_name: str
    gpu_ram_mb: int
    dph: float
    cuda: str
    reliability: float
    location: str
    inet_down: float
    disk_space: float


def search_offers(model: Model, limit: int = 25) -> list[Offer]:
    data = _run_json(
        [
            "vastai", "search", "offers", offer_query(model),
            "--storage", str(model.disk_gb),
            "--limit", str(SEARCH_LIMIT),
            "--raw", "-o", "dph",
        ],
        timeout=60,
    )
    offers = data if isinstance(data, list) else []
    out = []
    for o in offers:
        # enforce the fit client-side too (per-card VRAM must hold the model)
        if int(o.get("gpu_ram") or 0) < min_vram_mb(model):
            continue
        out.append(
            Offer(
                id=int(o["id"]),
                gpu_name=o.get("gpu_name") or "?",
                gpu_ram_mb=int(o.get("gpu_ram") or 0),
                dph=float(o.get("dph_total") or 0.0),
                cuda=str(o.get("cuda_max_good") or ""),
                reliability=float(o.get("reliability") or 0.0),
                location=o.get("geolocation") or "",
                inet_down=float(o.get("inet_down") or 0.0),
                disk_space=float(o.get("disk_space") or 0.0),
            )
        )
    out.sort(key=lambda o: o.dph)
    return out[:limit]


def _ssh_perms_cmd() -> str:
    inner = (
        "for i in $(seq 1 600); do "
        "chown root:root /root /root/.ssh /root/.ssh/authorized_keys 2>/dev/null || true; "
        "chmod 700 /root /root/.ssh 2>/dev/null || true; "
        "chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true; "
        "sleep 1; done"
    )
    return (
        "mkdir -p /root/.ssh; touch /root/.ssh/authorized_keys; "
        "chown root:root /root /root/.ssh /root/.ssh/authorized_keys; "
        "chmod 700 /root /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
        f"({inner}) &"
    )


def build_onstart(model: Model) -> str:
    primary = (
        f"./llama-server -hf {model.hf} --host 0.0.0.0 --port {model.port} "
        f"-ngl 99 -c {model.context} {model.extra_args}"
    )
    return (
        f"{_ssh_perms_cmd()} cd {LLAMA_DIR} && "
        f"export LD_LIBRARY_PATH={LLAMA_DIR}:$LD_LIBRARY_PATH && "
        f"tmux new -d -s {SESSION} '{primary} 2>&1 | tee /root/llama.log'"
    )


def create_instance(offer_id: int, model: Model) -> int:
    if not Path(SSH_PUB_KEY).is_file():
        raise VastError(f"missing SSH public key: {SSH_PUB_KEY}")
    out = _run(
        [
            "vastai", "create", "instance", str(offer_id),
            "--image", model.image,
            "--disk", str(model.disk_gb),
            "--ssh", "--direct",
            "--env", f"-p {model.port}:{model.port}",
            "--onstart-cmd", build_onstart(model),
            "--label", model.key,
        ],
        timeout=120,
    )
    m = re.search(r"new_contract['\"]?\s*:\s*(\d+)", out)
    if not m:
        raise VastError(f"could not parse new instance id from: {out.strip()[:200]}")
    new_id = int(m.group(1))
    try:  # attach ssh key (best effort)
        _run(["vastai", "attach", "ssh", str(new_id), SSH_PUB_KEY], timeout=60)
    except VastError:
        pass
    return new_id


def stop_instance(instance_id: int) -> None:
    _run(["vastai", "stop", "instance", str(instance_id)], timeout=60)


def start_instance(instance_id: int) -> None:
    _run(["vastai", "start", "instance", str(instance_id)], timeout=60)


def destroy_instance(instance_id: int) -> None:
    _run(["vastai", "destroy", "instance", str(instance_id), "-y"], timeout=60)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def cost_summary(instances: list[Instance]) -> dict:
    now = int(time.time())
    per_hour = sum(i.dph for i in instances if i.is_running)
    windows = {"l24h": 86400, "l7d": 7 * 86400, "l30d": 30 * 86400}
    result = {"per_hour": per_hour, "l24h": None, "l7d": None, "l30d": None}
    try:
        data = _run_json(
            [
                "vastai", "show", "invoices-v1", "--charges",
                "-s", str(now - 30 * 86400), "-e", str(now),
                "--raw", "--limit", "100",
            ],
            timeout=45,
        )
        charges = (data or {}).get("results", []) if isinstance(data, dict) else []
        for name, span in windows.items():
            cutoff = now - span
            result[name] = sum(
                float(c.get("amount") or 0.0)
                for c in charges
                if (c.get("end") or c.get("start") or 0) >= cutoff
            )
    except (VastError, json.JSONDecodeError):
        pass
    return result


def get_balance() -> float | None:
    """Available vast.ai prepaid credit (also serves as an auth check)."""
    data = _run_json(["vastai", "show", "user", "--raw"], timeout=20)
    if isinstance(data, dict):
        return float(data.get("credit") or 0.0)
    return None


# --------------------------------------------------------------------------- #
# environment / prerequisites
# --------------------------------------------------------------------------- #


@dataclass
class EnvCheck:
    vast_installed: bool
    vast_authed: bool
    opencode_installed: bool
    balance: float | None
    reason: str

    @property
    def can_edit(self) -> bool:
        # Editing (launch/pause/delete + opencode wiring) needs an authenticated
        # vast CLI AND opencode installed.
        return self.vast_authed and self.opencode_installed


def check_environment() -> EnvCheck:
    vast = shutil.which("vastai") is not None
    opencode = shutil.which("opencode") is not None
    balance: float | None = None
    authed = False
    if vast:
        try:
            balance = get_balance()
            authed = balance is not None
        except VastError:
            authed = False
    reasons = []
    if not vast:
        reasons.append("vastai CLI not installed")
    elif not authed:
        reasons.append("vastai not authenticated (run: vastai set api-key <KEY>)")
    if not opencode:
        reasons.append("opencode not installed")
    return EnvCheck(vast, authed, opencode, balance, " · ".join(reasons))


# --------------------------------------------------------------------------- #
# opencode wiring
# --------------------------------------------------------------------------- #

# Managed provider keys: the primary instance is "openvast"; any extras are
# "openvast-<id>". The legacy "vast" / "vast-<id>" forms are still matched so
# they get cleaned up on reconcile.
_MANAGED_RE = re.compile(r"^(?:open)?vast(?:-\d+)?$")


def _load_opencode() -> dict:
    if OPENCODE_CONFIG.is_file():
        try:
            return json.loads(OPENCODE_CONFIG.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _provider_block(inst: Instance) -> dict:
    """One provider per instance — its baseURL IS the instance, so opencode
    routes correctly (no reliance on unsupported per-model baseURL)."""
    m = inst.model
    return {
        "name": f"openvast · {inst.gpu_name}",
        "npm": "@ai-sdk/openai-compatible",
        "models": {
            m.key: {
                "id": m.hf,
                "name": f"{m.name} ({inst.gpu_name})",
                "temperature": True,
                "tool_call": m.tool_call,
                "limit": {"context": m.context, "output": m.output_limit},
            }
        },
        "options": {
            "apiKey": "sk-no-key-required",
            "baseURL": inst.base_url(),
            "timeout": False,
            "headerTimeout": False,
            "chunkTimeout": 600000,
        },
    }


def reconcile_opencode(healthy: list[Instance]) -> dict:
    """Wire one `openvast-<id>` provider per healthy instance.

    opencode has no per-model baseURL, so every instance (a distinct endpoint)
    is its own provider — selecting it truly routes to that GPU. Keys are
    `openvast-<id>`; the display name is `openvast · <gpu>`. Stale managed
    providers (legacy `vast-*`, bare `openvast`) are removed.
    Returns {"added": [...], "removed": [...]}.
    """
    data = _load_opencode()
    data.setdefault("$schema", "https://opencode.ai/config.json")
    providers = data.setdefault("provider", {})

    live = sorted((i for i in healthy if i.base_url()), key=lambda i: i.id)
    want = {f"openvast-{i.id}": i for i in live}
    default_model = (
        f"openvast-{live[0].id}/{live[0].model.key}" if live else None
    )
    managed = {k for k in providers if _MANAGED_RE.match(k)}

    removed, added, dirty = [], [], False
    for key in list(managed):
        if key not in want:                      # drops legacy vast-*/bare openvast
            providers.pop(key, None)
            removed.append(key)
            dirty = True

    for key, inst in want.items():
        block = _provider_block(inst)
        if providers.get(key) != block:
            if key not in providers:
                added.append(key)
            providers[key] = block
            dirty = True

    # keep a working default model pointing at the live provider, without
    # clobbering a non-managed default the user chose themselves.
    default = data.get("model")
    prov = default.split("/", 1)[0] if isinstance(default, str) and "/" in default else None
    model_part = default.split("/", 1)[1] if isinstance(default, str) and "/" in default else None
    stale = bool(prov) and bool(_MANAGED_RE.match(prov)) and (
        prov not in providers or model_part not in (providers.get(prov, {}).get("models") or {})
    )
    if default_model:
        if (default is None or stale) and default != default_model:
            data["model"] = default_model
            dirty = True
    elif stale:
        data.pop("model", None)
        dirty = True

    if dirty:
        OPENCODE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        OPENCODE_CONFIG.write_text(json.dumps(data, indent=2) + "\n")
    return {"added": added, "removed": removed}


def opencode_wired_urls() -> set[str]:
    """Base URLs wired into the managed `openvast` provider (provider-level and
    each model's per-model baseURL) — used to mark wired rows."""
    data = _load_opencode()
    urls = set()
    for k, v in (data.get("provider") or {}).items():
        if not _MANAGED_RE.match(k):
            continue
        url = (v.get("options") or {}).get("baseURL")
        if url:
            urls.add(url)
        for mv in (v.get("models") or {}).values():
            murl = (mv.get("options") or {}).get("baseURL")
            if murl:
                urls.add(murl)
    return urls


# --------------------------------------------------------------------------- #
# SSH log tail (for the start-progress view)
# --------------------------------------------------------------------------- #


def _ssh_target(inst: Instance) -> tuple[str, str] | None:
    try:
        url = _run(["vastai", "ssh-url", str(inst.id)], timeout=30).strip()
    except VastError:
        return None
    m = re.match(r"ssh://root@([^:]+):(\d+)", url)
    return (m.group(1), m.group(2)) if m else None


def _ssh(inst: Instance, remote: str, timeout: int = 25) -> str | None:
    tgt = _ssh_target(inst)
    if not tgt:
        return None
    host, port = tgt
    try:
        proc = subprocess.run(
            [
                "ssh", "-i", SSH_KEY, "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=12",
                "-p", port, f"root@{host}", remote,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.stdout if proc.returncode == 0 else (proc.stdout or proc.stderr)
    except Exception:  # noqa: BLE001
        return None


def ssh_tail_log(inst: Instance, lines: int = 60) -> str:
    remote = (
        "(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null "
        "| sed 's/^/GPU mem used,free: /' || true); "
        f"echo '--- llama.log (last {lines}) ---'; "
        f"tail -n {lines} /root/llama.log 2>/dev/null || echo '(no log yet — still booting)'"
    )
    out = _ssh(inst, remote, timeout=30)
    return (out or "").strip() or "(ssh unreachable — instance may still be booting)"


# --- download progress (% during the model download phase) ------------------ #

_MODEL_BYTES: dict[str, int | None] = {}


def model_download_bytes(model: Model) -> int | None:
    """Approx bytes llama.cpp downloads for this model's quant, from the HF API.

    Sums the .gguf files whose name contains the quant tag (handles sharded
    quants). mmproj is excluded, so actual bytes can slightly exceed this near
    the end — callers should cap the percentage. Cached per hf spec.
    """
    if model.hf in _MODEL_BYTES:
        return _MODEL_BYTES[model.hf]
    total: int | None = None
    try:
        repo, _, quant = model.hf.partition(":")
        url = f"https://huggingface.co/api/models/{repo}?blobs=true"
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)
        tot = sum(
            int(s.get("size") or 0)
            for s in data.get("siblings", [])
            if s.get("rfilename", "").endswith(".gguf")
            and quant and quant.lower() in s["rfilename"].lower()
        )
        total = tot or None
    except Exception:  # noqa: BLE001
        total = None
    _MODEL_BYTES[model.hf] = total
    return total


def fetch_download_pct(inst: Instance) -> float | None:
    """Percentage of the model downloaded on the instance (0–99), or None."""
    total = model_download_bytes(inst.model)
    if not total:
        return None
    repo = inst.model.hf.split(":")[0].replace("/", "--")
    out = _ssh(
        inst,
        f"du -sb /root/.cache/huggingface/hub/models--{repo}/blobs 2>/dev/null | cut -f1",
        timeout=18,
    )
    if not out:
        return None
    digits = out.strip().split()[0] if out.strip() else ""
    if not digits.isdigit():
        return None
    return min(99.0, int(digits) / total * 100.0)


# --------------------------------------------------------------------------- #
# self-test (no UI) — verifies the backend against the live account
# --------------------------------------------------------------------------- #


def selftest() -> int:
    print("== openvast backend self-test ==\n")

    env = check_environment()
    print("Environment:")
    print(f"  vastai installed : {env.vast_installed}")
    print(f"  vastai authed    : {env.vast_authed}")
    print(f"  opencode         : {env.opencode_installed}")
    print(f"  balance (credit) : {'n/a' if env.balance is None else f'${env.balance:.2f}'}")
    print(f"  editing allowed  : {env.can_edit}" + (f"  ({env.reason})" if env.reason else ""))

    print(f"\nModels registered ({MODELS_FILE}):")
    for m in MODELS.values():
        print(f"  - {m.key:24s} {m.name}  (needs {m.min_vram_gb}GB, ctx {m.context})")

    print("\nInstances:")
    insts = list_instances()
    if not insts:
        print("  (none)")
    healthy = []
    for i in insts:
        h = health_ok(i)
        if h:
            healthy.append(i)
        m = fetch_metrics(i) or {}
        toks = m.get("tok_s")
        kvs = m.get("kv_s")
        rate = f"tok/s={toks:.0f} kv/s={kvs:.0f}" if toks is not None else "tok/s=-- kv/s=--"
        print(
            f"  #{i.id}  {i.gpu_name:12s} {i.model.key:20s} "
            f"status={derive_status(i, h):12s} health={'ok' if h else '-':3s} "
            f"${i.dph:.3f}/hr up={i.uptime_str()} {rate}  {i.base_url() or ''}"
        )

    print("\nCost:")
    c = cost_summary(insts)
    fmt = lambda v: "n/a" if v is None else f"${v:.2f}"  # noqa: E731
    print(f"  now ${c['per_hour']:.3f}/hr | 24h {fmt(c['l24h'])} | "
          f"7d {fmt(c['l7d'])} | 30d {fmt(c['l30d'])}")

    print(f"\nOffers that fit '{DEFAULT_MODEL_KEY}':")
    try:
        for o in search_offers(MODELS[DEFAULT_MODEL_KEY], limit=5):
            print(f"  offer {o.id}  {o.gpu_name:12s} {gb(o.gpu_ram_mb)}GB "
                  f"${o.dph:.3f}/hr  cuda {o.cuda}  {o.location}")
    except VastError as e:
        print("  offer search failed:", e)

    print("\nReconciling opencode with healthy instances:", [i.id for i in healthy])
    res = reconcile_opencode(healthy)
    print("  added:", res["added"], " removed:", res["removed"])
    wired_data = _load_opencode()
    print("  opencode providers:", [k for k in (wired_data.get("provider") or {}) if _MANAGED_RE.match(k)])
    print("\nOK")
    return 0


# --------------------------------------------------------------------------- #
# TUI
# --------------------------------------------------------------------------- #


def build_app():
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Center, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.theme import Theme
    from textual.widgets import (
        Button, DataTable, Footer, Header, Label, ListItem, ListView, Log, Static,
    )

    # --- muted, semantic palette (Everforest-derived) -------------------- #
    MONO = bool(os.environ.get("NO_COLOR"))

    def _c(value: str) -> str:
        return "" if MONO else value

    PAL = {
        "healthy": _c("#a7c080"),   # muted green
        "loading": _c("#dbbc7f"),   # muted amber
        "paused":  _c("#859289"),   # muted gray
        "error":   _c("#e67e80"),   # muted red
        "ok":      _c("#a7c080"),
        "off":     _c("#7a8478"),   # dim gray (not alarming)
        "wired":   _c("#7fbbb3"),   # muted teal
        "muted":   _c("#859289"),
    }
    STATUS_STYLE = {
        "healthy": PAL["healthy"],
        "loading": PAL["loading"],
        "starting": PAL["loading"],
        "pulling": PAL["loading"],
        "downloading": PAL["loading"],
        "error": PAL["error"],
        "paused": PAL["paused"],
    }

    VAST_THEME = Theme(
        name="vast-muted",
        primary="#7fbbb3", secondary="#a7c080", accent="#83c092",
        foreground="#d3c6aa", background="#232a2e", surface="#2d353b", panel="#343f44",
        success="#a7c080", warning="#dbbc7f", error="#e67e80", dark=True,
        variables={
            "text-muted": "#859289",
            "footer-key-foreground": "#7fbbb3",
            "footer-description-foreground": "#859289",
        },
    )

    # instance-table columns: key -> (header, width, min terminal width to show)
    # key -> (header, width, min terminal width to show)
    COLUMNS = {
        "id":     ("ID", 9, 0),
        "gpu":    ("GPU", 15, 116),
        "model":  ("Model", 20, 0),
        "status": ("Status", 16, 0),
        "health": ("Health", 8, 80),
        "toks":   ("tok/s", 7, 0),      # generation throughput
        "kvs":    ("kv t/s", 7, 124),   # prefill / KV-cache fill rate
        "dph":    ("$/hr", 8, 0),
        "uptime": ("Up", 7, 96),
        "wired":  ("OC", 4, 96),
    }

    def cols_for_width(w: int) -> list[str]:
        return [k for k, (_h, _w, need) in COLUMNS.items() if w >= need] if w >= 68 \
            else ["id", "model", "status", "toks", "dph"]

    class ConfirmScreen(ModalScreen):
        def __init__(self, question: str, danger: bool = False):
            super().__init__()
            self.question = question
            self.danger = danger

        def compose(self) -> ComposeResult:
            with Vertical(id="confirm-box"):
                yield Label(self.question, id="confirm-q")
                with Horizontal(id="confirm-btns"):
                    yield Button("Yes", id="yes", classes="danger" if self.danger else "")
                    yield Button("No", id="no")
                yield Label("[dim]y / Enter · n / Esc[/]", id="confirm-hint")

        def on_mount(self) -> None:
            self.query_one("#no", Button).focus()   # default to the safe choice

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "yes")

        def on_key(self, event) -> None:
            if event.key in ("escape", "n"):
                event.stop()
                self.dismiss(False)
            elif event.key == "y":
                event.stop()
                self.dismiss(True)

    class ModelSelectScreen(ModalScreen):
        def __init__(self):
            super().__init__()
            self.keys = list(MODELS)

        def compose(self) -> ComposeResult:
            with Vertical(id="select-box"):
                yield Label("Select a model to launch  [dim](Esc to cancel)[/]", id="select-title")
                yield ListView(
                    *[
                        ListItem(Label(f"{MODELS[k].name}   [dim]· {MODELS[k].min_vram_gb}GB[/]"))
                        for k in self.keys
                    ]
                )

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            idx = event.list_view.index
            self.dismiss(self.keys[idx] if idx is not None else None)

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.dismiss(None)

    class OfferSelectScreen(ModalScreen):
        def __init__(self, model_key: str):
            super().__init__()
            self.model = MODELS[model_key]
            self.offers: list[Offer] = []

        def compose(self) -> ComposeResult:
            with Vertical(id="offer-box"):
                yield Label(
                    f"GPUs that fit {self.model.name} (>= {self.model.min_vram_gb}GB).  "
                    f"Enter = launch · Esc = cancel", id="offer-title",
                )
                self.table = DataTable(cursor_type="row")
                self.table.add_columns("offer", "GPU", "VRAM", "$/hr", "cuda", "rel", "location", "net↓")
                yield self.table
                yield Static("Searching offers…", id="offer-status")

        def on_mount(self) -> None:
            self.run_worker(self._load, thread=True)

        def _load(self) -> None:
            try:
                offers = search_offers(self.model, limit=30)
            except VastError as e:
                self.app.call_from_thread(self.query_one("#offer-status", Static).update, f"error: {e}")
                return
            self.offers = offers
            self.app.call_from_thread(self._populate)

        def _populate(self) -> None:
            self.table.clear()
            for o in self.offers:
                self.table.add_row(
                    str(o.id), o.gpu_name, f"{gb(o.gpu_ram_mb)}GB", f"${o.dph:.3f}",
                    o.cuda, f"{o.reliability:.2f}", o.location[:18], str(int(o.inet_down)),
                    key=str(o.id),
                )
            self.query_one("#offer-status", Static).update(
                f"{len(self.offers)} offers · cheapest first. Enter to launch on the selected GPU."
            )
            if self.offers:
                self.table.focus()

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            event.stop()
            self.dismiss((self.model.key, int(event.row_key.value)))

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.dismiss(None)

    class LogScreen(ModalScreen):
        BINDINGS = [Binding("escape", "close", "close")]

        def __init__(self, inst: Instance):
            super().__init__()
            self.inst = inst
            self._stop = False

        def compose(self) -> ComposeResult:
            with Vertical(id="log-box"):
                yield Label(f"Instance {self.inst.id} — {self.inst.model.name}   (Esc to close)")
                self.logw = Log(highlight=False)
                yield self.logw

        def on_mount(self) -> None:
            self._timer = self.set_interval(4, self._refresh)
            self._refresh()

        def action_close(self) -> None:
            self._stop = True
            self.dismiss(None)

        def _refresh(self) -> None:
            self.run_worker(self._refresh_now, thread=True)

        def _refresh_now(self) -> None:
            text = ssh_tail_log(self.inst)
            if not self._stop:
                self.app.call_from_thread(self._set, text)

        def _set(self, text: str) -> None:
            self.logw.clear()
            self.logw.write(text)

    class DetailScreen(ModalScreen):
        """Detail-on-Enter: every field for the selected instance."""
        BINDINGS = [
            Binding("escape", "close", "close"),
            Binding("enter", "close", "close"),
        ]

        def __init__(self, inst: Instance, healthy: bool):
            super().__init__()
            self.inst = inst
            self.healthy = healthy

        def compose(self) -> ComposeResult:
            i, h = self.inst, self.healthy
            wired = i.base_url() in opencode_wired_urls()
            rows = [
                ("model", i.model.key),
                ("gpu", f"{i.gpu_name}  {gb(i.gpu_ram_mb)}GB x {i.num_gpus}"),
                ("status", derive_status(i, h)),
                ("health", "healthy" if h else "not responding"),
                ("$/hr", f"${i.dph:.3f}"),
                ("uptime", i.uptime_str()),
                ("endpoint", i.base_url() or "-"),
                ("opencode", "wired · openvast" if wired else "not wired"),
                ("context", str(i.model.context)),
                ("image", i.model.image),
            ]
            if i.status_msg:
                rows.append(("vast msg", i.status_msg))
            body = f"[b]Instance {i.id}[/]  ·  {i.model.name}\n\n" + "\n".join(
                f"[dim]{k:<9}[/] {v}" for k, v in rows
            ) + "\n\n[dim]Esc / Enter to close[/]"
            with Center():
                with Vertical(id="detail-box"):
                    yield Static(body)

        def action_close(self) -> None:
            self.dismiss(None)

    class VastTUI(App):
        CSS = """
        Screen { background: $background; }
        #costbar { height: 1; padding: 0 1; background: $surface; color: $text-muted; }
        #insttable { height: 1fr; background: $background; }
        #insttable > .datatable--header { background: $surface; color: $text-muted; text-style: none; }
        #insttable > .datatable--cursor { background: $panel; color: $text; text-style: bold; }
        #toosmall { display: none; width: 100%; height: 1fr; content-align: center middle; color: $warning; }
        #banner { display: none; height: 1; padding: 0 1; background: $surface; color: $warning; }
        ConfirmScreen { align: center middle; }
        ModelSelectScreen { align: center middle; }
        #confirm-box { width: auto; max-width: 60; height: auto; padding: 1 3;
                       border: round $primary; background: $surface; }
        #confirm-q { width: 100%; content-align: center middle; text-align: center; margin-bottom: 1; }
        #confirm-hint { width: 100%; text-align: center; margin-top: 1; }
        #confirm-btns { width: auto; height: auto; align: center middle; }
        #confirm-btns Button {
            min-width: 8; height: 3; margin: 0 1; border: round $panel;
            background: $surface; color: $text-muted; text-style: none;
        }
        #confirm-btns Button:focus { background: $panel; color: $text; text-style: bold; }
        #confirm-btns Button.danger { color: $error; border: round $error; }
        #select-box { width: 64; max-height: 80%; height: auto; padding: 1 2;
                      border: round $primary; background: $surface; }
        #detail-box { width: 74; padding: 1 2; border: round $primary; background: $surface; }
        #offer-box { width: 100%; height: 100%; padding: 1 2; }
        #offer-box DataTable { height: 1fr; }
        #log-box { width: 100%; height: 100%; padding: 1 1; }
        #offer-title, #offer-status { color: $text-muted; }
        """
        BINDINGS = [
            Binding("n", "new", "Launch"),
            Binding("p", "pause", "Pause"),
            Binding("r", "resume", "Resume"),
            Binding("d", "delete", "Delete"),
            Binding("enter", "details", "Details"),
            Binding("l", "logs", "Logs"),
            Binding("q", "quit", "Quit"),
        ]

        def __init__(self):
            super().__init__()
            self.instances: list[Instance] = []
            self.health: dict[int, bool] = {}
            self.stages: dict[int, str] = {}
            self.progress: dict[int, float | None] = {}
            self.metrics: dict[int, dict] = {}
            # when each instance entered its current loading spell (reset on
            # resume), so the timer counts from 0 rather than vast's start_date
            self._load_since: dict[int, float] = {}
            self._first_poll = True
            self._wired: set[str] = set()
            self._active_cols: list[str] | None = None
            self._last_cost: dict | None = None
            self._balance: float | None = None
            self._modal = False
            self.can_edit = False
            self._opencode_ok = False
            self._edit_reason = "checking environment…"

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static("", id="costbar")
            yield Static("", id="banner")
            self.table = DataTable(cursor_type="row", id="insttable", zebra_stripes=not MONO)
            yield self.table
            yield Static("Terminal too small — resize to at least 68x12", id="toosmall")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "openvast"
            self.sub_title = "vast.ai · llama.cpp · opencode"
            if not MONO:
                self.register_theme(VAST_THEME)
                self.theme = "vast-muted"
            self._relayout()
            # auto-refresh the table every minute (no manual refresh key)...
            self.set_interval(60, self.refresh_instances)
            # ...but re-poll every 10s while loading, and re-render the elapsed
            # timers every 1s so the seconds tick smoothly (never stuck).
            self.set_interval(10, self.refresh_if_loading)
            self.set_interval(1, self._tick)
            self.set_interval(60, self.refresh_cost)
            self.set_interval(180, self.refresh_env)
            self.refresh_env()
            self.refresh_instances()
            self.refresh_cost()

        def _is_loading(self) -> bool:
            return any(
                i.is_running and not i.is_paused and not self.health.get(i.id)
                for i in self.instances
            )

        def refresh_if_loading(self) -> None:
            if self._is_loading():
                self.refresh_instances()

        def _tick(self) -> None:
            # cheap in-memory re-render so "loading M:SS" advances every second
            if self._is_loading():
                self._render_rows()

        # ---------- environment / prerequisites ----------
        def refresh_env(self) -> None:
            self.run_worker(self._poll_env, thread=True, group="env")

        def _poll_env(self) -> None:
            env = check_environment()
            self.call_from_thread(self._apply_env, env)

        def _apply_env(self, env: EnvCheck) -> None:
            self.can_edit = env.can_edit
            self._opencode_ok = env.opencode_installed
            self._edit_reason = env.reason
            self._balance = env.balance
            banner = self.query_one("#banner", Static)
            if env.can_edit:
                banner.display = False
            else:
                banner.display = True
                banner.update(f"⚠ editing disabled — {env.reason}")
            self._update_costbar()

        # ---------- responsive layout ----------
        def on_resize(self, event) -> None:
            self._relayout()

        def _relayout(self) -> None:
            small = self.size.width < 68 or self.size.height < 12
            try:
                self.query_one("#toosmall").display = small
                self.query_one("#insttable").display = not small
                self.query_one("#costbar").display = not small
            except Exception:  # noqa: BLE001
                return
            if not small:
                self._ensure_columns()

        def _ensure_columns(self) -> None:
            active = cols_for_width(self.size.width)
            if active == self._active_cols:
                return
            self._active_cols = active
            self.table.clear(columns=True)
            for k in active:
                header, width, _ = COLUMNS[k]
                self.table.add_column(header, key=k, width=width)
            self._render_rows()

        # ---------- refresh loops ----------
        def refresh_instances(self) -> None:
            self.run_worker(self._poll_instances, thread=True, exclusive=True, group="poll")

        def _poll_instances(self) -> None:
            try:
                insts = list_instances()
            except VastError as e:
                self.call_from_thread(self._notify, f"list error: {e}", "error")
                return
            probes = {i.id: probe(i) for i in insts}
            health = {iid: p[0] for iid, p in probes.items()}
            stages = {iid: p[1] for iid, p in probes.items()}
            metrics = {
                i.id: (fetch_metrics(i) or {})
                for i in insts
                if health.get(i.id)
            }
            # download-phase % (SSH); only while the server hasn't bound yet
            progress = {}
            for i in insts:
                if (i.is_running and not i.is_paused and not health.get(i.id)
                        and stages.get(i.id) == "" and i.host_port() is not None):
                    progress[i.id] = fetch_download_pct(i)
            healthy = [i for i in insts if health.get(i.id)]
            # only touch opencode's config when opencode is actually installed
            res = {"added": [], "removed": []}
            if self._opencode_ok:
                try:
                    res = reconcile_opencode(healthy)
                except Exception as e:  # noqa: BLE001
                    self.call_from_thread(self._notify, f"opencode: {e}", "warning")
            self.call_from_thread(
                self._apply_instances, insts, health, stages, progress, metrics, res
            )

        def _apply_instances(self, insts, health, stages, progress, metrics, res) -> None:
            self.instances = insts
            self.health = health
            self.stages = stages
            self.progress = progress
            self.metrics = metrics
            self._wired = opencode_wired_urls()

            # track the current loading spell per instance so the timer resets
            # to 0 on resume (vast's start_date keeps the original launch time)
            now = time.time()
            load_since = {}
            for i in insts:
                loading = i.is_running and not i.is_paused and not health.get(i.id)
                if not loading:
                    continue
                if i.id in self._load_since:
                    load_since[i.id] = self._load_since[i.id]        # still loading → keep
                elif self._first_poll and i.start_date:
                    load_since[i.id] = i.start_date                  # app opened mid-load
                else:
                    load_since[i.id] = now                           # just entered loading (resume)
            self._load_since = load_since
            self._first_poll = False

            self._render_rows()
            self._update_costbar()
            for key, sev in (("added", "+"), ("removed", "-")):
                if res.get(key):
                    self._notify(f"opencode {sev}{', '.join(res[key])}", "information")

        def _cells(self, i: Instance, h: bool) -> dict:
            status = derive_status(i, h, self.stages.get(i.id, ""))
            label = status
            if status in ("loading", "starting", "pulling", "downloading"):
                pct = self.progress.get(i.id)
                since = self._load_since.get(i.id)
                if status == "downloading" and pct is not None:
                    label = f"downloading {pct:.0f}%"     # real download progress
                elif since is not None:
                    label = f"{status} {dur(time.time() - since)}"  # elapsed since this load
            m = self.metrics.get(i.id) or {}

            def rate(v):
                return Text("—", style=PAL["off"]) if v is None else Text(f"{v:.0f}", style=PAL["muted"])

            return {
                "id": Text(str(i.id)),
                "gpu": Text(f"{i.gpu_name} {gb(i.gpu_ram_mb)}G", style=PAL["muted"]),
                "model": Text(i.model.key),
                "status": Text(label, style=STATUS_STYLE.get(status, PAL["muted"])),
                "health": Text("● ok", style=PAL["ok"]) if h else Text("○ —", style=PAL["off"]),
                "toks": rate(m.get("tok_s")),
                "kvs": rate(m.get("kv_s")),
                "dph": Text(f"${i.dph:.3f}"),
                "uptime": Text(i.uptime_str(), style=PAL["muted"]),
                "wired": Text("✓", style=PAL["wired"]) if i.base_url() in self._wired
                else Text("·", style=PAL["off"]),
            }

        def _render_rows(self) -> None:
            if self._active_cols is None:
                return
            sel = self._selected_id()
            self.table.clear()
            for i in self.instances:
                cells = self._cells(i, self.health.get(i.id, False))
                self.table.add_row(*[cells[k] for k in self._active_cols], key=str(i.id))
            if sel is not None:
                try:
                    self.table.move_cursor(row=self._row_index(sel))
                except Exception:  # noqa: BLE001
                    pass

        def refresh_cost(self) -> None:
            self.run_worker(self._poll_cost, thread=True, group="cost")

        def _poll_cost(self) -> None:
            c = cost_summary(self.instances)
            bal = None
            try:
                bal = get_balance()
            except VastError:
                pass
            self.call_from_thread(self._apply_cost, c, bal)

        def _apply_cost(self, c, bal) -> None:
            self._last_cost = c
            if bal is not None:
                self._balance = bal
            self._update_costbar()

        def _update_costbar(self) -> None:
            per_hour = sum(i.dph for i in self.instances if i.is_running)
            running = sum(1 for i in self.instances if i.is_running)
            w = self._last_cost or {}
            fmt = lambda v: "—" if v is None else f"${v:.2f}"  # noqa: E731
            bal = "—" if self._balance is None else f"${self._balance:.2f}"
            hint = "" if self.instances else "     [dim]no instances · press n to launch[/]"
            try:
                self.query_one("#costbar", Static).update(
                    f"[dim]balance[/] [b]{bal}[/]    [dim]running[/] {running}    "
                    f"[dim]now[/] [b]${per_hour:.3f}/hr[/]"
                    f"     [dim]24h[/] {fmt(w.get('l24h'))}   [dim]7d[/] {fmt(w.get('l7d'))}"
                    f"   [dim]30d[/] {fmt(w.get('l30d'))}{hint}"
                )
            except Exception:  # noqa: BLE001
                pass

        # ---------- helpers ----------
        def _selected_id(self) -> int | None:
            if self.table.row_count == 0:
                return None
            try:
                cell = self.table.coordinate_to_cell_key(self.table.cursor_coordinate)
                return int(cell.row_key.value)
            except Exception:  # noqa: BLE001
                return None

        def _row_index(self, instance_id: int) -> int:
            for idx, i in enumerate(self.instances):
                if i.id == instance_id:
                    return idx
            return 0

        def _current(self) -> Instance | None:
            sid = self._selected_id()
            return next((i for i in self.instances if i.id == sid), None)

        def _notify(self, msg: str, severity: str = "information") -> None:
            try:
                self.notify(msg, severity=severity, timeout=6)
            except Exception:  # noqa: BLE001
                pass

        # ---------- actions ----------
        def _guard_edit(self) -> bool:
            if not self.can_edit:
                self._notify(f"editing disabled — {self._edit_reason}", "warning")
                return False
            return True

        def action_new(self) -> None:
            if not self._guard_edit():
                return

            def after_model(model_key):
                if not model_key:
                    return

                def after_offer(choice):
                    if not choice:
                        return
                    mkey, offer_id = choice
                    self._notify(f"launching {mkey} on offer {offer_id}…")
                    self.run_worker(lambda: self._do_create(offer_id, mkey), thread=True)

                self.push_screen(OfferSelectScreen(model_key), after_offer)

            self.push_screen(ModelSelectScreen(), after_model)

        def _do_create(self, offer_id: int, model_key: str) -> None:
            try:
                new_id = create_instance(offer_id, MODELS[model_key])
                self.call_from_thread(self._notify, f"created instance {new_id}", "information")
            except VastError as e:
                self.call_from_thread(self._notify, f"create failed: {e}", "error")
            self.call_from_thread(self.refresh_instances)

        def action_pause(self) -> None:
            if not self._guard_edit():
                return
            inst = self._current()
            if not inst:
                return

            def go(ok):
                if ok:
                    self.run_worker(lambda: self._simple_op(stop_instance, inst.id, "paused"), thread=True)

            self.push_screen(ConfirmScreen(f"Pause (stop) instance {inst.id}?"), go)

        def action_resume(self) -> None:
            if not self._guard_edit():
                return
            inst = self._current()
            if inst:
                self.run_worker(lambda: self._simple_op(start_instance, inst.id, "resuming"), thread=True)

        def action_delete(self) -> None:
            if not self._guard_edit():
                return
            inst = self._current()
            if not inst:
                return

            def go(ok):
                if ok:
                    self.run_worker(lambda: self._simple_op(destroy_instance, inst.id, "deleted"), thread=True)

            self.push_screen(
                ConfirmScreen(f"Delete (destroy) instance {inst.id}?\nThis is permanent.", danger=True), go
            )

        def _simple_op(self, fn, instance_id: int, verb: str) -> None:
            try:
                fn(instance_id)
                self.call_from_thread(self._notify, f"{verb} {instance_id}", "information")
            except VastError as e:
                self.call_from_thread(self._notify, f"{verb} failed: {e}", "error")
            self.call_from_thread(self.refresh_instances)

        def action_logs(self) -> None:
            inst = self._current()
            if inst:
                self.push_screen(LogScreen(inst))

        def action_details(self) -> None:
            inst = self._current()
            if not inst or self._modal:
                return
            self._modal = True
            self.push_screen(
                DetailScreen(inst, self.health.get(inst.id, False)),
                lambda _res: setattr(self, "_modal", False),
            )

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            if event.data_table is self.table:
                self.action_details()

    return VastTUI


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="vast.ai + llama.cpp + opencode TUI")
    ap.add_argument("--selftest", action="store_true", help="run backend checks, no UI")
    ap.add_argument("--models", action="store_true", help="print model registry and exit")
    args = ap.parse_args()

    if args.models:
        for m in MODELS.values():
            print(json.dumps(m.__dict__, default=str))
        return 0
    if args.selftest:
        return selftest()

    # Monitoring is impossible without the vast CLI; fail fast with guidance.
    if shutil.which("vastai") is None:
        print("vastai CLI not found. Install it with:  python3 -m pip install vastai",
              file=sys.stderr)
        return 1

    try:
        App = build_app()
    except ModuleNotFoundError:
        print("textual is not installed. Run:  python3 -m pip install textual", file=sys.stderr)
        return 1
    App().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
