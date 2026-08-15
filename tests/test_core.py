"""
Unit tests for openvast's pure/deterministic logic.

These cover the functions that don't require a live vast.ai account,
a GPU, or opencode installed — so they're safe to run in CI on every
push/PR. Anything that shells out to `vastai` or hits a live model
endpoint (search_offers, create_instance, ssh, fetch_metrics, ...) is
out of scope here and stays covered by the manual `openvast --selftest`
flow described in the README.
"""
from __future__ import annotations

import time

import pytest

from openvast import (
    Instance,
    Model,
    _model_from_dict,
    cost_summary,
    derive_status,
    dur,
    gb,
    load_models,
    min_vram_mb,
    model_for_label,
    offer_query,
)


# --------------------------------------------------------------------------- #
# dur()
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "secs, expected",
    [
        (0, "0:00"),
        (5, "0:05"),
        (59, "0:59"),
        (60, "1:00"),
        (125, "2:05"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (3661, "1:01:01"),
        (90061, "25:01:01"),
        (-5, "0:00"),  # negative durations clamp to zero
    ],
)
def test_dur(secs, expected):
    assert dur(secs) == expected


# --------------------------------------------------------------------------- #
# gb() / min_vram_mb()
# --------------------------------------------------------------------------- #

def test_gb_rounds_to_nearest():
    assert gb(1024) == 1
    assert gb(24564) == 24  # real-world reported VRAM for a "24GB" card
    assert gb(0) == 0


def test_min_vram_mb_allows_headroom_under_nominal():
    model = Model(key="m", name="M", hf="x", min_vram_gb=24, disk_gb=80, context=8192)
    # A "24GB" card reports ~24564MB, not a full 24576MB (24*1024) — the
    # -1GB fudge factor must not reject those real-world cards.
    assert min_vram_mb(model) == 24 * 1024 - 1024
    assert gb(24564) >= gb(min_vram_mb(model))


# --------------------------------------------------------------------------- #
# offer_query()
# --------------------------------------------------------------------------- #

def test_offer_query_includes_model_vram_floor():
    model = Model(key="m", name="M", hf="x", min_vram_gb=32, disk_gb=80, context=8192)
    q = offer_query(model)
    assert "gpu_ram>=32" in q
    assert "num_gpus=1" in q
    assert "verified=true" in q
    assert "rentable=true" in q


# --------------------------------------------------------------------------- #
# _model_from_dict() / load_models()
# --------------------------------------------------------------------------- #

def test_model_from_dict_applies_defaults_and_overrides():
    d = {
        "key": "test-model",
        "name": "Test Model",
        "hf": "org/repo:Q4",
        "min_vram_gb": 24,
        "reasoning": False,
    }
    m = _model_from_dict(d)
    assert m.key == "test-model"
    assert m.min_vram_gb == 24
    assert m.reasoning is False
    assert m.tool_call is True          # class default preserved
    assert m.output_limit == 8192       # class default preserved


def test_model_from_dict_missing_required_field_raises():
    with pytest.raises(KeyError):
        _model_from_dict({"key": "no-hf", "name": "N", "min_vram_gb": 16})


def test_load_models_falls_back_when_file_missing(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    monkeypatch.setattr("openvast.MODELS_FILE", missing)
    models, default_key = load_models()
    assert default_key in models
    assert len(models) >= 1


def test_load_models_skips_bad_entries_but_keeps_good_ones(monkeypatch, tmp_path):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        """
defaults:
  disk_gb: 80
  context: 8192
default_model: good-model
models:
  - key: good-model
    name: Good Model
    hf: org/good:Q4
    min_vram_gb: 24
  - key: bad-model
    name: Missing HF field
    min_vram_gb: 16
"""
    )
    monkeypatch.setattr("openvast.MODELS_FILE", yaml_path)
    models, default_key = load_models()
    assert "good-model" in models
    assert "bad-model" not in models
    assert default_key == "good-model"


# --------------------------------------------------------------------------- #
# model_for_label()
# --------------------------------------------------------------------------- #

def test_model_for_label_unknown_falls_back_to_default():
    from openvast import DEFAULT_MODEL_KEY, MODELS
    result = model_for_label("totally-unknown-label")
    assert result is MODELS[DEFAULT_MODEL_KEY]


def test_model_for_label_none_falls_back_to_default():
    from openvast import DEFAULT_MODEL_KEY, MODELS
    assert model_for_label(None) is MODELS[DEFAULT_MODEL_KEY]


# --------------------------------------------------------------------------- #
# derive_status()
# --------------------------------------------------------------------------- #

def _make_instance(**overrides) -> Instance:
    base = dict(
        id=1,
        label="qwen3.6-35b-a3b",
        gpu_name="RTX 4090",
        num_gpus=1,
        gpu_ram_mb=24564,
        dph=0.34,
        actual_status="running",
        cur_state="running",
        intended_status="running",
        public_ip="1.2.3.4",
        ports={},
        start_date=time.time(),
        status_msg="",
    )
    base.update(overrides)
    return Instance(**base)


def test_derive_status_error_takes_priority_over_everything():
    inst = _make_instance(status_msg="Error: no such container", intended_status="stopped")
    assert derive_status(inst, healthy=True) == "error"


def test_derive_status_paused_when_stopped_and_not_erroring():
    inst = _make_instance(intended_status="stopped", cur_state="exited")
    assert derive_status(inst, healthy=False) == "paused"


def test_derive_status_healthy_when_probe_succeeds():
    inst = _make_instance()
    assert derive_status(inst, healthy=True) == "healthy"


def test_derive_status_pulling_when_image_not_cached():
    inst = _make_instance(
        actual_status="running",
        cur_state="running",
        ports={},
        status_msg="pulling image layer 3/9",
    )
    assert derive_status(inst, healthy=False) == "pulling"


def test_derive_status_downloading_when_port_mapped_but_not_healthy():
    inst = _make_instance(
        ports={"18000/tcp": [{"HostPort": "40213"}]},
        status_msg="",
    )
    assert derive_status(inst, healthy=False) == "downloading"


# --------------------------------------------------------------------------- #
# Instance helpers
# --------------------------------------------------------------------------- #

def test_instance_host_port_parses_docker_port_mapping():
    inst = _make_instance(ports={"18000/tcp": [{"HostPort": "40213"}]})
    assert inst.host_port() == 40213


def test_instance_host_port_none_when_unmapped():
    inst = _make_instance(ports={})
    assert inst.host_port() is None


def test_instance_base_url_requires_ip_and_port():
    inst = _make_instance(public_ip="5.6.7.8", ports={"18000/tcp": [{"HostPort": "40213"}]})
    assert inst.base_url() == "http://5.6.7.8:40213/v1"

    inst_no_ip = _make_instance(public_ip=None, ports={"18000/tcp": [{"HostPort": "40213"}]})
    assert inst_no_ip.base_url() is None


def test_instance_is_running_false_if_intended_stopped():
    inst = _make_instance(actual_status="running", cur_state="running", intended_status="stopped")
    assert inst.is_running is False


def test_instance_is_paused_true_when_exited():
    inst = _make_instance(actual_status="exited")
    assert inst.is_paused is True


# --------------------------------------------------------------------------- #
# cost_summary() — per_hour math shouldn't need a live vast.ai account
# --------------------------------------------------------------------------- #

def test_cost_summary_sums_only_running_instances():
    running = _make_instance(id=1, dph=0.34, actual_status="running", cur_state="running", intended_status="running")
    paused = _make_instance(id=2, dph=0.50, actual_status="exited", cur_state="exited", intended_status="stopped")
    result = cost_summary([running, paused])
    assert result["per_hour"] == pytest.approx(0.34)


def test_cost_summary_handles_no_vastai_cli_gracefully():
    # In CI there's no `vastai` CLI / auth, so the invoice lookup should
    # fail closed (None) rather than raising.
    result = cost_summary([])
    assert result["per_hour"] == 0
    assert result["l24h"] is None or isinstance(result["l24h"], float)
