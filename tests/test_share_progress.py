# SPDX-License-Identifier: Apache-2.0
"""Pool-only signs of life without model downloads or wall-clock sleeps."""

import http.client
import io
import threading
import urllib.error
from email.message import Message
from unittest.mock import MagicMock

import pytest

from rapid_mlx import _download_gate as gate
from rapid_mlx.share import cli, ws_tunnel
from rapid_mlx.share import quicksilver as qs


def metrics_response(body):
    response = io.BytesIO(body)
    response.status = 200
    response.headers = Message()
    response.headers["Content-Type"] = "text/plain; version=0.0.4"
    return response


def metrics(requests=0, tokens=0, speed_sum=0, speed_count=0):
    return (
        f'rapid_mlx_model_requests_total{{model="private model",outcome="succeeded"}} {requests}\n'
        'rapid_mlx_model_requests_total{model="private model",outcome="failed"} 90\n'
        f'rapid_mlx_model_completion_tokens_total{{model="private model"}} {tokens}\n'
        f'rapid_mlx_model_decode_tokens_per_second_sum{{model="private model"}} {speed_sum}\n'
        f'rapid_mlx_model_decode_tokens_per_second_count{{model="private model"}} {speed_count}\n'
    )


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(qs.time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def status(monkeypatch, clock):
    snapshot = [metrics(1, 8, 40, 1)]  # Local warm-up.
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(snapshot[0].encode())
    )
    return qs._PoolStatus(8765, lambda: True), snapshot


def test_status_first_request_and_throttle(status, clock, capsys):
    reporter, snapshot = status
    reporter.poll(True)
    assert capsys.readouterr().err == ""
    snapshot[0] = metrics(2, 9, 40, 1)
    clock[0] = 110
    reporter.poll(True)
    assert capsys.readouterr().err == (
        "QuickSilver status: 1 request served, 1 output token since start, "
        "average 0.0 tok/s, connected.\n"
    )
    snapshot[0] = metrics(13, 1848, 82.6, 3)
    for second in range(120, 170, 10):
        clock[0] = second
        reporter.poll(True)
        assert capsys.readouterr().err == ""
    clock[0] = 170
    reporter.poll(True)
    assert capsys.readouterr().err == (
        "QuickSilver status: 12 requests served, 1,840 output tokens since start, "
        "average 21.3 tok/s, connected.\n"
    )
    for second in range(180, 310, 10):
        clock[0] = second
        reporter.poll(True)
    assert capsys.readouterr().err == ""
    assert reporter.last_print == 170  # Idle polls never advance the throttle.


def test_status_disconnect_and_reconnect_are_coalesced(status, clock, capsys):
    reporter, snapshot = status
    snapshot[0] = metrics(1001, 2008, 60, 2)
    reporter.poll(True)
    assert "1,000 requests served, 2,000 output tokens" in capsys.readouterr().err
    clock[0] = 110
    reporter.poll(False)
    assert capsys.readouterr().err == ""
    clock[0] = 160
    reporter.poll(False)
    assert "disconnected, reconnecting." in capsys.readouterr().err
    clock[0] = 220
    reporter.poll(True)
    assert "connected." in capsys.readouterr().err


def test_status_connection_changes_with_unchanged_totals_share_throttle(
    status, clock, capsys
):
    reporter, snapshot = status
    snapshot[0] = metrics(2, 18, 60, 2)
    reporter.poll(True)
    connected = capsys.readouterr().err
    clock[0] = 159
    reporter.poll(False)
    assert capsys.readouterr().err == ""
    clock[0] = 169
    reporter.poll(False)
    assert capsys.readouterr().err == connected.replace(
        "connected.", "disconnected, reconnecting."
    )
    clock[0] = 228
    reporter.poll(True)
    assert capsys.readouterr().err == ""
    clock[0] = 238
    reporter.poll(True)
    assert capsys.readouterr().err == connected
    clock[0] = 298
    reporter.poll(True)
    assert capsys.readouterr().err == ""


def test_status_late_warmup_accounting_before_pool_traffic(monkeypatch, clock, capsys):
    snapshot = [metrics()]
    monkeypatch.setattr(qs, "_open", MagicMock(side_effect=TimeoutError()))
    qs._warmup(8765, "local-key", "local-model")
    assert "warm-up failed" in capsys.readouterr().err
    monkeypatch.setattr(
        qs, "_open", lambda *a, **kw: metrics_response(snapshot[0].encode())
    )
    started = threading.Event()
    reporter = qs._PoolStatus(8765, lambda: True, requests_started_fn=started.is_set)
    reporter.poll(True)
    snapshot[0] = metrics(1, 8, 40, 1)  # Server accounts after client timeout.
    clock[0] = 110
    reporter.poll(True)
    assert reporter.baseline == (1, 8, 40, 1)
    assert reporter.baseline_at_start
    assert capsys.readouterr().err == ""
    started.set()
    snapshot[0] = metrics(2, 18, 60, 2)
    clock[0] = 120
    reporter.poll(True)
    assert "1 request served, 10 output tokens since start" in capsys.readouterr().err
    clock[0] = 180
    reporter.poll(False)  # Reconnecting must retain the pool baseline.
    assert reporter.baseline == (1, 8, 40, 1)
    assert "10 output tokens since start" in capsys.readouterr().err


def test_status_does_not_rebaseline_traffic_arriving_during_scrape(
    monkeypatch, clock, capsys
):
    monkeypatch.setattr(qs._PoolStatus, "_read", lambda self: (1, 8, 40, 1))
    started = threading.Event()
    reporter = qs._PoolStatus(8765, lambda: True, requests_started_fn=started.is_set)

    def read():
        started.set()
        return (2, 18, 60, 2)

    monkeypatch.setattr(reporter, "_read", read)
    reporter.poll(True)
    assert reporter.baseline == (1, 8, 40, 1)
    assert capsys.readouterr().err == ""
    clock[0] += 10
    reporter.poll(True)
    assert "1 request served, 10 output tokens" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_tunnel_request_signal_is_shared_across_reconnects():
    started = threading.Event()
    client = ws_tunnel.TunnelClient(local_port=1, requests_started_event=started)
    client._dispatch_inbound({"t": "abort", "id": "absent"})
    assert not started.is_set()

    async def handle(msg):
        assert started.is_set()  # Set before the request can reach serve.

    client._handle_request = handle
    client._dispatch_inbound({"t": "req", "id": "first"})
    assert started.is_set()
    import asyncio

    await asyncio.gather(*client._tasks)
    reconnected = ws_tunnel.TunnelClient(local_port=1, requests_started_event=started)
    assert reconnected._requests_started.is_set()


def test_status_no_empty_node_announcement(status, capsys):
    reporter, _ = status
    reporter.poll(False)
    assert capsys.readouterr().err == ""


def test_status_polling_is_cheap(status, clock, monkeypatch):
    reporter, _ = status
    read = MagicMock(return_value=reporter.baseline)
    monkeypatch.setattr(reporter, "_read", read)
    for now in (100, 101, 105, 109.9, 110):
        clock[0] = now
        reporter.poll(True)
    assert read.call_count == 2


def test_status_failure_does_not_advance_print_throttle(
    status, clock, monkeypatch, capsys
):
    reporter, snapshot = status
    snapshot[0] = metrics(2, 18, 60, 2)
    reporter.poll(True)
    capsys.readouterr()
    clock[0] = 160
    with monkeypatch.context() as patch:
        patch.setattr(reporter, "_read", lambda: None)
        reporter.poll(True)
    assert reporter.last_print == 100
    assert capsys.readouterr().err == ""
    snapshot[0] = metrics(3, 28, 80, 3)
    clock[0] = 170
    reporter.poll(True)
    assert "2 requests served, 20 output tokens" in capsys.readouterr().err


def test_status_throttle_includes_scrape_latency(status, clock, monkeypatch, capsys):
    reporter, _ = status

    def slow_read():
        clock[0] += 1
        return (2, 18, 60, 2)

    monkeypatch.setattr(reporter, "_read", slow_read)
    reporter.poll(True)
    assert reporter.last_print == 101
    capsys.readouterr()
    monkeypatch.setattr(reporter, "_read", lambda: (3, 28, 80, 3))
    clock[0] = 160  # 60 seconds since scrape began, 59 since emission.
    reporter.poll(True)
    assert capsys.readouterr().err == ""
    clock[0] = 170
    reporter.poll(True)
    assert "2 requests served" in capsys.readouterr().err


def test_status_parse_labels_whitespace_timestamps_and_multiple_models():
    raw = metrics(2, 20, 50, 2) + metrics(3, 30, 100, 3)
    raw = raw.replace(
        'model="private model",outcome="succeeded"',
        'outcome="succeeded", model="escaped \\" label"',
    )
    raw = raw.replace("} 2\n", "}\t2 1234\n")
    raw += "# HELP ignored\ninvalid exposition\nother_metric NaN\n"
    assert qs._PoolStatus._parse(raw) == (5, 50, 150, 5)


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf", "-1", "invalid"])
def test_status_parse_rejects_invalid_samples(value):
    assert qs._PoolStatus._parse(metrics().replace("} 0\n", f"}} {value}\n")) is None


def test_status_parse_requires_all_series():
    assert qs._PoolStatus._parse("\n".join(metrics().splitlines()[:-1])) is None
    assert qs._PoolStatus._parse("") is None


def test_status_parse_rejects_aggregate_overflow():
    assert qs._PoolStatus._parse(metrics(1e308) + metrics(1e308)) is None


def test_status_parses_existing_performance_ledger_exposition():
    from rapid_mlx.routes.metrics import _render_model_performance
    from rapid_mlx.runtime.model_performance import ModelPerformanceLedger

    ledger = ModelPerformanceLedger("local model")
    ledger.record_success(
        "local-id",
        prompt_tokens=3,
        completion_tokens=10,
        ttft_seconds=0.1,
        decode_tokens_per_second=21.3,
    )
    ledger.record_failure("failed-id", completion_tokens=2)
    raw = "\n".join(
        _render_model_performance({"model_performance": ledger.snapshot().__dict__})
    )
    # Requests count successes; token aggregates include partial failed work.
    assert qs._PoolStatus._parse(raw) == (1, 12, 21.3, 1)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("private error"),
        TimeoutError(),
        ConnectionError(),
        http.client.IncompleteRead(b"secret"),
        urllib.error.HTTPError("local", 503, "secret", {}, None),
    ],
)
def test_status_metrics_failure_is_silent(monkeypatch, clock, capsys, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(qs, "_open", fail)
    reporter = qs._PoolStatus(8765, lambda: True)
    reporter.poll(True)
    assert reporter.baseline is None
    assert capsys.readouterr() == ("", "")


def test_status_metrics_http_error_closes_response(monkeypatch):
    body = io.BytesIO(b"private error")
    error = urllib.error.HTTPError("local", 503, "private error", {}, body)

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(qs, "_open", fail)
    assert qs._PoolStatus(8765, lambda: True).baseline is None
    assert body.closed


@pytest.mark.parametrize("body", [b"x" * (256 * 1024 + 1), b"\xff"])
def test_status_bounded_read_and_invalid_encoding(monkeypatch, body):
    monkeypatch.setattr(qs, "_open", lambda *a, **kw: metrics_response(body))
    assert qs._PoolStatus(8765, lambda: True).baseline is None


def test_status_late_baseline_excludes_warmup_and_labels_window(
    monkeypatch, clock, capsys
):
    monkeypatch.setattr(qs._PoolStatus, "_read", lambda self: None)
    reporter = qs._PoolStatus(8765, lambda: True)
    monkeypatch.setattr(reporter, "_read", lambda: (1, 8, 40, 1))
    reporter.poll(True)
    assert capsys.readouterr().err == ""
    clock[0] += 10
    monkeypatch.setattr(reporter, "_read", lambda: (2, 18, 60, 2))
    reporter.poll(True)
    assert "10 output tokens since metrics became available" in capsys.readouterr().err


def test_status_stop_during_scrape_does_not_print(status, monkeypatch, capsys):
    reporter, _ = status
    stop = threading.Event()
    reporter._stop = stop

    def read():
        stop.set()
        return (2, 18, 60, 2)

    monkeypatch.setattr(reporter, "_read", read)
    reporter.poll(True)
    assert capsys.readouterr().err == ""
    reporter._run()  # Already cancelled: no further scrape or sleep.


def test_status_thread_polls_every_ten_seconds_and_stops(status, monkeypatch):
    reporter, _ = status
    stop = MagicMock()
    stop.wait.side_effect = [False, False, True]
    reporter._stop = stop
    poll = MagicMock()
    monkeypatch.setattr(reporter, "poll", poll)
    reporter._run()
    assert [c.args for c in stop.wait.call_args_list] == [(10,), (10,), (10,)]
    assert [c.args for c in poll.call_args_list] == [(True,), (True,)]


def test_status_start_is_daemon_with_heartbeat_cancellation(status, monkeypatch):
    reporter, _ = status
    factory = MagicMock()
    monkeypatch.setattr(qs.threading, "Thread", factory)
    stop = threading.Event()
    reporter.start(stop)
    assert reporter._stop is stop
    assert factory.call_args.kwargs["daemon"] is True
    assert factory.call_args.kwargs["target"] == reporter._run
    factory.return_value.start.assert_called_once()


def test_heartbeat_owns_status_stop_event(monkeypatch):
    reporter = MagicMock()
    thread = MagicMock()
    monkeypatch.setattr(qs.threading, "Thread", MagicMock(return_value=thread))
    heartbeat = qs._Heartbeat(
        "https://hb.test", "key", 3600, lambda: 0, status=reporter
    )
    heartbeat.start()
    reporter.start.assert_called_once_with(heartbeat._stop)
    heartbeat.stop()
    assert heartbeat._stop.is_set()


def test_fatal_heartbeat_stops_status(monkeypatch):
    heartbeat = qs._Heartbeat("https://hb.test", "key", 10, lambda: 0)
    heartbeat._stop = MagicMock()
    heartbeat._stop.wait.return_value = False
    heartbeat.enabled.set()
    monkeypatch.setattr(heartbeat, "_beat_once", heartbeat.fatal.set)
    heartbeat._run()
    heartbeat._stop.set.assert_called_once()


@pytest.fixture
def cache(monkeypatch, tmp_path):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(gate, "is_repo_cached", lambda repo: False)
    monkeypatch.setattr(gate, "estimate_download_size_bytes", lambda repo: 15 * 1024**3)
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.delenv("RAPID_MLX_CHAT_SPAWN", raising=False)
    return tmp_path


@pytest.mark.parametrize("tty", [False, True])
def test_pool_gate_resolves_real_alias_and_reuses_estimate(
    cache, clock, monkeypatch, capsys, tty
):
    from rapid_mlx.model_aliases import resolve_profile

    repo = resolve_profile("qwen3.8-27b-4bit").hf_path
    cached = MagicMock(return_value=False)
    estimated = MagicMock(return_value=15 * 1024**3)
    confirm = MagicMock()
    monkeypatch.setattr(gate, "is_repo_cached", cached)
    monkeypatch.setattr(gate, "estimate_download_size_bytes", estimated)
    monkeypatch.setattr(gate, "confirm_or_abort", confirm)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: tty)
    reporter = cli._maybe_confirm_download("qwen3.8-27b-4bit", pool_mode=True)
    assert reporter.cache_dir == cache / f"models--{repo.replace('/', '--')}"
    cached.assert_called_once_with(repo)
    estimated.assert_called_once_with(repo)
    if tty:
        confirm.assert_called_once_with(repo, 15 * 1024**3)
    else:
        confirm.assert_not_called()
    assert "approximately 15.0 GiB" in capsys.readouterr().err


@pytest.mark.parametrize("variable", ["RAPID_MLX_AUTO_PULL", "RAPID_MLX_CHAT_SPAWN"])
def test_pool_gate_env_skips_confirmation_but_keeps_progress(
    cache, monkeypatch, variable
):
    monkeypatch.setenv(variable, "1")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    confirm = MagicMock()
    monkeypatch.setattr(gate, "confirm_or_abort", confirm)
    assert cli._maybe_confirm_download("org/repo", pool_mode=True) is not None
    confirm.assert_not_called()


def test_cached_pool_model_is_silent(cache, monkeypatch, capsys):
    monkeypatch.setattr(gate, "is_repo_cached", lambda repo: True)
    estimate = MagicMock()
    monkeypatch.setattr(gate, "estimate_download_size_bytes", estimate)
    assert cli._maybe_confirm_download("org/repo", pool_mode=True) is None
    estimate.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_pool_local_path_and_unknown_alias_are_silent(cache, capsys):
    assert cli._maybe_confirm_download(str(cache), pool_mode=True) is None
    assert cli._maybe_confirm_download("unknown-alias", pool_mode=True) is None
    assert capsys.readouterr() == ("", "")


def test_download_progress_fake_cache_and_clock(cache, clock, capsys):
    reporter = cli._DownloadProgress("org/repo", None)
    assert "estimated size unknown" in capsys.readouterr().err
    blobs = reporter.cache_dir / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "weights.incomplete").write_bytes(b"x" * 1840)
    snapshot = reporter.cache_dir / "snapshots" / "sha"
    snapshot.mkdir(parents=True)
    (blobs / "config").write_bytes(b"xx")
    (snapshot / "config.json").symlink_to(blobs / "config")
    assert reporter._cache_size() == 1842
    for now in (100, 101, 129.9):
        clock[0] = now
        reporter.poll()
    assert capsys.readouterr().err == ""
    clock[0] = 130
    reporter.poll()
    assert "30s elapsed, model cache size 0.0 GiB" in capsys.readouterr().err
    (blobs / "weights.incomplete").rename(blobs / "weights")
    (snapshot / "weights").symlink_to(blobs / "weights")
    assert reporter._cache_size() == 1842
    clock[0] = 160
    reporter.poll()
    assert "60s elapsed, model cache size 0.0 GiB" in capsys.readouterr().err


def test_download_progress_missing_cache_and_permission_failure(
    cache, clock, monkeypatch, capsys
):
    reporter = cli._DownloadProgress("org/repo", 1)
    capsys.readouterr()
    clock[0] = 130
    reporter.poll()
    assert "model cache size 0.0 GiB" in capsys.readouterr().err

    def fail(*a, **kw):
        raise PermissionError()

    monkeypatch.setattr(cli.os, "walk", fail)
    clock[0] = 160
    reporter.poll()
    assert "model cache size unavailable" in capsys.readouterr().err


def test_download_progress_relocated_blobs_and_safe_symlinks(cache, clock, capsys):
    reporter = cli._DownloadProgress("org/repo", None)
    capsys.readouterr()
    reporter.cache_dir.mkdir()
    blobs = cache / "relocated-blobs"
    blobs.mkdir()
    reporter.cache_dir.joinpath("blobs").symlink_to(blobs, target_is_directory=True)
    weights = blobs / "weights.incomplete"
    with weights.open("wb") as stream:
        stream.truncate(2 * 1024**3)
    snapshot = reporter.cache_dir / "snapshots" / "sha"
    snapshot.mkdir(parents=True)
    snapshot.joinpath("weights").symlink_to(weights)
    cli.os.link(weights, blobs / "hardlink")
    outside = cache / "unrelated"
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside")
    for base in (reporter.cache_dir, blobs):
        (base / "outside-dir").symlink_to(outside, target_is_directory=True)
        (base / "outside-file").symlink_to(outside / "secret")
    assert reporter._cache_size() == 2 * 1024**3
    clock[0] += 30
    reporter.poll()
    assert "model cache size 2.0 GiB" in capsys.readouterr().err
    weights.rename(blobs / "weights")
    # The snapshot is now dangling; the renamed blob and hardlink still count once.
    assert reporter._cache_size() == 2 * 1024**3


def test_download_progress_is_polled_only_while_waiting_for_health(monkeypatch):
    proc = MagicMock()
    proc.poll.return_value = None
    response = MagicMock()
    response.__enter__.return_value.status = 200
    urlopen = MagicMock(side_effect=[TimeoutError(), response])
    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    progress = MagicMock()
    assert cli._wait_for_healthz(8765, proc, progress=progress)
    assert progress.poll.call_count == 2
    proc.poll.return_value = 1
    progress.reset_mock()
    assert not cli._wait_for_healthz(8765, proc, progress=progress)
    progress.poll.assert_not_called()
