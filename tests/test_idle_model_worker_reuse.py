"""Idle unload must not create a fresh MLX stream worker on every reload."""

import concurrent.futures
import gc
import threading
import weakref
from unittest.mock import AsyncMock, Mock

import pytest

from rapid_mlx.runtime.primary_lifecycle import PrimaryModelLifecycle


@pytest.mark.asyncio
async def test_idle_lifecycle_uses_idle_stop_when_available():
    class Engine:
        _loaded = True

        def __init__(self):
            self.calls = []

        async def idle_stop(self):
            self.calls.append("idle_stop")
            self._loaded = False

        async def stop(self):
            self.calls.append("stop")
            self._loaded = False

    engine = Engine()
    lifecycle = PrimaryModelLifecycle(engine, idle_unload_seconds=1, clock=lambda: 2)
    lifecycle._last_activity = 0

    assert await lifecycle.evict_if_idle() is True
    assert engine.calls == ["idle_stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize("dual_scheduler", [False, True])
async def test_mllm_idle_stop_releases_model_but_reuses_worker(
    monkeypatch, dual_scheduler
):
    from rapid_mlx.engine.batched import BatchedEngine

    class Model:
        pass

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    text_engine = Mock(stop=AsyncMock(), engine=Mock(close=Mock()))
    engine._engine = text_engine if dual_scheduler else None
    engine._mllm_scheduler = Mock(stop=AsyncMock())
    engine._guided_requests_lock = threading.Lock()
    monkeypatch.setattr(engine, "_abort_all_guided_requests", lambda: None)
    engine._model_load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    worker = engine._model_load_executor
    model = Model()
    model_ref = weakref.ref(model)
    engine._model = model
    engine._mllm_instance = model
    engine._tokenizer = None
    engine._processor = None
    engine._prompt_host_cache = Mock()
    engine._loaded = True

    try:
        del model
        await engine.idle_stop()
        gc.collect()

        assert model_ref() is None
        assert engine._model_load_executor is worker
        assert worker.submit(lambda: 1).result(timeout=2) == 1
        assert engine._mllm_scheduler is None
        if dual_scheduler:
            text_engine.stop.assert_awaited_once()
            text_engine.engine.close.assert_called_once()

        await engine.stop()
        assert engine._model_load_executor is None
        assert worker._shutdown is True
    finally:
        worker.shutdown(wait=True)
