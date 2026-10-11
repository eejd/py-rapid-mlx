# SPDX-License-Identifier: Apache-2.0
"""Cache diagnostics follow the serving backend without optional global APIs."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.requires_mlx


@pytest.mark.parametrize("stats", [None, {"entry_count": 2, "hits": 3}])
def test_vision_cache_stats_use_active_engine(monkeypatch, stats):
    from rapid_mlx.routes import health

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", SimpleNamespace(utils=SimpleNamespace())
    )
    engine = SimpleNamespace(is_mllm=True, get_cache_stats=Mock(return_value=stats))
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    assert asyncio.run(health.cache_stats()) == {
        "model_type": "mllm",
        "multimodal_kv_cache": stats,
        "pixel_values_cache": None,
        "pil_image_cache": None,
    }
    engine.get_cache_stats.assert_called_once_with()


@pytest.mark.parametrize(
    "engine,model_type", [(None, None), (SimpleNamespace(is_mllm=False), "llm")]
)
def test_cache_stats_do_not_infer_model_type_from_installed_packages(
    monkeypatch, engine, model_type
):
    from rapid_mlx.routes import health

    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    result = asyncio.run(health.cache_stats())
    assert result["model_type"] == model_type
    assert result["message"]


@pytest.mark.parametrize(
    "pixel_available,pil_available", [(True, True), (True, False), (False, True)]
)
def test_vision_cache_stats_preserve_available_optional_image_stats(
    monkeypatch, pixel_available, pil_available
):
    from rapid_mlx.routes import health

    kv_stats = {"entry_count": 2}
    pixel_stats = {"hits": 7}
    pil_stats = {"hits": 11}
    utils = SimpleNamespace()
    if pixel_available:
        utils.get_pixel_values_cache_stats = Mock(return_value=pixel_stats)
    if pil_available:
        utils.get_pil_cache_stats = Mock(return_value=pil_stats)
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(utils=utils))
    engine = SimpleNamespace(is_mllm=True, get_cache_stats=lambda: kv_stats)
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))

    assert asyncio.run(health.cache_stats()) == {
        "model_type": "mllm",
        "multimodal_kv_cache": kv_stats,
        "pixel_values_cache": pixel_stats if pixel_available else None,
        "pil_image_cache": pil_stats if pil_available else None,
    }


def test_vision_cache_stats_work_without_optional_image_apis(monkeypatch):
    from rapid_mlx.routes import health

    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    engine = SimpleNamespace(is_mllm=True, get_cache_stats=lambda: {"hits": 3})
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    result = asyncio.run(health.cache_stats())
    assert result["model_type"] == "mllm"
    assert result["multimodal_kv_cache"] == {"hits": 3}
    assert result["pixel_values_cache"] is None
    assert result["pil_image_cache"] is None
