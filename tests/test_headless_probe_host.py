# SPDX-License-Identifier: Apache-2.0
"""Headless probes must pass the real default HTTP Host boundary (#4512)."""

import http.client
import socket
import threading
import time
from contextlib import contextmanager

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from rapid_mlx.headless_service.install import _probe_host, _readyz_ready
from rapid_mlx.headless_service.status import _endpoint_health
from rapid_mlx.middleware.browser_security import (
    DEFAULT_TRUSTED_HOSTS,
    BrowserSecurityMiddleware,
)


@contextmanager
def secured_endpoint(bind_host):
    requests = []

    async def healthy(request):
        requests.append(dict(request.headers))
        return JSONResponse({"ready": True, "model_loaded": True})

    app = Starlette(routes=[Route("/livez", healthy), Route("/readyz", healthy)])
    app.add_middleware(
        BrowserSecurityMiddleware,
        hosts=list(DEFAULT_TRUSTED_HOSTS),
        deny_cross_origin=True,
    )
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((bind_host, 0))
    except OSError:
        sock.close()
        if family == socket.AF_INET6:
            pytest.skip("IPv6 loopback is unavailable")
        raise
    sock.listen()
    server = uvicorn.Server(
        uvicorn.Config(app, lifespan="off", log_level="critical", ws="none")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "probe test server did not start"
        yield sock.getsockname()[1], requests
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive(), "probe test server did not stop"


@pytest.mark.parametrize(
    "bind_host,configured_host",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1", "localhost"),
        ("0.0.0.0", "0.0.0.0"),
        ("::1", "::1"),
        ("::", "::"),
    ],
)
def test_service_probes_pass_default_host_boundary(bind_host, configured_host):
    with secured_endpoint(bind_host) as (port, requests):
        destination = _probe_host(configured_host)
        # Do not fix qualification by relaxing the browser security boundary.
        connection = http.client.HTTPConnection(destination, port, timeout=2)
        try:
            connection.request("GET", "/readyz", headers={"Host": "x"})
            assert connection.getresponse().status == 400
        finally:
            connection.close()

        assert _readyz_ready(configured_host, port, require_model_loaded=True)
        assert _endpoint_health(configured_host, port) == (True, True)
        assert len(requests) == 3
        authority = f"[{destination}]" if ":" in destination else destination
        assert all(request["host"] == f"{authority}:{port}" for request in requests)
        assert all("authorization" not in request for request in requests)
