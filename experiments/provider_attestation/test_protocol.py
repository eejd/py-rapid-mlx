"""Adversarial protocol tests plus real Swift/CryptoKit cross-process checks."""

import base64
import copy
import hashlib
import pathlib
import plistlib
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_probe import build
from native import NativeProvider
from protocol import (
    AuthorizationRefusedError,
    Coordinator,
    FixtureAuthority,
    b64,
    open_response,
)


@pytest.fixture
def setup():
    authority = FixtureAuthority()
    now = [100.0]
    coordinator = Coordinator(
        mode="fixture-only",
        fixture_key=authority.public_key,
        approved_builds=["a" * 64],
        clock=lambda: now[0],
    )
    key = x25519.X25519PrivateKey.generate()
    endpoint = coordinator.connect(
        account="account-a",
        model="model-a",
        public_key=b64(key.public_key().public_bytes_raw()),
        build="a" * 64,
    )
    return coordinator, authority, endpoint, now


def grant(setup):
    coordinator, authority, endpoint, _ = setup
    challenge = coordinator.challenge(endpoint)
    token = coordinator.authorize(
        endpoint=endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
    )
    return token, challenge


def test_apple_mode_never_accepts_fixture_or_unsupported_fallback(setup):
    fixture, authority, endpoint, _ = setup
    apple = Coordinator(approved_builds=[endpoint.build])
    endpoint = apple.connect(
        **{k: v for k, v in endpoint.fields().items() if k != "connection"}
    )
    challenge = apple.challenge(endpoint)
    with pytest.raises(AuthorizationRefusedError, match="apple_verifier_not_qualified"):
        apple.authorize(
            endpoint=endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
        )
    assert not apple.leases
    with pytest.raises(AuthorizationRefusedError, match="fixture_key_in_apple_mode"):
        Coordinator(fixture_key=authority.public_key)


def test_replayed_challenge_cannot_create_second_lease(setup):
    coordinator, authority, endpoint, _ = setup
    _, challenge = grant(setup)
    with pytest.raises(AuthorizationRefusedError, match="replayed"):
        coordinator.authorize(
            endpoint=endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
        )
    assert len(coordinator.leases) == 1


def test_forged_proof_consumes_nonce(setup):
    coordinator, authority, endpoint, _ = setup
    challenge = coordinator.challenge(endpoint)
    with pytest.raises(AuthorizationRefusedError, match="invalid_proof"):
        coordinator.authorize(
            endpoint=endpoint,
            nonce=challenge.nonce,
            proof=FixtureAuthority().sign(challenge),
        )
    with pytest.raises(AuthorizationRefusedError, match="replayed"):
        coordinator.authorize(
            endpoint=endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("account", "account-b"),
        ("model", "model-b"),
        ("connection", "replacement"),
        ("public_key", b64(bytes(range(32)))),
        ("build", "b" * 64),
    ],
)
def test_identity_or_key_substitution_never_authorizes(setup, field, value):
    coordinator, authority, endpoint, _ = setup
    challenge = coordinator.challenge(endpoint)
    substituted = replace(endpoint, **{field: value})
    with pytest.raises(AuthorizationRefusedError):
        coordinator.authorize(
            endpoint=substituted, nonce=challenge.nonce, proof=authority.sign(challenge)
        )
    assert not coordinator.leases


def test_challenge_expiry_is_not_extended_by_retry(setup):
    coordinator, authority, endpoint, now = setup
    challenge = coordinator.challenge(endpoint)
    now[0] += 15
    with pytest.raises(AuthorizationRefusedError, match="challenge_expired"):
        coordinator.authorize(
            endpoint=endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
        )


def test_expired_lease_cannot_send_and_build_revocation_is_immediate(setup):
    coordinator, _, endpoint, now = setup
    token, _ = grant(setup)
    now[0] += 30
    with pytest.raises(AuthorizationRefusedError, match="lease_expired"):
        coordinator.dispatch(token, endpoint, b"hello")
    token, _ = grant(setup)
    coordinator.approved_builds.clear()
    with pytest.raises(AuthorizationRefusedError, match="build_not_approved"):
        coordinator.dispatch(token, endpoint, b"hello")


def test_reconnect_requires_new_proof_not_old_permission(setup):
    coordinator, _, endpoint, _ = setup
    token, _ = grant(setup)
    coordinator.disconnect(endpoint.connection)
    replacement = coordinator.connect(
        **{k: v for k, v in endpoint.fields().items() if k != "connection"}
    )
    with pytest.raises(AuthorizationRefusedError, match="no_serving_permission"):
        coordinator.dispatch(token, replacement, b"hello")


@pytest.mark.parametrize(
    "field,value", [("account", "b"), ("model", "b"), ("public_key", b64(bytes(32)))]
)
def test_lease_cannot_be_used_for_another_scope(setup, field, value):
    coordinator, _, endpoint, _ = setup
    token, _ = grant(setup)
    with pytest.raises(AuthorizationRefusedError, match="endpoint_mismatch"):
        coordinator.dispatch(token, replace(endpoint, **{field: value}), b"hello")


def test_unapproved_build_and_missing_permission_cannot_send(setup):
    coordinator, _, endpoint, _ = setup
    unapproved = coordinator.connect(
        account=endpoint.account,
        model=endpoint.model,
        public_key=endpoint.public_key,
        build="b" * 64,
    )
    with pytest.raises(AuthorizationRefusedError, match="build_not_approved"):
        coordinator.challenge(unapproved)
    with pytest.raises(AuthorizationRefusedError, match="no_serving_permission"):
        coordinator.dispatch("unknown", endpoint, b"hello")


def test_null_x25519_key_and_oversized_payload_fail_closed(setup):
    coordinator, authority, endpoint, _ = setup
    null_endpoint = coordinator.connect(
        account=endpoint.account,
        model=endpoint.model,
        public_key=b64(bytes(32)),
        build=endpoint.build,
    )
    challenge = coordinator.challenge(null_endpoint)
    token = coordinator.authorize(
        endpoint=null_endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
    )
    with pytest.raises(AuthorizationRefusedError, match="invalid_endpoint_key"):
        coordinator.dispatch(token, null_endpoint, b"hello")
    token, _ = grant(setup)
    with pytest.raises(AuthorizationRefusedError, match="payload_too_large"):
        coordinator.dispatch(token, endpoint, b"a" * 16385)


@pytest.fixture(scope="session")
def executable(tmp_path_factory):
    if sys.platform != "darwin":
        pytest.skip("native Apple probe requires macOS")
    return build(tmp_path_factory.mktemp("native-provider"))


@pytest.fixture
def native_setup(executable):
    with NativeProvider(executable) as native:
        authority = FixtureAuthority()
        coordinator = Coordinator(
            mode="fixture-only",
            fixture_key=authority.public_key,
            approved_builds=[native.hello["build"]],
        )
        endpoint = coordinator.connect(
            account="fixture",
            model="no-model",
            public_key=native.hello["public_key"],
            build=native.hello["build"],
        )
        challenge = coordinator.challenge(endpoint)
        token = coordinator.authorize(
            endpoint=endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
        )
        yield native, coordinator, endpoint, token


def test_native_roundtrip_uses_real_cryptokit_and_no_plaintext_reply(native_setup):
    native, coordinator, endpoint, token = native_setup
    payload = b"synthetic request bytes"
    envelope, key = coordinator.dispatch(token, endpoint, payload)
    reply = native.call({"operation": "open", "envelope": envelope})
    result = open_response(envelope, reply, key)
    assert result == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "inference": False,
    }
    assert native.hello["apple_verified"] is False
    assert native.hello["code_identity"]["official_identity"] is False
    assert native.hello["code_identity"]["hardened_runtime"] is True
    assert native.hello["core_dumps_disabled"] is True
    assert "sha256" not in reply and "bytes" not in reply


def test_native_rejects_replayed_request(native_setup):
    native, coordinator, endpoint, token = native_setup
    envelope, key = coordinator.dispatch(token, endpoint, b"synthetic")
    open_response(
        envelope, native.call({"operation": "open", "envelope": envelope}), key
    )
    assert (
        native.call({"operation": "open", "envelope": envelope})["error"]
        == "request_replayed"
    )


@pytest.mark.parametrize(
    "tamper", ["ciphertext", "account", "public_key", "build", "sender_key"]
)
def test_native_rejects_modified_ciphertext_or_binding(native_setup, tamper):
    native, coordinator, endpoint, token = native_setup
    envelope, _ = coordinator.dispatch(token, endpoint, b"synthetic")
    changed = copy.deepcopy(envelope)
    if tamper in {"ciphertext", "sender_key"}:
        raw = bytearray(base64.b64decode(changed[tamper]))
        raw[0] ^= 1
        changed[tamper] = b64(raw)
    else:
        changed["binding"][tamper] = "tampered"
    assert "error" in native.call({"operation": "open", "envelope": changed})


def test_response_replay_to_different_request_is_rejected(native_setup):
    native, coordinator, endpoint, token = native_setup
    first, key = coordinator.dispatch(token, endpoint, b"synthetic")
    reply = native.call({"operation": "open", "envelope": first})
    second, _ = coordinator.dispatch(token, endpoint, b"synthetic")
    with pytest.raises(AuthorizationRefusedError, match="invalid_response"):
        open_response(second, reply, key)


def test_native_unsupported_apple_path_is_denied(native_setup):
    native, _, _, _ = native_setup
    if native.hello["app_attest_supported"]:
        pytest.skip(
            "Apple is supported here; needs an entitled official qualification run"
        )
    assert native.call({"operation": "attest"})["error"] == "app_attest_unsupported"


def test_actual_signed_bundle_tampering_is_detected(executable, tmp_path):
    original = executable.parents[2]
    copied = tmp_path / original.name
    shutil.copytree(original, copied)
    plist = copied / "Contents" / "Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info["CFBundleVersion"] = "tampered"
    plist.write_bytes(plistlib.dumps(info))
    result = subprocess.run(
        ["codesign", "--verify", "--strict", str(copied)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0


def test_second_native_process_cannot_decrypt_first_process_request(
    native_setup, executable
):
    native, coordinator, endpoint, token = native_setup
    envelope, _ = coordinator.dispatch(token, endpoint, b"synthetic")
    with NativeProvider(executable) as other:
        assert other.hello["public_key"] != native.hello["public_key"]
        assert (
            other.call({"operation": "open", "envelope": envelope})["error"]
            == "endpoint_mismatch"
        )


def test_cross_language_binding_with_model_path(native_setup):
    native, coordinator, endpoint, _ = native_setup
    authority = FixtureAuthority()
    coordinator.fixture_key = authority.public_key
    endpoint = coordinator.connect(
        account=endpoint.account,
        model="synthetic/model",
        public_key=endpoint.public_key,
        build=endpoint.build,
    )
    challenge = coordinator.challenge(endpoint)
    token = coordinator.authorize(
        endpoint=endpoint, nonce=challenge.nonce, proof=authority.sign(challenge)
    )
    envelope, key = coordinator.dispatch(token, endpoint, b"synthetic")
    response = native.call({"operation": "open", "envelope": envelope})
    assert open_response(envelope, response, key)["bytes"] == 9
