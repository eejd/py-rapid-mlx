"""Isolated authorization experiment. Fixture proofs NEVER establish Apple trust.

No production imports, endpoints, model inference, persistence or deployments.
The coordinator object models an authenticated gateway, not a public API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

DOMAIN = "rapid-provider-attestation-experiment-v1"
MAX_PAYLOAD = 16 * 1024


class AuthorizationRefusedError(ValueError):
    """Closed diagnostic category, never request content."""


def canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def unb64(value: str, length: int) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        raise AuthorizationRefusedError("invalid_encoding") from None
    if len(raw) != length:
        raise AuthorizationRefusedError("invalid_length")
    return raw


@dataclass(frozen=True)
class Endpoint:
    # Account comes from authenticated coordinator context, not proof claims.
    account: str
    model: str
    connection: str
    public_key: str
    build: str

    def fields(self) -> dict:
        return dict(
            account=self.account,
            model=self.model,
            connection=self.connection,
            public_key=self.public_key,
            build=self.build,
        )


@dataclass(frozen=True)
class Challenge:
    endpoint: Endpoint
    nonce: str
    expires: float

    def transcript(self) -> dict:
        return {
            "domain": DOMAIN,
            "proof_kind": "fixture-only",
            "nonce": self.nonce,
            **self.endpoint.fields(),
        }


@dataclass(frozen=True)
class Lease:
    endpoint: Endpoint
    lease_id: str
    expires: float

    def binding(self, request_id: str) -> dict:
        return {
            "domain": DOMAIN,
            "proof_kind": "fixture-only",
            "lease_id": self.lease_id,
            "request_id": request_id,
            **self.endpoint.fields(),
        }


class FixtureAuthority:
    """Synthetic signer controlled by the harness, NOT by Apple or Secure Enclave."""

    def __init__(self) -> None:
        self._key = ed25519.Ed25519PrivateKey.generate()

    @property
    def public_key(self) -> ed25519.Ed25519PublicKey:
        return self._key.public_key()

    def sign(self, challenge: Challenge) -> dict:
        return {
            "kind": "fixture-only",
            "signature": b64(self._key.sign(canonical(challenge.transcript()))),
        }


class Coordinator:
    def __init__(
        self,
        *,
        mode: str = "apple",
        fixture_key=None,
        approved_builds=(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in {"apple", "fixture-only"}:
            raise AuthorizationRefusedError("unknown_verifier")
        if mode == "fixture-only" and fixture_key is None:
            raise AuthorizationRefusedError("missing_fixture_authority")
        if mode == "apple" and fixture_key is not None:
            raise AuthorizationRefusedError("fixture_key_in_apple_mode")
        self.mode = mode
        self.fixture_key = fixture_key
        self.approved_builds = set(approved_builds)
        self.clock = clock
        self.connections: dict[str, Endpoint] = {}
        self.challenges: dict[str, Challenge] = {}
        self.leases: dict[str, Lease] = {}

    def connect(
        self, *, account: str, model: str, public_key: str, build: str
    ) -> Endpoint:
        unb64(public_key, 32)
        if not account or not model or len(build) != 64:
            raise AuthorizationRefusedError("invalid_endpoint")
        endpoint = Endpoint(account, model, secrets.token_hex(16), public_key, build)
        self.connections[endpoint.connection] = endpoint
        return endpoint

    def disconnect(self, connection: str) -> None:
        self.connections.pop(connection, None)
        self.challenges = {
            n: c
            for n, c in self.challenges.items()
            if c.endpoint.connection != connection
        }
        self.leases = {
            t: lease
            for t, lease in self.leases.items()
            if lease.endpoint.connection != connection
        }

    def _current(self, endpoint: Endpoint) -> None:
        if self.connections.get(endpoint.connection) != endpoint:
            raise AuthorizationRefusedError("connection_changed")
        if endpoint.build not in self.approved_builds:
            raise AuthorizationRefusedError("build_not_approved")

    def challenge(self, endpoint: Endpoint) -> Challenge:
        self._current(endpoint)
        self._prune()
        challenge = Challenge(endpoint, secrets.token_hex(32), self.clock() + 15)
        self.challenges[challenge.nonce] = challenge
        return challenge

    def authorize(self, *, endpoint: Endpoint, nonce: str, proof: dict) -> str:
        # Consume on the first attempt, including a failed proof. No retries of
        # one challenge with changed identities, keys, or signatures.
        challenge = self.challenges.pop(nonce, None)
        if challenge is None:
            raise AuthorizationRefusedError("unknown_or_replayed_challenge")
        self._current(endpoint)
        if self.clock() >= challenge.expires:
            raise AuthorizationRefusedError("challenge_expired")
        if endpoint != challenge.endpoint:
            raise AuthorizationRefusedError("endpoint_mismatch")
        if self.mode != "fixture-only":
            # Deliberately closed until a complete, independently qualified
            # Apple verifier and signed artifact exist. Never accept a fixture
            # because the local API is unsupported or Apple is unavailable.
            raise AuthorizationRefusedError("apple_verifier_not_qualified")
        if proof.get("kind") != "fixture-only":
            raise AuthorizationRefusedError("wrong_proof_kind")
        try:
            self.fixture_key.verify(
                unb64(proof.get("signature"), 64), canonical(challenge.transcript())
            )
        except InvalidSignature:
            raise AuthorizationRefusedError("invalid_proof") from None
        token = secrets.token_urlsafe(32)
        self.leases[token] = Lease(endpoint, secrets.token_hex(16), self.clock() + 30)
        return token

    def _lease(self, token: str, endpoint: Endpoint) -> Lease:
        lease = self.leases.get(token)
        if lease is None:
            raise AuthorizationRefusedError("no_serving_permission")
        if self.clock() >= lease.expires:
            self.leases.pop(token, None)
            raise AuthorizationRefusedError("lease_expired")
        if lease.endpoint != endpoint:
            raise AuthorizationRefusedError("endpoint_mismatch")
        self._current(endpoint)  # Check revocation at handoff, not only enrollment.
        return lease

    def dispatch(
        self, token: str, endpoint: Endpoint, payload: bytes
    ) -> tuple[dict, bytes]:
        lease = self._lease(token, endpoint)
        if not isinstance(payload, bytes) or len(payload) > MAX_PAYLOAD:
            raise AuthorizationRefusedError("payload_too_large")
        binding = lease.binding(secrets.token_hex(16))
        sender = x25519.X25519PrivateKey.generate()
        peer = x25519.X25519PublicKey.from_public_bytes(unb64(endpoint.public_key, 32))
        context = canonical(binding)
        try:
            shared = sender.exchange(peer)
        except ValueError:
            raise AuthorizationRefusedError("invalid_endpoint_key") from None
        key = derive(shared, context)
        nonce = secrets.token_bytes(12)
        encrypted = AESGCM(key).encrypt(nonce, payload, context + b"|request")
        self._lease(token, endpoint)  # Explicit final handoff guard.
        return {
            "binding": binding,
            "sender_key": b64(sender.public_key().public_bytes_raw()),
            "nonce": b64(nonce),
            "ciphertext": b64(encrypted),
        }, key

    def _prune(self) -> None:
        now = self.clock()
        self.challenges = {n: c for n, c in self.challenges.items() if c.expires > now}
        self.leases = {
            t: lease for t, lease in self.leases.items() if lease.expires > now
        }


def derive(shared: bytes, context: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=hashlib.sha256(context).digest(),
        info=DOMAIN.encode(),
    ).derive(shared)


def open_response(envelope: dict, response: dict, key: bytes) -> dict:
    from cryptography.exceptions import InvalidTag

    try:
        ciphertext = base64.b64decode(response["ciphertext"], validate=True)
        plaintext = AESGCM(key).decrypt(
            unb64(response["nonce"], 12),
            ciphertext,
            canonical(envelope["binding"]) + b"|response",
        )
        return json.loads(plaintext)
    except (InvalidTag, ValueError, KeyError, TypeError):
        raise AuthorizationRefusedError("invalid_response") from None
