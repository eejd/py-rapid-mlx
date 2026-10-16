"""Real native encryption plus explicitly synthetic authorization. No inference."""

import argparse
import hashlib
import json
import pathlib

from native import NativeProvider
from protocol import (
    DOMAIN,
    AuthorizationRefusedError,
    Coordinator,
    FixtureAuthority,
    open_response,
)


def run(executable: pathlib.Path, fixture: bool) -> dict:
    with NativeProvider(executable) as provider:
        hello = provider.hello
        receipt = {
            "native_probe": hello,
            "apple_verified": False,
            "inference_tested": False,
        }
        # Exercise the actual native availability guard rather than inferring
        # availability solely from the OS version or enclave presence.
        if not hello["app_attest_supported"]:
            receipt["native_attestation_attempt"] = provider.call(
                {"operation": "attest"}
            )
        if not fixture:
            receipt["result"] = "not_authorized"
            receipt["reason"] = "apple_verifier_not_qualified"
            return receipt
        authority = FixtureAuthority()
        # This harness approves its own freshly built binary ONLY for testing;
        # a real coordinator needs an independently trusted release manifest.
        coordinator = Coordinator(
            mode="fixture-only",
            fixture_key=authority.public_key,
            approved_builds=[hello["build"]],
        )
        endpoint = coordinator.connect(
            account="synthetic-account",
            model="synthetic-transform",
            public_key=hello["public_key"],
            build=hello["build"],
        )
        challenge = coordinator.challenge(endpoint)
        proof = authority.sign(challenge)
        token = coordinator.authorize(
            endpoint=endpoint, nonce=challenge.nonce, proof=proof
        )
        payload = b'{"messages":[{"role":"user","content":"synthetic privacy canary"}]}'
        envelope, key = coordinator.dispatch(token, endpoint, payload)
        encrypted_reply = provider.call({"operation": "open", "envelope": envelope})
        result = open_response(envelope, encrypted_reply, key)
        assert result["sha256"] == hashlib.sha256(payload).hexdigest()
        assert result["inference"] is False
        replay = provider.call({"operation": "open", "envelope": envelope})
        assert replay.get("error") == "request_replayed"
        coordinator.disconnect(endpoint.connection)
        try:
            coordinator.dispatch(token, endpoint, payload)
        except AuthorizationRefusedError as error:
            assert str(error) == "no_serving_permission"
        else:
            raise AssertionError("disconnected endpoint was allowed")
        # Never archive lease tokens, private keys or plaintext transport frames.
        receipt.update(
            mode="fixture-only",
            result="protocol_roundtrip_passed",
            transcript_domain=DOMAIN,
            native_encrypted_roundtrip=True,
            native_replay_rejected=True,
            disconnected_handoff_rejected=True,
        )
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=pathlib.Path, required=True)
    parser.add_argument(
        "--fixture-demo",
        action="store_true",
        help="EXPLICITLY enable synthetic proof; never establishes Apple trust",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.provider.resolve(), args.fixture_demo), indent=2, sort_keys=True
        )
    )
