# Provider authorization feasibility experiment

This is a development experiment, **not a confidential inference feature**.
It does not participate in Rapid's CLI, Desktop, QuickSilver routing, billing
or model serving. There is no public listener or production verifier.

It separates three questions:

1. Can the native host call the Apple APIs and use Secure Enclave signing?
2. Can a coordinator bind a one-use challenge, current connection, account,
   model, build and encryption endpoint to a short-lived serving permission?
3. Can the approved native endpoint decrypt authenticated request bytes and
   return an encrypted result without a plaintext Python inference sidecar?

The first question uses actual Apple APIs. The second uses an **explicit
fixture authority**, not Apple. The third uses real Swift CryptoKit and Python
cryptography across a subprocess boundary, with a synthetic hash transform,
**not MLX inference**. Passing any fixture test establishes neither Apple
attestation nor protection against the Mac owner.

## Run locally

Use macOS with Xcode command-line tools, Python 3.10+ and an isolated venv.
Build and scratch outputs belong outside the repository:

```sh
python3 -m venv /private/tmp/rapid-mlx-attestation-mvp/venv
/private/tmp/rapid-mlx-attestation-mvp/venv/bin/pip install \
  -r experiments/provider_attestation/requirements.txt

/private/tmp/rapid-mlx-attestation-mvp/venv/bin/python \
  experiments/provider_attestation/build_probe.py \
  --output /private/tmp/rapid-mlx-attestation-mvp/build
```

The executable is
`/private/tmp/rapid-mlx-attestation-mvp/build/RapidProviderProbe.app/Contents/MacOS/RapidProviderProbe`.
It is signed **ad hoc**, with Hardened Runtime and no entitlement exceptions.
It is not an official Developer ID artifact and has no App Attest provisioning.
It never installs services, downloads models or accesses account credentials.
The Secure Enclave signing probe uses an ephemeral handle, not a named
persistent Keychain entry. App Attest key creation is attempted only for an
explicit `attest` command when `isSupported` is true; Apple-created key state
may then outlive the process. Do not loop key creation during qualification.

First run the default, non-authorizing capability check:

```sh
/private/tmp/rapid-mlx-attestation-mvp/venv/bin/python \
  experiments/provider_attestation/demo.py \
  --provider /private/tmp/rapid-mlx-attestation-mvp/build/RapidProviderProbe.app/Contents/MacOS/RapidProviderProbe
```

It reports native availability but **never grants Apple serving permission**.
The native IPC `attest` action can produce evidence for a supplied transcript
only on a supported, appropriately provisioned build; evidence generation is
not server validation. The Python coordinator's default `apple` mode refuses
all authorization until a complete Apple verifier is separately qualified.

To exercise the synthetic authorization and real encryption path explicitly:

```sh
/private/tmp/rapid-mlx-attestation-mvp/venv/bin/python \
  experiments/provider_attestation/demo.py \
  --provider /private/tmp/rapid-mlx-attestation-mvp/build/RapidProviderProbe.app/Contents/MacOS/RapidProviderProbe \
  --fixture-demo

/private/tmp/rapid-mlx-attestation-mvp/venv/bin/python -m pytest -q \
  experiments/provider_attestation/test_protocol.py
```

Native tests compile and run the real Swift source on macOS. They skip on
other platforms; such a run does not qualify the native endpoint. No synthetic
result may be renamed `apple_verified=true` or treated as production approval.

## What the protocol verifies

- The harness uses a fresh Ed25519 **test** authority. A valid signature over
  a domain-separated transcript binds a 32-byte random challenge and endpoint
  identity. The coordinator, not the proof, supplies the authenticated account.
- Each challenge expires after 15 seconds and is consumed on its first attempt.
  A granted opaque permission expires after 30 seconds. These are local
  experiment constants, not proposed production freshness guarantees.
- Every dispatch checks the current connection, exact identity, live build
  allowlist and permission lifetime, again at final handoff. Disconnect removes
  that connection's challenges and permissions. Reconnect requires new proof.
- A fresh X25519 sender key and the native process's X25519 private key derive
  an AES-GCM key using HKDF-SHA256. Account, model, build, connection, request
  and permission IDs are authenticated context. Request and response directions
  use different authenticated-data labels and fresh nonces.
- The native private transport key never leaves the native process. Only an
  encrypted synthetic result returns; the coordinator can check its digest.
  In-process replay tracking fails closed after 256 requests rather than
  evicting old request IDs. The native harness is sequential.

The coordinator is an in-process, single-threaded model of an authenticated
gateway. Its `connect(account=...)` is **not** an authentication API. The harness
allowlists its own freshly built binary for fixture tests; this is **not**
independent measurement of an official release. Production requires trusted
release manifests, exact artifact qualification, durable replay/revocation
state, authenticated transport, and concurrency-safe handoff semantics.

The native stdin channel is a trusted test control channel, not a security
boundary against its owning user. It accepts properly encrypted fixture frames;
it does not independently validate coordinator serving permissions or the
coordinator's identity. It must not be exposed as a public provider endpoint.
The coordinator sees the synthetic plaintext. This experiment is not
consumer-to-provider encryption hiding content from the gateway.

## Native trust remains unqualified

An enclave signing test proves that this process can use a hardware key. An
App Attest availability flag, local signature check, self-reported build hash,
or ad-hoc hardened signature does not remotely prove a trusted runtime.

A production feasibility pass still needs all of:

- A supported OS/device and exact officially signed, entitled artifact.
- Real Apple evidence validated by a server using Apple's trust chain, nonce,
  application identity, macOS access policy and assertion replay checks.
- Binding that proof to the same process endpoint that handles plaintext,
  without a mutable Python/MLX sidecar breaking the trust chain.
- Adversarial evidence for modified code, imports, dynamic libraries, debugger
  attachment and security-state changes; app proof alone is not model-output
  correctness or guaranteed memory erasure.

See the [engineering findings](../../docs/engineering/decisions/2026-10-15-provider-attestation-feasibility.md)
for measured results, environment limits and the next checkpoint.
