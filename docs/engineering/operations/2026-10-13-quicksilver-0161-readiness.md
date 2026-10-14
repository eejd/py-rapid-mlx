# QuickSilver Share Compute readiness for Rapid-MLX 0.16.1

Assessment date: 2026-10-13. Owner: Atlas; validation host: Studio.
Scope: Desktop Share Compute / Compute & Earn, CLI provider mode, pool relay,
gateway routing and metering, contributor accounting, and website onboarding.
No production configuration, deployments, keys, registrations, or balances were
changed by this assessment.

## Decision

The 0.16.1 provider implementation is suitable for a narrowly scoped,
explicitly enabled Qwen3.8-27B beta after the release actions below. Do not
promote the current experience to a default-on, general-availability feature.
The code is already included in the released 0.16.1 Desktop and engine tags;
the Desktop preference remains default-off. Shipping the feature code and
opening contributor participation are separate decisions.

The immediate work is integration and operational acceptance, rather than a
new inference architecture. The absence of a Mac GPU TEE does not by itself
block this product scope: it does constrain the privacy promises and the
traffic eligible for community execution.

## Evidence and version binding

- Rapid release tags `v0.16.1` and `rapid-mac-v0.16.1` both resolve to
  `df1f4db1fc0548ec97c6376739a76f6330e9b000`.
- Initial discovery used main `b638c2ab6`; there are no differences from the
  release in `rapid_mlx/share`, Desktop `ShareCompute`, or the focused share
  tests. Python validation was nevertheless repeated on the exact release.
- Backend source examined: `74d8ef8b0a3bfb12c2757ce865d04cd8808f24f6`.
  The verified production manifest records that stripe commit and gateway
  `c222d021ba649821d361d8dd75b10dcd189208ea`; the pool gateway sources have no
  intervening differences.
- Frontend source and recorded Pages deployment:
  `48a2b3bfe011855bde30e9b54df10b46b5f337dc`.
- The live relay health response identifies
  `b6b090a24e0f6d540538bf2169c8e3548e217b8c`; its `relay/` tree has no
  differences from the backend checkout tested here.
- A public pool snapshot at 23:52 UTC reported two ready Qwen nodes and four
  available slots. This is a point-in-time capacity observation, not an SLA.
- Release candidate qualification/admission checks for the Rapid release
  commit and the latest backend/frontend verification runs reported success.

Existing live evidence includes the 2026-09-30 customer UX diagnostic and the
2026-10-10 real-model privacy validation. The former traced independent
consumer traffic to three Macs and cloud fallback. The latter validated
streaming/nonstreaming requests through an isolated relay fixture, restart
with cached registration, prompt-artifact cleanup and verified TLS behavior.
Neither substitutes for a fresh signed-0.16.1 end-to-end acceptance run.

## Confirmed integration finding: model availability contract

Severity: P1 for broad Desktop onboarding. Owners: backend owner and Pixel;
Atlas owns the Qwen-only scope decision.

Backend `stripe/src/config.py` declares only `qwen3.8-27b` in
`POOL_OPEN_MODELS`. Registration in `pool_provider.register_node` rejects the
other three supported model IDs. However, `pool_summary._model_entry` always
returns `enabled: true` for every entry in the broader `POOL_MODELS` catalog.
The public production summary reproduces this mismatch for Nemotron,
Qwen3.6-35B, and GLM5.3-Flash.

Desktop `ShareComputePoolAvailability.acceptsConnections` interprets enabled
as permission to connect. Furthermore, `ShareComputeView.shareableModels`
filters only local availability and hardware fit, and
`ShareComputeManager.join` checks the static supported catalog. An already
downloaded closed model can therefore reach registration from Share even if
the public summary flag is corrected. The user can pause their local model
only to receive a preventable unsupported-model error.

Required action: make the public connection eligibility agree with the
registration allowlist, and apply that eligibility to both Live Pool and the
Share picker/action. Test all four models, including locally cached closed
models and unknown/stale availability. Do not infer real customer demand
from a model merely being present in the supported catalog.

## Required operational disposition: old clients

Owners: Harbor and QuickSilver backend/relay owner.

0.16.1 includes the prompt-cache privacy fixes and legacy cache cleanup.
The backend exposes `POOL_MIN_RAPID_VERSION`, but its source default is empty
and its check runs only at registration. Existing cached share credentials
reconnect directly to the relay. Setting the registration floor alone does
not retire previously registered vulnerable clients. Version strings are
self-reported and are upgrade hygiene, not attestation against a malicious
node owner.

Before expanding participation, verify the actual production floor, inventory
existing node versions, and document retirement/re-registration for affected
nodes. Prefer 0.16.1 as the beta baseline validated here. Confirm that an old
cached client cannot remain an unintended eligible production node. The
actual production environment variable and old-node inventory were not
accessible in this assessment; their state is unknown, not proved disabled.

## Final acceptance still required

Owner: Harbor, with Pixel for Desktop behavior and backend owner for ledger
traceability. Use the signed 0.16.1 artifact and a distinct consumer account.

1. First-run key entry, permitted-model download/selection, registration,
   warmup, readiness and a successful streamed inference.
2. Client cancellation, node disconnect/reconnect, capacity exhaustion and
   cloud fallback before output. A partial stream must fail explicitly rather
   than silently restart on another model/node.
3. Stop sharing and quit: verify child teardown, released capacity, and the
   prior local model becoming ready again.
4. Trace request attribution to exactly one accounting window and verify the
   contributor credit appears after settlement; verify no double credit and
   owner-account exclusion.
5. Inspect prompt-cache artifacts after stop; exercise a known legacy snapshot
   and confirm the fixed client removes it before accepting pool traffic.

The standalone login LaunchAgent is outside the recommended Desktop beta.
The existing privacy-validation report explicitly leaves service inference
unvalidated on that host despite successful install/reinstall/stop mechanics.
Do not advertise unattended login-service readiness based on those mechanics.

## Existing strengths and boundaries

- Relay readiness, bounded per-request deadlines, first-output gating,
  retry-before-output, reconnect fencing, admission limits and cancellation
  have implementation and behavioral tests.
- Gateway recounts billable usage and creates its own attribution ID rather
  than trusting node-reported token counts or request IDs.
- Settlement includes owner-account exclusion, unique windows, locking,
  per-node/pool caps, pause controls and alerting. These limit several abuse
  paths; they do not prove that a contributor ran the advertised model.
- Desktop has explicit opt-in, key transport over stdin, a separate Keychain
  read key, exclusive model residency, and process-group teardown.
- Pool mode disables optional disk caches, retires recognized legacy prompt
  snapshots and fails closed on cleanup/TLS failures.
- Website terms, also verified on the live privacy page, disclose community execution and that host compliance cannot
  be technically guaranteed. Neither stateless requests nor disabled disk
  caches protect prompt plaintext from a malicious execution host.
- A cloud-only customer routing preference was not found in the inspected
  pool policy. Website policy directs such customers to contact support;
  do not promise a self-service privacy toggle that is absent.

## Verification

On the exact Rapid release, Python 3.11.16:

```sh
python3.11 -m pytest -q tests/test_share_cli.py tests/test_share_quicksilver.py \
  tests/test_share_privacy.py tests/test_share_progress.py \
  tests/test_share_progress_adversarial.py tests/test_share_pool_accounting.py
```

Result: **492 passed**. The accounting test uses the real batched/MTP engine
with model doubles; this run did not download or load real model weights.

Backend checks: all nine focused gateway/stripe scripts passed after installing
missing local test dependencies in an isolated scratch environment. Relay
TypeScript checking passed, and **107 tests passed** under the Workers test
runtime. Disconnect/timeout/revoke cases emit runtime exception diagnostics;
the runner still reports success. This is not evidence of clean production
error logs or a reason to erase those diagnostics from the record.

Frontend: `pnpm test:share-compute` and TypeScript checking passed, including
four-locale onboarding, earnings estimates/caps, node removal, authentication,
polling and stale-response handling.

Desktop: `swift test --filter ShareCompute` on the exact release passed
**125 tests in seven suites**, using a fresh scratch build and four build jobs.
This validates source behavior, not the signed DMG's real production session.
An initial main checkout disappeared while Swift was compiling; its
missing-file failure was resolved by the fresh exact-release checkout.

## Rollout and rollback handoff

### Follow-up: signed artifact on a community-class Mac

At the owner's request, Machine Control assigned an available M4 Pro Mac mini
with 48 GiB RAM. The released DMG's SHA-256 matched the release asset:
`91374263df64e4a7e1c62b0f905dfa3214ac55812eb26f6f85222d7f318cacee`.
Deep/strict code-signature verification passed. Both the app version and its
bundled engine reported 0.16.1. The installed app and resident provider were
left untouched; the signed app was staged in task scratch space.

Weights were copied from the internal hub, SHA-256 checked by the hub client,
into the worker's canonical HF cache, with no external model download:
`rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` at
`1784d09391ee8affa203f58f5103e2a5f139135c`.

The bundled engine, using the ordinary `qwen3.8-27b-4bit` alias, offline mode,
two sequence slots and disabled disk caches, passed seven local checks:
readiness/catalog identity; missing-key rejection (401); nonstreaming answer
and positive usage; SSE content/usage/DONE; two concurrent requests; a new
request after closing a stream; and a forced `get_weather` tool call.
Readiness took approximately 16 seconds in this small diagnostic. This is not
a throughput benchmark or proof of GPU memory behavior under production load.

The existing resident QuickSilver provider reported version 0.15.7. Its
processes were preserved; no credentials were inspected or reused and no
upgrade was performed. This confirms that old-client disposition concerns an
actual resident node, not only a hypothetical migration case. Its cache
retention behavior was not tested, so the version observation alone does not
establish that it persisted any customer prompts.

The task server stopped successfully, only the task-created model copy was
removed, remote task credentials and staged app/DMG were cleaned, and the
machine was returned. Real production registration, routed consumer requests,
and earnings verification remain pending owner-provided test credentials.
Local checks do not close that end-to-end requirement or Desktop UI restoration.

Atlas recommends Qwen-only, default-off beta once the items above have explicit
dispositions. Harbor verifies production upgrade enforcement, signed-artifact
acceptance, monitored cloud fallback, settlement alerts and rollback ownership.
The documented rollback is to disable the Qwen pool route (or the pool gateway
secret for all routes) and release the gateway so traffic uses cloud providers;
node removal/revocation and payout pause are separate controls. Desktop's local
experimental preference is not a server-side fleet kill switch.

No production release or rollback was executed. Remaining state: **review**,
with the model-eligibility fix, old-node disposition and exact-artifact acceptance
as the next concrete actions.
