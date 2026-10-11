# Pool share status: live Mac dogfood

PR #4516, `feat/pool-share-status-and-download-progress`. Owner: Vector;
Harbor owns the real-Mac execution evidence. Warm-up attribution is bounded
share-side: sample metrics before warm-up, then wait up to 10 seconds for its
request counter to advance before opening the relay. Poll about every 0.5
seconds; unavailable metrics skip the accounting wait and shutdown interrupts
it. Keep refreshing the baseline until the first relay dispatch.

Known limitation [NIT], accepted by the PR owner: if a timed-out warm-up is
accounted only after pool traffic starts, totals may include one warm-up request
and its few output tokens. The status line is informational; payouts are metered
by QuickSilver, not by this line. Automated tests pin both normal exclusion and
this residual behavior. No server/engine/scheduler/metrics accounting changes
are part of this fix. This checklist is verification work, not release
authorization.

Run the command sequence in the same launching shell; stop `tail -F` with
Ctrl-C before running the next commands. Use an eligible model that fits the
Mac. These commands use
`qwen3.8-27b-4bit`, catalog `qwen3.8-27b`. Run the candidate checkout's installed
CLI and Python environment; first-time node registration needs a provider key
through the existing supported setup. No keys should be pasted into this
checklist, shell history, or review evidence.

## 1. Cold cache, non-TTY, interrupted download, cached restart

```sh
ALIAS=qwen3.8-27b-4bit
CATALOG=qwen3.8-27b
DOGFOOD_DIR=$(mktemp -d "${TMPDIR%/}/rmlx-pr4516.XXXXXX")
RAPID_MLX_AUTO_PULL=1 HF_HOME="$DOGFOOD_DIR/hf" rapid-mlx share "$ALIAS" \
  --quicksilver --worker pr4516-dogfood --port 18765 \
  </dev/null >"$DOGFOOD_DIR/share.out" 2>"$DOGFOOD_DIR/share.err" &
DOGFOOD_PID=$!
tail -F "$DOGFOOD_DIR/share.err"
```

Expect an estimated GiB size or an explicit unknown estimate, immediately and
without a terminal prompt. At about 30 seconds and then each 30 seconds while
preparing, expect a newline with elapsed seconds and model cache GiB (or
unavailable if a scan fails/is still pending). There must be no carriage-return
animation. Blob growth should change the observed size; the value is cache
occupancy, not a percentage or transferred bytes. After `/healthz` becomes
healthy, preparation lines stop, including during warm-up and reconnects.

Stop the log tail and, in the launching shell before the download finishes:

```sh
kill -TERM "$DOGFOOD_PID"
wait "$DOGFOOD_PID"
lsof -nP -iTCP:18765 -sTCP:LISTEN
```

The listener must disappear. Repeat the startup command using the same
`DOGFOOD_DIR`: the partial cache must resume and still show progress, without
counting snapshot symlinks twice. Finish the download and stop/restart again.
The cached restart must print neither `First-run model download:` nor
`Preparing model:`. Preserve this isolated cache; no cache cleanup is part of
this test. A foreground cached run should also exit cleanly with Ctrl-C:

```sh
RAPID_MLX_AUTO_PULL=1 HF_HOME="$DOGFOOD_DIR/hf" rapid-mlx share "$ALIAS" \
  --quicksilver --worker pr4516-dogfood --port 18765
```

Check `HF_HUB_CACHE` precedence by repeating the cold-cache run with
`HF_HUB_CACHE="$DOGFOOD_DIR/override-hub"` added to its environment. The observed
cache must be that directory, not the `HF_HOME` default.

## 2. Real streaming/non-streaming pool work, timing, count-only logs

With the node online, use an existing pool consumer credential in
`POOL_API_KEY`. Arrange through pool operator tooling for these requests to
reach this node, or confirm actual traffic to this node before measuring the
status deadline. A completion routed to another node cannot validate this Mac.

```sh
POOL_BASE=https://rapidserver.quicksilverpro.io/pool/v1
curl --silent --show-error --max-time 90 "$POOL_BASE/chat/completions" \
  -H "Authorization: Bearer $POOL_API_KEY" -H 'Content-Type: application/json' \
  --data-binary '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"PR4516-PRIVATE-SENTINEL"}],"max_tokens":32,"stream":false}' \
  >"$DOGFOOD_DIR/nonstream.private.json"
curl --silent --show-error --no-buffer --max-time 90 "$POOL_BASE/chat/completions" \
  -H "Authorization: Bearer $POOL_API_KEY" -H 'Content-Type: application/json' \
  --data-binary '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"PR4516-PRIVATE-SENTINEL"}],"max_tokens":32,"stream":true,"stream_options":{"include_usage":true}}' \
  >"$DOGFOOD_DIR/stream.private.sse"
rg 'QuickSilver status:' "$DOGFOOD_DIR/share.err"
rg 'PR4516-PRIVATE-SENTINEL' "$DOGFOOD_DIR/share.err"
```

Expect the first status within 15 seconds of a confirmed served pool request,
then at most one line per 60 seconds when counts or connection state change.
No status before real relay work; warm-up accounted within the startup bound
must contribute zero (the accepted timed-out residual above may contribute one
request and a few tokens). Compare
request/token increases with the returned usage and actual node assignment,
allowing for other pool traffic/readiness requests. Exercise accepted MTP
requests and one-token completions; the latter have no decode-rate observation.
Compare average tok/s with the model performance metrics' sum/count deltas,
not with aggregate wall-clock throughput. No sentinel, output text, customer
identifier, or credential should appear in the status lines. The sentinel
search of the supervisor log must find nothing.

Leave serving for at least 24 hours with periods of traffic and idle time.
Check timestamps around emissions: no interval under 60 seconds, no periodic
unchanged idle line, and at most 1,440 status lines per 24 hours. Other existing
supervisor/reconnect messages have their own frequency; count only lines
starting `QuickSilver status:`. Logs grow over time; this PR does not add log
rotation or a total-size limit.

## 3. Disconnects, child death, SIGTERM and Ctrl-C

For a Mac whose pool traffic uses Wi-Fi, identify its device:

```sh
networksetup -listallhardwareports
# Set this to the Device listed for Wi-Fi, e.g. en0.
DOGFOOD_WIFI_IFACE=en0
networksetup -setairportpower "$DOGFOOD_WIFI_IFACE" off
sleep 75
networksetup -setairportpower "$DOGFOOD_WIFI_IFACE" on
```

Expect `disconnected, reconnecting`, then `connected` under the same 60-second
status throttle. Repeat with several shorter disconnect/reconnect cycles;
counts must survive reconnects and rapid flapping must not create extra status
lines. Restore connectivity after the test.

On a background run, terminate only its serve child:

```sh
SERVE_PID=$(pgrep -P "$DOGFOOD_PID" -f 'rapid_mlx.cli serve')
kill -TERM "$SERVE_PID"
wait "$DOGFOOD_PID"
lsof -nP -iTCP:18765 -sTCP:LISTEN
```

Expect share to leave the pool and return a failure; no lingering listener or
reporter process. Restart for additional SIGTERM and foreground Ctrl-C tests.
Shutdown time with faulty metrics below must not increase relative to the same
cached healthy setup's normal tunnel/heartbeat/child teardown.

## 4. Faulty metrics while actual serving and heartbeat continue

Use this foreground bootstrap to inject metrics-only failures into the
candidate while preserving all real serving, heartbeat, and relay calls. Use
`unavailable` first, then `blocked`. Stop each with Ctrl-C; SIGTERM can also be
sent to this bootstrap's PID from another terminal.

```sh
RAPID_MLX_AUTO_PULL=1 HF_HOME="$DOGFOOD_DIR/hf" python - blocked "$ALIAS" <<'PY'
import sys
import threading
from rapid_mlx.share import quicksilver as qs
from rapid_mlx.cli import main

mode, alias = sys.argv[1:]
real_open = qs._open

def injected(req, timeout):
    if req.full_url.endswith('/metrics'):
        if mode == 'blocked':
            threading.Event().wait(300)
        raise TimeoutError('PR4516-PRIVATE-EXCEPTION-SENTINEL')
    return real_open(req, timeout=timeout)

qs._open = injected
sys.argv = ['rapid-mlx', 'share', alias, '--quicksilver',
            '--worker', 'pr4516-dogfood', '--port', '18765']
main()
PY
```

Startup must continue despite unavailable/dripping metrics, pool requests must
still complete, heartbeat must continue, and stopping must not wait 300
seconds. Expect no traceback or exception sentinel. Only one metrics worker
may remain blocked; it must be a daemon. Real network timeout/redirect/body
handling is covered by the sandbox's transport and wire-response harnesses;
repeat against an isolated local fault server on the real Mac if desired.

## 5. External-volume cache and existing LaunchAgent

On a dedicated test volume, create a fresh HF cache using the explicit
`HF_HUB_CACHE` environment. Before the cold download starts, point that
repository's `blobs` directory at an empty test directory on the other volume:

```sh
REPO=$(python -c 'import sys; from rapid_mlx.model_aliases import resolve_profile; print(resolve_profile(sys.argv[1]).hf_path)' "$ALIAS")
CACHE_REPO="models--${REPO//\//--}"
DOGFOOD_EXTERNAL=$(mktemp -d /Volumes/DogfoodDisk/rmlx-pr4516.XXXXXX)
mkdir -p "$DOGFOOD_DIR/external-hub/$CACHE_REPO"
ln -s "$DOGFOOD_EXTERNAL" "$DOGFOOD_DIR/external-hub/$CACHE_REPO/blobs"
RAPID_MLX_AUTO_PULL=1 HF_HUB_CACHE="$DOGFOOD_DIR/external-hub" rapid-mlx share \
  "$ALIAS" --quicksilver --worker pr4516-dogfood --port 18765
```

Expect nonzero cache occupancy on the actual second volume and no double count
for snapshots/hardlinks. Exercise removable-volume access denial/disconnection
on this isolated test cache: a stuck scanner must not freeze health supervision
or shutdown; a failed/pending scan reports unavailable or the last completed
observation. Restore the volume and its access afterward. This sandbox could
not exercise macOS TCC or an actual second filesystem/device.

For an existing test LaunchAgent that already runs the candidate, inspect its
configured error log (no service installation/change is part of this PR):

```sh
LABEL="com.quicksilver.node.$CATALOG"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SERVICE_ERR=$(/usr/libexec/PlistBuddy -c 'Print :StandardErrorPath' "$PLIST")
launchctl kickstart -k "gui/$(id -u)/$LABEL"
tail -F "$SERVICE_ERR"
```

Confirm flushed newline-only status lines without a TTY, no private sentinel
or credentials, throttling over 24 hours, and proper cleanup/restart. The
existing LaunchAgent must point at the candidate executable and have its own
correct cache environment. Record commit, Mac model/RAM, macOS version, model,
cache layout, timings, usage deltas, and redacted count-only log evidence.
