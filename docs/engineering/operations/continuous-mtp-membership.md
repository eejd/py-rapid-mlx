# Continuous MTP membership

Continuous routing retains fixed cohorts by default. To opt into dynamic joins and singleton absorption on a qualified target, use the existing controls:

```json
{"method":"mtp","continuous_batching":true,"allow_dynamic_membership":true}
```

Omit `allow_dynamic_membership` or set it to `false` to keep fixed cohorts. This does not change model qualification or authorize an unqualified target.

New cohorts and joins require valid memory-headroom telemetry, a projected lane footprint, room under the lane limit, and the hard reserve. Unknown or failed telemetry keeps the ordinary queue or singleton as owner. Pending joins are charged before admitting additional lanes.

Singleton absorption reserves the exact remaining-token/cache snapshot before removing its verifier. Refused admission or removal retains the singleton; a staged join is cancelled on removal refusal. If deferred preparation fails after a successful handoff, the original resumable tuple returns to the ordinary queue. No token replay or regenerated cache is substituted.
