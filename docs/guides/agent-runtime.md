# Legacy Agent Runtime API

Rapid-MLX Desktop no longer uses the server-owned Agent Runtime. Its Chat tab
sends plain chat requests without tools; Goose-powered Work owns agent tasks.

The server still exposes the authenticated, process-local Agent Runtime API.
The existing qualification script calls these routes. This API remains
separate from `rapid-mlx agents <name> --setup`, which configures external
coding agents to call the OpenAI-compatible inference endpoints. The legacy
Agent Runtime API is a follow-up for removal after the qualification tooling
and any other API consumers have been audited and migrated.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/v1/agent/runs` | Create a bounded run |
| `GET` | `/v1/agent/runs/{id}` | Read run status or output |
| `GET` | `/v1/agent/runs/{id}/events?after=N` | Poll run events |
| `POST` | `/v1/agent/runs/{id}/approval` | Resolve an approval |
| `POST` | `/v1/agent/runs/{id}/tool-result` | Return a client tool result |
| `POST` | `/v1/agent/runs/{id}/cancel` | Cancel a run |

All routes use the server's Bearer authentication and rate limiting. Runs are
lost when the server restarts. For ordinary inference, use
`/v1/chat/completions` or `/v1/responses`.
