# 0009. Three generators from three model families, through OpenRouter

- Date: 2026-09-25
- Status: Accepted

## Context

On the 50 comparison questions of `v0.2.1-gold`, venue_split puts every named venue in the top 10 on 0.98 (49/50) of them, against 0.70 (35/50) for hybrid_rerank (`results/20260925T012314Z`). Whether that reaches the answer depends on the generator: one model may keep venues apart in any context, another may merge them even when both are present. A single generator would confound the retrieval effect with that model's habits.

OpenRouter serves all three generator families and both judge candidates through one OpenAI-compatible API and one key, and for a model several providers may serve it at different precision, price and parameter support. An alias such as "latest" moves under a running study.

In a five-question trial on 2026-09-25:

- `google/gemini-3.8-flash` had no available AI Studio endpoint; the Vertex endpoint accepted the JSON schema and returned valid JSON, and rejected any temperature. Sent without requiring every parameter, the request fell to an endpoint that dropped the schema and returned prose.
- At reasoning effort low, Qwen averaged 57 s per answer and once took over 300 s. At effort minimal it used 371 to 416 reasoning tokens and took 23 to 31 s on a comparison question. Gemini refuses to disable reasoning; at minimal it reported no reasoning tokens. Mercury at minimal used 257 reasoning tokens in 1.8 s.

## Decision

- Every model call goes through OpenRouter, the project's only model provider, with one key (`OPENROUTER_API_KEY`).
- Three generators, each from a different family:
  - `google/gemini-3.8-flash`: a proprietary model, the reference generator, and the one the primary comparison holds fixed;
  - `qwen/qwen3.8-27b`: an open-weight model, so the weights behind a result can be named and served again;
  - `inception/mercury-2.5`: a diffusion language model, a different decoding architecture, and the cheapest of the three ($0.04 / $0.15 per million input / output tokens on 2026-09-25).
- Exact model ids in `configs/default.yaml`; the config refuses an id containing "latest". Every call records the id requested and the id and provider the response reports.
- Providers are pinned where the choice matters, with fallback disabled and every parameter required (`require_parameters`):
  - Gemini on OpenRouter's `google-vertex/global` endpoint, the one that enforces the JSON schema; it accepts no temperature, so none is sent (`temperature_supported: false`);
  - Qwen on DeepInfra's bf16 endpoint. On 2026-09-25 OpenRouter listed 16 providers for it. DeepInfra serves the reference bf16 precision. It has the lowest output price among providers of known precision ($0.15 / $1.875 per million tokens), supports structured output, and had 99.1% uptime over the 24 hours before that check. Parasail had higher uptime (99.97%) at fp8 and a higher price; Novita and Alibaba do not support structured output.
- One rule for every generator: JSON-schema output; reasoning effort minimal, the lowest level all three accept; at most 4,000 output tokens; temperature 0 where the endpoint accepts it. A response that is not valid JSON or fails its schema is asked for again, at most twice. The parameters actually sent, reasoning tokens (apart from answer tokens), invalid responses and HTTP retries are written into every result.
- Cost is the cost OpenRouter reports in each response. Each run has a budget (`generation.budget_usd`) and stops before a call whose expected cost, the most expensive known call of that model, would pass it; the first call of a model with no cached call is not bounded.

## Consequences

- A difference between generators is a difference between models under the same prompt, retrieved chunks and parameter rule, except temperature, which Gemini's endpoint does not take.
- Pinning trades availability for reproducibility: if a pinned endpoint is down, calls fail rather than silently moving to another precision or parameter set.
- Mercury is not pinned; the provider each response reports is recorded, so a provider change is visible in `per_answer.csv`.
