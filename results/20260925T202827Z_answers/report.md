# Answer evaluation

generator prompt v2 (methodology 9a, exploratory, D10); no judge

Spent this run: $0.9087 of a $3.00 budget.

per_answer.csv, summary.csv, paired.csv and this report were rewritten from the unchanged answers.jsonl under the outcome rule of methodology section 9a.

114 questions x 4 configurations; top-8 chunks. Judge: none (generation only).

## Metrics per configuration

| config | answers | errors | refused | asked | faithfulness | venue attribution error | coverage | behaviour | coverage (comparison) | venue error (comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 114 | 0 | 31 | 0 | - | - | - | - | - | - |
| qwen@venue_split | 114 | 0 | 19 | 1 | - | - | - | - | - | - |
| mercury@venue_split | 114 | 0 | 6 | 1 | - | - | - | - | - | - |
| gemini@hybrid_rerank | 114 | 0 | 33 | 0 | - | - | - | - | - | - |

## Cost and latency per configuration

Generator tokens: input, reasoning and answer counted separately. USD is the cost OpenRouter reported for each answer; a cached answer carries the cost paid when it was made.

| config | tokens in | reasoning | answer | generator USD | judge USD | p50 ms | p95 ms | invalid responses | HTTP retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 452006 | 0 | 29656 | 0.4502 | 0.0000 | 3860 | 9735 | 3 | 0 |
| qwen@venue_split | 408275 | 57375 | 41576 | 0.2375 | 0.0000 | 19678 | 53807 | 0 | 0 |
| mercury@venue_split | 465517 | 29071 | 45203 | 0.0298 | 0.0000 | 1856 | 122181 | 10 | 0 |
| gemini@hybrid_rerank | 441110 | 0 | 28965 | 0.4395 | 0.0000 | 3628 | 8321 | 1 | 0 |

## Parameters sent

- gemini@venue_split: `{"max_tokens": 4000, "model": "google/gemini-3.8-flash", "provider": {"allow_fallbacks": false, "order": ["google-vertex/global"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "usage": {"include": true}}`
- qwen@venue_split: `{"max_tokens": 4000, "model": "qwen/qwen3.8-27b", "provider": {"allow_fallbacks": false, "order": ["deepinfra/bf16"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "temperature": 0.0, "usage": {"include": true}}`
- mercury@venue_split: `{"max_tokens": 4000, "model": "inception/mercury-2.5", "provider": {"require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "temperature": 0.0, "usage": {"include": true}}`
- gemini@hybrid_rerank: `{"max_tokens": 4000, "model": "google/gemini-3.8-flash", "provider": {"allow_fallbacks": false, "order": ["google-vertex/global"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "usage": {"include": true}}`

## Paired comparisons

a - b over questions both answered and judged; 95% paired bootstrap interval, two-sided randomisation test, Holm within the primary family and within each secondary metric and scope.

| a vs b | metric | scope | n | a - b | 95% CI | randomisation p | role | Holm p |
|---|---|---|---:|---:|---|---:|---|---:|
