# Answer evaluation

- `index_version`: `2825a7160c4d09bf`

Judged by opus from `results/20260925T175128Z_answers`; answers copied unchanged.

114 questions × 4 configurations; top-8 chunks. Judge: opus (anthropic/claude-opus-5.5).

Uncalibrated: `faithfulness`. The judge's dimension behind each failed its pre-registered agreement threshold with the human ratings (methodology section 9); these numbers are the judge's grades only and support no conclusion.

## Metrics per configuration

| config | answers | errors | refused | asked | faithfulness (uncalibrated) | venue attribution error | coverage | behavior | coverage (comparison) | venue error (comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 114 | 0 | 51 | 0 | 0.995 | 0.000 | 0.587 | 0.675 | 0.454 | 0.000 |
| qwen@venue_split | 114 | 0 | 39 | 0 | 0.943 | 0.037 | 0.658 | 0.772 | 0.586 | 0.000 |
| mercury@venue_split | 114 | 0 | 20 | 2 | 0.944 | 0.013 | 0.582 | 0.833 | 0.500 | 0.000 |
| gemini@hybrid_rerank | 114 | 0 | 55 | 0 | 0.997 | 0.000 | 0.493 | 0.632 | 0.286 | 0.000 |

## Cost and call duration per configuration

Generator tokens: input, reasoning and answer counted separately. USD is the cost OpenRouter reported for each answer; a cached answer carries the cost paid when it was made. p50 and p95 are the duration of the generator's calls, in seconds.

| config | tokens in | reasoning | answer | generator USD | judge USD | p50 s | p95 s | invalid responses | HTTP retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 396,830 | 203 | 21,304 | $0.378 | $2.639 | 3 | 6 | 1 | 0 |
| qwen@venue_split | 366,437 | 46,229 | 30,319 | $0.198 | $3.409 | 21 | 72 | 0 | 0 |
| mercury@venue_split | 399,057 | 28,576 | 35,860 | $0.026 | $3.418 | 2 | 122 | 7 | 0 |
| gemini@hybrid_rerank | 385,934 | 203 | 20,169 | $0.366 | $2.555 | 3 | 6 | 1 | 0 |

## Parameters sent

- gemini@venue_split: `max_tokens=4000`, `model=google/gemini-3.8-flash`, `provider.allow_fallbacks=false`, `provider.order=google-vertex/global`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `usage.include=true`
- qwen@venue_split: `max_tokens=4000`, `model=qwen/qwen3.8-27b`, `provider.allow_fallbacks=false`, `provider.order=deepinfra/bf16`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `temperature=0.0`, `usage.include=true`
- mercury@venue_split: `max_tokens=4000`, `model=inception/mercury-2.5`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `temperature=0.0`, `usage.include=true`
- gemini@hybrid_rerank: `max_tokens=4000`, `model=google/gemini-3.8-flash`, `provider.allow_fallbacks=false`, `provider.order=google-vertex/global`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `usage.include=true`

## Paired comparisons

a − b over questions both answered and judged; 95% paired bootstrap interval, two-sided randomization test, Holm within the pre-registered comparison and within each other metric and scope.

| a vs b | metric | scope | n | a − b | 95% CI | randomization p | role | Holm p |
|---|---|---|---:|---:|---|---:|---|---:|
| gemini@venue_split vs qwen@venue_split | `faithfulness` | comparison | 26 | +0.021 | [−0.026, +0.082] | 0.625 | secondary, uncalibrated | 1.000 |
| gemini@venue_split vs qwen@venue_split | `faithfulness` | all | 64 | +0.029 | [+0.001, +0.063] | 0.087 | secondary, uncalibrated | 0.260 |
| gemini@venue_split vs qwen@venue_split | `venue_attribution_error` | comparison | 26 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs qwen@venue_split | `venue_attribution_error` | all | 64 | −0.042 | [−0.094, +0.000] | 0.249 | secondary | 1.000 |
| gemini@venue_split vs qwen@venue_split | `coverage` | comparison | 50 | −0.132 | [−0.229, −0.041] | 0.007 | secondary | 0.022 |
| gemini@venue_split vs qwen@venue_split | `coverage` | all | 89 | −0.071 | [−0.133, −0.015] | 0.021 | secondary | 0.064 |
| gemini@venue_split vs qwen@venue_split | `behaviour_correct` | comparison | 50 | −0.160 | [−0.280, −0.040] | 0.021 | secondary | 0.062 |
| gemini@venue_split vs qwen@venue_split | `behaviour_correct` | all | 114 | −0.096 | [−0.167, −0.035] | 0.007 | secondary | 0.022 |
| gemini@venue_split vs mercury@venue_split | `faithfulness` | comparison | 26 | +0.037 | [+0.000, +0.085] | 0.257 | secondary, uncalibrated | 1.000 |
| gemini@venue_split vs mercury@venue_split | `faithfulness` | all | 64 | +0.041 | [+0.015, +0.072] | 0.008 | secondary, uncalibrated | 0.044 |
| gemini@venue_split vs mercury@venue_split | `venue_attribution_error` | comparison | 26 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs mercury@venue_split | `venue_attribution_error` | all | 64 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs mercury@venue_split | `coverage` | comparison | 50 | −0.045 | [−0.155, +0.058] | 0.409 | secondary | 0.409 |
| gemini@venue_split vs mercury@venue_split | `coverage` | all | 89 | +0.005 | [−0.064, +0.070] | 0.895 | secondary | 0.895 |
| gemini@venue_split vs mercury@venue_split | `behaviour_correct` | comparison | 50 | −0.280 | [−0.400, −0.160] | < 0.001 | secondary | 0.001 |
| gemini@venue_split vs mercury@venue_split | `behaviour_correct` | all | 114 | −0.158 | [−0.246, −0.079] | < 0.001 | secondary | 0.002 |
| gemini@venue_split vs gemini@hybrid_rerank | `faithfulness` | comparison | 17 | +0.012 | [+0.000, +0.035] | 1.000 | secondary, uncalibrated | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `faithfulness` | all | 55 | +0.004 | [+0.000, +0.011] | 1.000 | secondary, uncalibrated | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | comparison | 17 | +0.000 | [+0.000, +0.000] | 1.000 | primary | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | all | 55 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `coverage` | comparison | 50 | +0.168 | [+0.048, +0.292] | 0.009 | primary | 0.019 |
| gemini@venue_split vs gemini@hybrid_rerank | `coverage` | all | 89 | +0.094 | [+0.026, +0.166] | 0.010 | secondary | 0.049 |
| gemini@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | comparison | 50 | +0.100 | [−0.020, +0.240] | 0.231 | secondary | 0.231 |
| gemini@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | all | 114 | +0.044 | [−0.009, +0.105] | 0.228 | secondary | 0.282 |
| qwen@venue_split vs mercury@venue_split | `faithfulness` | comparison | 38 | +0.023 | [−0.040, +0.085] | 0.483 | secondary, uncalibrated | 1.000 |
| qwen@venue_split vs mercury@venue_split | `faithfulness` | all | 84 | −0.007 | [−0.054, +0.037] | 0.764 | secondary, uncalibrated | 1.000 |
| qwen@venue_split vs mercury@venue_split | `venue_attribution_error` | comparison | 38 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| qwen@venue_split vs mercury@venue_split | `venue_attribution_error` | all | 84 | +0.038 | [+0.006, +0.079] | 0.125 | secondary | 0.753 |
| qwen@venue_split vs mercury@venue_split | `coverage` | comparison | 50 | +0.086 | [−0.013, +0.179] | 0.085 | secondary | 0.171 |
| qwen@venue_split vs mercury@venue_split | `coverage` | all | 89 | +0.076 | [+0.015, +0.135] | 0.016 | secondary | 0.062 |
| qwen@venue_split vs mercury@venue_split | `behaviour_correct` | comparison | 50 | −0.120 | [−0.220, −0.020] | 0.072 | secondary | 0.144 |
| qwen@venue_split vs mercury@venue_split | `behaviour_correct` | all | 114 | −0.061 | [−0.132, +0.009] | 0.141 | secondary | 0.282 |
| qwen@venue_split vs gemini@hybrid_rerank | `faithfulness` | comparison | 20 | −0.034 | [−0.083, +0.011] | 0.254 | secondary, uncalibrated | 1.000 |
| qwen@venue_split vs gemini@hybrid_rerank | `faithfulness` | all | 58 | −0.034 | [−0.066, −0.008] | 0.025 | secondary, uncalibrated | 0.099 |
| qwen@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | comparison | 20 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| qwen@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | all | 58 | +0.046 | [+0.000, +0.103] | 0.251 | secondary | 1.000 |
| qwen@venue_split vs gemini@hybrid_rerank | `coverage` | comparison | 50 | +0.299 | [+0.163, +0.439] | < 0.001 | secondary | < 0.001 |
| qwen@venue_split vs gemini@hybrid_rerank | `coverage` | all | 89 | +0.165 | [+0.081, +0.256] | < 0.001 | secondary | 0.002 |
| qwen@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | comparison | 50 | +0.260 | [+0.120, +0.400] | 0.001 | secondary | 0.006 |
| qwen@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | all | 114 | +0.140 | [+0.070, +0.219] | < 0.001 | secondary | 0.001 |
| mercury@venue_split vs gemini@hybrid_rerank | `faithfulness` | comparison | 21 | −0.036 | [−0.081, +0.007] | 0.186 | secondary, uncalibrated | 1.000 |
| mercury@venue_split vs gemini@hybrid_rerank | `faithfulness` | all | 59 | −0.041 | [−0.072, −0.014] | 0.007 | secondary, uncalibrated | 0.044 |
| mercury@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | comparison | 21 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| mercury@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | all | 59 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| mercury@venue_split vs gemini@hybrid_rerank | `coverage` | comparison | 50 | +0.213 | [+0.090, +0.343] | 0.002 | secondary | 0.009 |
| mercury@venue_split vs gemini@hybrid_rerank | `coverage` | all | 89 | +0.090 | [+0.011, +0.174] | 0.035 | secondary | 0.069 |
| mercury@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | comparison | 50 | +0.380 | [+0.240, +0.520] | < 0.001 | secondary | < 0.001 |
| mercury@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | all | 114 | +0.202 | [+0.114, +0.289] | < 0.001 | secondary | < 0.001 |
