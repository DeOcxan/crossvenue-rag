# Answer evaluation

- `index_version`: `2825a7160c4d09bf`

Judged by opus from `results/20260925T202827Z_answers`; answers copied unchanged.

114 questions × 4 configurations; top-8 chunks. Judge: opus (anthropic/claude-opus-5.5).

Uncalibrated: `faithfulness`. The judge's dimension behind each failed its pre-registered agreement threshold with the human ratings (methodology section 9); these numbers are the judge's grades only and support no conclusion.

## Metrics per configuration

| config | answers | errors | refused | asked | faithfulness (uncalibrated) | venue attribution error | coverage | behavior | coverage (comparison) | venue error (comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 114 | 0 | 31 | 0 | 0.986 | 0.016 | 0.641 | 0.807 | 0.539 | 0.000 |
| qwen@venue_split | 114 | 0 | 19 | 1 | 0.953 | 0.048 | 0.698 | 0.895 | 0.639 | 0.000 |
| mercury@venue_split | 114 | 0 | 6 | 1 | 0.910 | 0.019 | 0.621 | 0.842 | 0.546 | 0.000 |
| gemini@hybrid_rerank | 114 | 0 | 33 | 0 | 0.985 | 0.021 | 0.623 | 0.798 | 0.507 | 0.009 |

## Cost and call duration per configuration

Generator tokens: input, reasoning and answer counted separately. USD is the cost OpenRouter reported for each answer; a cached answer carries the cost paid when it was made. p50 and p95 are the duration of the generator's calls, in seconds.

| config | tokens in | reasoning | answer | generator USD | judge USD | p50 s | p95 s | invalid responses | HTTP retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 452,006 | 0 | 29,656 | $0.450 | $2.798 | 4 | 10 | 3 | 0 |
| qwen@venue_split | 408,275 | 57,375 | 41,576 | $0.238 | $3.502 | 20 | 54 | 0 | 0 |
| mercury@venue_split | 465,517 | 29,071 | 45,203 | $0.030 | $3.660 | 2 | 122 | 10 | 0 |
| gemini@hybrid_rerank | 441,110 | 0 | 28,965 | $0.439 | $2.722 | 4 | 8 | 1 | 0 |

## Parameters sent

- gemini@venue_split: `max_tokens=4000`, `model=google/gemini-3.8-flash`, `provider.allow_fallbacks=false`, `provider.order=google-vertex/global`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `usage.include=true`
- qwen@venue_split: `max_tokens=4000`, `model=qwen/qwen3.8-27b`, `provider.allow_fallbacks=false`, `provider.order=deepinfra/bf16`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `temperature=0.0`, `usage.include=true`
- mercury@venue_split: `max_tokens=4000`, `model=inception/mercury-2.5`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `temperature=0.0`, `usage.include=true`
- gemini@hybrid_rerank: `max_tokens=4000`, `model=google/gemini-3.8-flash`, `provider.allow_fallbacks=false`, `provider.order=google-vertex/global`, `provider.require_parameters=true`, `reasoning.effort=minimal`, `reasoning.exclude=true`, `response_format=json_schema`, `usage.include=true`

## Paired comparisons

a − b over questions both answered and judged; 95% paired bootstrap interval, two-sided randomization test, Holm within the pre-registered comparison and within each other metric and scope.

| a vs b | metric | scope | n | a − b | 95% CI | randomization p | role | Holm p |
|---|---|---|---:|---:|---|---:|---|---:|
| gemini@venue_split vs qwen@venue_split | `faithfulness` | comparison | 41 | +0.067 | [+0.026, +0.114] | 0.008 | secondary, uncalibrated | 0.032 |
| gemini@venue_split vs qwen@venue_split | `faithfulness` | all | 83 | +0.033 | [+0.009, +0.060] | 0.019 | secondary, uncalibrated | 0.057 |
| gemini@venue_split vs qwen@venue_split | `venue_attribution_error` | comparison | 41 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs qwen@venue_split | `venue_attribution_error` | all | 83 | −0.024 | [−0.056, +0.000] | 0.249 | secondary | 1.000 |
| gemini@venue_split vs qwen@venue_split | `coverage` | comparison | 50 | −0.099 | [−0.186, −0.022] | 0.021 | secondary | 0.063 |
| gemini@venue_split vs qwen@venue_split | `coverage` | all | 89 | −0.057 | [−0.114, −0.005] | 0.049 | secondary | 0.196 |
| gemini@venue_split vs qwen@venue_split | `behaviour_correct` | comparison | 50 | −0.100 | [−0.180, −0.020] | 0.064 | secondary | 0.322 |
| gemini@venue_split vs qwen@venue_split | `behaviour_correct` | all | 114 | −0.088 | [−0.149, −0.035] | 0.007 | secondary | 0.036 |
| gemini@venue_split vs mercury@venue_split | `faithfulness` | comparison | 41 | +0.107 | [+0.043, +0.174] | 0.002 | secondary, uncalibrated | 0.013 |
| gemini@venue_split vs mercury@venue_split | `faithfulness` | all | 83 | +0.071 | [+0.034, +0.110] | < 0.001 | secondary, uncalibrated | 0.001 |
| gemini@venue_split vs mercury@venue_split | `venue_attribution_error` | comparison | 41 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs mercury@venue_split | `venue_attribution_error` | all | 83 | +0.004 | [−0.032, +0.040] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs mercury@venue_split | `coverage` | comparison | 50 | −0.007 | [−0.096, +0.076] | 0.884 | secondary | 1.000 |
| gemini@venue_split vs mercury@venue_split | `coverage` | all | 89 | +0.019 | [−0.040, +0.077] | 0.530 | secondary | 1.000 |
| gemini@venue_split vs mercury@venue_split | `behaviour_correct` | comparison | 50 | −0.100 | [−0.220, +0.000] | 0.176 | secondary | 0.529 |
| gemini@venue_split vs mercury@venue_split | `behaviour_correct` | all | 114 | −0.035 | [−0.105, +0.035] | 0.480 | secondary | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `faithfulness` | comparison | 38 | +0.010 | [−0.031, +0.046] | 0.629 | secondary, uncalibrated | 0.629 |
| gemini@venue_split vs gemini@hybrid_rerank | `faithfulness` | all | 80 | +0.000 | [−0.020, +0.019] | 1.000 | secondary, uncalibrated | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | comparison | 38 | −0.009 | [−0.026, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | all | 80 | −0.004 | [−0.013, +0.000] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `coverage` | comparison | 50 | +0.032 | [−0.045, +0.110] | 0.427 | secondary | 0.855 |
| gemini@venue_split vs gemini@hybrid_rerank | `coverage` | all | 89 | +0.018 | [−0.025, +0.062] | 0.435 | secondary | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | comparison | 50 | +0.020 | [−0.040, +0.080] | 1.000 | secondary | 1.000 |
| gemini@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | all | 114 | +0.009 | [−0.018, +0.044] | 1.000 | secondary | 1.000 |
| qwen@venue_split vs mercury@venue_split | `faithfulness` | comparison | 47 | +0.067 | [−0.002, +0.133] | 0.064 | secondary, uncalibrated | 0.131 |
| qwen@venue_split vs mercury@venue_split | `faithfulness` | all | 97 | +0.053 | [+0.015, +0.093] | 0.008 | secondary, uncalibrated | 0.034 |
| qwen@venue_split vs mercury@venue_split | `venue_attribution_error` | comparison | 47 | +0.000 | [+0.000, +0.000] | 1.000 | secondary | 1.000 |
| qwen@venue_split vs mercury@venue_split | `venue_attribution_error` | all | 97 | +0.027 | [−0.007, +0.065] | 0.184 | secondary | 1.000 |
| qwen@venue_split vs mercury@venue_split | `coverage` | comparison | 50 | +0.092 | [+0.045, +0.145] | < 0.001 | secondary | 0.003 |
| qwen@venue_split vs mercury@venue_split | `coverage` | all | 89 | +0.076 | [+0.041, +0.115] | < 0.001 | secondary | < 0.001 |
| qwen@venue_split vs mercury@venue_split | `behaviour_correct` | comparison | 50 | +0.000 | [−0.080, +0.080] | 1.000 | secondary | 1.000 |
| qwen@venue_split vs mercury@venue_split | `behaviour_correct` | all | 114 | +0.053 | [−0.009, +0.114] | 0.185 | secondary | 0.742 |
| qwen@venue_split vs gemini@hybrid_rerank | `faithfulness` | comparison | 39 | −0.061 | [−0.118, −0.009] | 0.044 | secondary, uncalibrated | 0.131 |
| qwen@venue_split vs gemini@hybrid_rerank | `faithfulness` | all | 81 | −0.034 | [−0.063, −0.007] | 0.027 | secondary, uncalibrated | 0.057 |
| qwen@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | comparison | 39 | −0.009 | [−0.026, +0.000] | 1.000 | secondary | 1.000 |
| qwen@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | all | 81 | +0.021 | [−0.004, +0.053] | 0.376 | secondary | 1.000 |
| qwen@venue_split vs gemini@hybrid_rerank | `coverage` | comparison | 50 | +0.132 | [+0.036, +0.234] | 0.012 | secondary | 0.047 |
| qwen@venue_split vs gemini@hybrid_rerank | `coverage` | all | 89 | +0.075 | [+0.013, +0.140] | 0.023 | secondary | 0.113 |
| qwen@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | comparison | 50 | +0.120 | [+0.040, +0.220] | 0.032 | secondary | 0.190 |
| qwen@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | all | 114 | +0.096 | [+0.035, +0.158] | 0.004 | secondary | 0.027 |
| mercury@venue_split vs gemini@hybrid_rerank | `faithfulness` | comparison | 39 | −0.108 | [−0.177, −0.043] | 0.003 | secondary, uncalibrated | 0.013 |
| mercury@venue_split vs gemini@hybrid_rerank | `faithfulness` | all | 81 | −0.075 | [−0.113, −0.038] | < 0.001 | secondary, uncalibrated | < 0.001 |
| mercury@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | comparison | 39 | −0.009 | [−0.026, +0.000] | 1.000 | secondary | 1.000 |
| mercury@venue_split vs gemini@hybrid_rerank | `venue_attribution_error` | all | 81 | −0.008 | [−0.045, +0.029] | 0.748 | secondary | 1.000 |
| mercury@venue_split vs gemini@hybrid_rerank | `coverage` | comparison | 50 | +0.039 | [−0.072, +0.154] | 0.511 | secondary | 1.000 |
| mercury@venue_split vs gemini@hybrid_rerank | `coverage` | all | 89 | −0.001 | [−0.072, +0.070] | 0.979 | secondary | 1.000 |
| mercury@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | comparison | 50 | +0.120 | [+0.020, +0.220] | 0.070 | secondary | 0.322 |
| mercury@venue_split vs gemini@hybrid_rerank | `behaviour_correct` | all | 114 | +0.044 | [−0.026, +0.114] | 0.340 | secondary | 1.000 |
