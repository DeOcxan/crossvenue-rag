# Answer evaluation

Judged by opus from results/20260925T202827Z_answers; answers copied unchanged. matrix v2, methodology section 9a

114 questions x 4 configurations; top-8 chunks. Judge: opus (anthropic/claude-opus-5.5).

## Metrics per configuration

| config | answers | errors | refused | asked | faithfulness | venue attribution error | coverage | behaviour | coverage (comparison) | venue error (comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 114 | 0 | 31 | 0 | 0.984 | 0.016 | 0.620 | 0.781 | 0.538 | 0.000 |
| qwen@venue_split | 114 | 0 | 19 | 1 | 0.946 | 0.049 | 0.690 | 0.866 | 0.631 | 0.000 |
| mercury@venue_split | 114 | 0 | 6 | 1 | 0.910 | 0.019 | 0.606 | 0.833 | 0.553 | 0.000 |
| gemini@hybrid_rerank | 114 | 0 | 33 | 0 | 0.983 | 0.021 | 0.600 | 0.763 | 0.502 | 0.009 |

## Cost and latency per configuration

Generator tokens: input, reasoning and answer counted separately. USD is the cost OpenRouter reported for each answer; a cached answer carries the cost paid when it was made.

| config | tokens in | reasoning | answer | generator USD | judge USD | p50 ms | p95 ms | invalid responses | HTTP retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 452006 | 0 | 29656 | 0.4502 | 2.9364 | 3860 | 9735 | 3 | 0 |
| qwen@venue_split | 408275 | 57375 | 41576 | 0.2375 | 3.6927 | 19678 | 53807 | 0 | 0 |
| mercury@venue_split | 465517 | 29071 | 45203 | 0.0298 | 3.6299 | 1856 | 122181 | 10 | 0 |
| gemini@hybrid_rerank | 441110 | 0 | 28965 | 0.4395 | 2.8671 | 3628 | 8321 | 1 | 0 |

## Parameters sent

- gemini@venue_split: `{"max_tokens": 4000, "model": "google/gemini-3.8-flash", "provider": {"allow_fallbacks": false, "order": ["google-vertex/global"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "usage": {"include": true}}`
- qwen@venue_split: `{"max_tokens": 4000, "model": "qwen/qwen3.8-27b", "provider": {"allow_fallbacks": false, "order": ["deepinfra/bf16"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "temperature": 0.0, "usage": {"include": true}}`
- mercury@venue_split: `{"max_tokens": 4000, "model": "inception/mercury-2.5", "provider": {"require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "temperature": 0.0, "usage": {"include": true}}`
- gemini@hybrid_rerank: `{"max_tokens": 4000, "model": "google/gemini-3.8-flash", "provider": {"allow_fallbacks": false, "order": ["google-vertex/global"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "usage": {"include": true}}`

## Paired comparisons

a - b over questions both answered and judged; 95% paired bootstrap interval, two-sided randomisation test, Holm within the primary family and within each secondary metric and scope.

| a vs b | metric | scope | n | a - b | 95% CI | randomisation p | role | Holm p |
|---|---|---|---:|---:|---|---:|---|---:|
| gemini@venue_split vs qwen@venue_split | faithfulness | comparison | 41 | +0.064 | [+0.021, +0.112] | 0.0091 | secondary | 0.0546 |
| gemini@venue_split vs qwen@venue_split | faithfulness | all | 81 | +0.038 | [+0.011, +0.069] | 0.0108 | secondary | 0.0432 |
| gemini@venue_split vs qwen@venue_split | venue_attribution_error | comparison | 41 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs qwen@venue_split | venue_attribution_error | all | 81 | -0.025 | [-0.058, +0.000] | 0.2430 | secondary | 1.0000 |
| gemini@venue_split vs qwen@venue_split | coverage | comparison | 50 | -0.092 | [-0.181, -0.013] | 0.0360 | secondary | 0.1080 |
| gemini@venue_split vs qwen@venue_split | coverage | all | 88 | -0.078 | [-0.140, -0.020] | 0.0119 | secondary | 0.0476 |
| gemini@venue_split vs qwen@venue_split | behaviour_correct | comparison | 50 | -0.080 | [-0.160, -0.020] | 0.1262 | secondary | 0.5047 |
| gemini@venue_split vs qwen@venue_split | behaviour_correct | all | 112 | -0.089 | [-0.152, -0.036] | 0.0059 | secondary | 0.0295 |
| gemini@venue_split vs mercury@venue_split | faithfulness | comparison | 41 | +0.087 | [+0.024, +0.154] | 0.0136 | secondary | 0.0680 |
| gemini@venue_split vs mercury@venue_split | faithfulness | all | 83 | +0.066 | [+0.027, +0.107] | 0.0013 | secondary | 0.0065 |
| gemini@venue_split vs mercury@venue_split | venue_attribution_error | comparison | 41 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs mercury@venue_split | venue_attribution_error | all | 83 | +0.004 | [-0.032, +0.040] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs mercury@venue_split | coverage | comparison | 50 | -0.015 | [-0.106, +0.069] | 0.7477 | secondary | 0.7915 |
| gemini@venue_split vs mercury@venue_split | coverage | all | 90 | +0.014 | [-0.045, +0.071] | 0.6407 | secondary | 1.0000 |
| gemini@venue_split vs mercury@venue_split | behaviour_correct | comparison | 50 | -0.100 | [-0.220, +0.000] | 0.1763 | secondary | 0.5288 |
| gemini@venue_split vs mercury@venue_split | behaviour_correct | all | 114 | -0.053 | [-0.132, +0.026] | 0.2643 | secondary | 0.7928 |
| gemini@venue_split vs gemini@hybrid_rerank | faithfulness | comparison | 38 | +0.010 | [-0.031, +0.046] | 0.6292 | secondary | 0.6292 |
| gemini@venue_split vs gemini@hybrid_rerank | faithfulness | all | 80 | +0.000 | [-0.020, +0.019] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | venue_attribution_error | comparison | 38 | -0.009 | [-0.026, +0.000] | 1.0000 | primary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | venue_attribution_error | all | 80 | -0.004 | [-0.013, +0.000] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | coverage | comparison | 50 | +0.037 | [-0.044, +0.117] | 0.3857 | primary | 0.7713 |
| gemini@venue_split vs gemini@hybrid_rerank | coverage | all | 90 | +0.020 | [-0.024, +0.067] | 0.3904 | secondary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | behaviour_correct | comparison | 50 | +0.040 | [-0.040, +0.120] | 0.6268 | secondary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | behaviour_correct | all | 114 | +0.018 | [-0.018, +0.053] | 0.6251 | secondary | 0.8577 |
| qwen@venue_split vs mercury@venue_split | faithfulness | comparison | 47 | +0.058 | [-0.018, +0.132] | 0.1439 | secondary | 0.2878 |
| qwen@venue_split vs mercury@venue_split | faithfulness | all | 95 | +0.049 | [+0.001, +0.097] | 0.0445 | secondary | 0.0890 |
| qwen@venue_split vs mercury@venue_split | venue_attribution_error | comparison | 47 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| qwen@venue_split vs mercury@venue_split | venue_attribution_error | all | 95 | +0.028 | [-0.004, +0.067] | 0.1917 | secondary | 1.0000 |
| qwen@venue_split vs mercury@venue_split | coverage | comparison | 50 | +0.077 | [+0.022, +0.134] | 0.0102 | secondary | 0.0510 |
| qwen@venue_split vs mercury@venue_split | coverage | all | 88 | +0.093 | [+0.050, +0.139] | 0.0001 | secondary | 0.0006 |
| qwen@venue_split vs mercury@venue_split | behaviour_correct | comparison | 50 | -0.020 | [-0.100, +0.060] | 1.0000 | secondary | 1.0000 |
| qwen@venue_split vs mercury@venue_split | behaviour_correct | all | 112 | +0.036 | [-0.027, +0.098] | 0.4289 | secondary | 0.8577 |
| qwen@venue_split vs gemini@hybrid_rerank | faithfulness | comparison | 39 | -0.058 | [-0.116, -0.004] | 0.0596 | secondary | 0.1788 |
| qwen@venue_split vs gemini@hybrid_rerank | faithfulness | all | 79 | -0.039 | [-0.072, -0.008] | 0.0180 | secondary | 0.0540 |
| qwen@venue_split vs gemini@hybrid_rerank | venue_attribution_error | comparison | 39 | -0.009 | [-0.026, +0.000] | 1.0000 | secondary | 1.0000 |
| qwen@venue_split vs gemini@hybrid_rerank | venue_attribution_error | all | 79 | +0.021 | [-0.004, +0.055] | 0.3720 | secondary | 1.0000 |
| qwen@venue_split vs gemini@hybrid_rerank | coverage | comparison | 50 | +0.129 | [+0.032, +0.230] | 0.0135 | secondary | 0.0540 |
| qwen@venue_split vs gemini@hybrid_rerank | coverage | all | 88 | +0.099 | [+0.033, +0.168] | 0.0062 | secondary | 0.0310 |
| qwen@venue_split vs gemini@hybrid_rerank | behaviour_correct | comparison | 50 | +0.120 | [+0.040, +0.220] | 0.0309 | secondary | 0.1854 |
| qwen@venue_split vs gemini@hybrid_rerank | behaviour_correct | all | 112 | +0.107 | [+0.045, +0.170] | 0.0012 | secondary | 0.0072 |
| mercury@venue_split vs gemini@hybrid_rerank | faithfulness | comparison | 39 | -0.086 | [-0.158, -0.024] | 0.0165 | secondary | 0.0680 |
| mercury@venue_split vs gemini@hybrid_rerank | faithfulness | all | 81 | -0.069 | [-0.111, -0.032] | 0.0004 | secondary | 0.0024 |
| mercury@venue_split vs gemini@hybrid_rerank | venue_attribution_error | comparison | 39 | -0.009 | [-0.026, +0.000] | 1.0000 | secondary | 1.0000 |
| mercury@venue_split vs gemini@hybrid_rerank | venue_attribution_error | all | 81 | -0.008 | [-0.045, +0.029] | 0.7477 | secondary | 1.0000 |
| mercury@venue_split vs gemini@hybrid_rerank | coverage | comparison | 50 | +0.052 | [-0.064, +0.169] | 0.3958 | secondary | 0.7915 |
| mercury@venue_split vs gemini@hybrid_rerank | coverage | all | 90 | +0.006 | [-0.066, +0.080] | 0.8683 | secondary | 1.0000 |
| mercury@venue_split vs gemini@hybrid_rerank | behaviour_correct | comparison | 50 | +0.140 | [+0.040, +0.260] | 0.0387 | secondary | 0.1935 |
| mercury@venue_split vs gemini@hybrid_rerank | behaviour_correct | all | 114 | +0.070 | [-0.009, +0.149] | 0.1140 | secondary | 0.4560 |
