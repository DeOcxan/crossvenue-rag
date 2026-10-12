# Answer evaluation

Judged by opus from results/20260925T175128Z_answers; answers copied unchanged. matrix v1, methodology section 9

114 questions x 4 configurations; top-8 chunks. Judge: opus (anthropic/claude-opus-5.5).

## Metrics per configuration

| config | answers | errors | refused | asked | faithfulness | venue attribution error | coverage | behaviour | coverage (comparison) | venue error (comparison) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 114 | 0 | 51 | 0 | 0.995 | 0.000 | 0.559 | 0.655 | 0.440 | 0.000 |
| qwen@venue_split | 114 | 0 | 39 | 0 | 0.934 | 0.037 | 0.666 | 0.754 | 0.621 | 0.000 |
| mercury@venue_split | 114 | 0 | 20 | 2 | 0.946 | 0.013 | 0.555 | 0.816 | 0.472 | 0.000 |
| gemini@hybrid_rerank | 114 | 0 | 55 | 0 | 0.997 | 0.000 | 0.476 | 0.623 | 0.283 | 0.000 |

## Cost and latency per configuration

Generator tokens: input, reasoning and answer counted separately. USD is the cost OpenRouter reported for each answer; a cached answer carries the cost paid when it was made.

| config | tokens in | reasoning | answer | generator USD | judge USD | p50 ms | p95 ms | invalid responses | HTTP retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini@venue_split | 396830 | 203 | 21304 | 0.3783 | 2.5889 | 3242 | 5700 | 1 | 0 |
| qwen@venue_split | 366437 | 46229 | 30319 | 0.1985 | 3.5791 | 21496 | 72427 | 0 | 0 |
| mercury@venue_split | 399057 | 28576 | 35860 | 0.0256 | 3.5289 | 1773 | 122283 | 7 | 0 |
| gemini@hybrid_rerank | 385934 | 203 | 20169 | 0.3658 | 2.7346 | 3120 | 5686 | 1 | 0 |

## Parameters sent

- gemini@venue_split: `{"max_tokens": 4000, "model": "google/gemini-3.8-flash", "provider": {"allow_fallbacks": false, "order": ["google-vertex/global"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "usage": {"include": true}}`
- qwen@venue_split: `{"max_tokens": 4000, "model": "qwen/qwen3.8-27b", "provider": {"allow_fallbacks": false, "order": ["deepinfra/bf16"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "temperature": 0.0, "usage": {"include": true}}`
- mercury@venue_split: `{"max_tokens": 4000, "model": "inception/mercury-2.5", "provider": {"require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "temperature": 0.0, "usage": {"include": true}}`
- gemini@hybrid_rerank: `{"max_tokens": 4000, "model": "google/gemini-3.8-flash", "provider": {"allow_fallbacks": false, "order": ["google-vertex/global"], "require_parameters": true}, "reasoning": {"effort": "minimal", "exclude": true}, "response_format": "json_schema", "usage": {"include": true}}`

## Paired comparisons

a - b over questions both answered and judged; 95% paired bootstrap interval, two-sided randomisation test, Holm within the primary family and within each secondary metric and scope.

| a vs b | metric | scope | n | a - b | 95% CI | randomisation p | role | Holm p |
|---|---|---|---:|---:|---|---:|---|---:|
| gemini@venue_split vs qwen@venue_split | faithfulness | comparison | 26 | +0.021 | [-0.026, +0.082] | 0.6249 | secondary | 1.0000 |
| gemini@venue_split vs qwen@venue_split | faithfulness | all | 63 | +0.043 | [+0.010, +0.081] | 0.0232 | secondary | 0.0696 |
| gemini@venue_split vs qwen@venue_split | venue_attribution_error | comparison | 26 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs qwen@venue_split | venue_attribution_error | all | 63 | -0.042 | [-0.095, +0.000] | 0.2526 | secondary | 1.0000 |
| gemini@venue_split vs qwen@venue_split | coverage | comparison | 50 | -0.180 | [-0.270, -0.096] | 0.0002 | secondary | 0.0008 |
| gemini@venue_split vs qwen@venue_split | coverage | all | 89 | -0.103 | [-0.165, -0.046] | 0.0012 | secondary | 0.0048 |
| gemini@venue_split vs qwen@venue_split | behaviour_correct | comparison | 50 | -0.180 | [-0.300, -0.060] | 0.0109 | secondary | 0.0327 |
| gemini@venue_split vs qwen@venue_split | behaviour_correct | all | 113 | -0.097 | [-0.168, -0.035] | 0.0069 | secondary | 0.0207 |
| gemini@venue_split vs mercury@venue_split | faithfulness | comparison | 26 | +0.027 | [+0.000, +0.073] | 0.5028 | secondary | 1.0000 |
| gemini@venue_split vs mercury@venue_split | faithfulness | all | 63 | +0.038 | [+0.012, +0.068] | 0.0166 | secondary | 0.0664 |
| gemini@venue_split vs mercury@venue_split | venue_attribution_error | comparison | 26 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs mercury@venue_split | venue_attribution_error | all | 63 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs mercury@venue_split | coverage | comparison | 50 | -0.032 | [-0.122, +0.056] | 0.5016 | secondary | 0.5016 |
| gemini@venue_split vs mercury@venue_split | coverage | all | 89 | +0.009 | [-0.055, +0.070] | 0.7801 | secondary | 0.7801 |
| gemini@venue_split vs mercury@venue_split | behaviour_correct | comparison | 50 | -0.300 | [-0.420, -0.180] | 0.0002 | secondary | 0.0010 |
| gemini@venue_split vs mercury@venue_split | behaviour_correct | all | 113 | -0.159 | [-0.239, -0.080] | 0.0007 | secondary | 0.0028 |
| gemini@venue_split vs gemini@hybrid_rerank | faithfulness | comparison | 17 | +0.012 | [+0.000, +0.035] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | faithfulness | all | 54 | +0.004 | [+0.000, +0.011] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | venue_attribution_error | comparison | 17 | +0.000 | [+0.000, +0.000] | 1.0000 | primary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | venue_attribution_error | all | 54 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| gemini@venue_split vs gemini@hybrid_rerank | coverage | comparison | 50 | +0.158 | [+0.040, +0.279] | 0.0119 | primary | 0.0238 |
| gemini@venue_split vs gemini@hybrid_rerank | coverage | all | 89 | +0.089 | [+0.021, +0.159] | 0.0135 | secondary | 0.0405 |
| gemini@venue_split vs gemini@hybrid_rerank | behaviour_correct | comparison | 50 | +0.080 | [-0.060, +0.220] | 0.3933 | secondary | 0.3933 |
| gemini@venue_split vs gemini@hybrid_rerank | behaviour_correct | all | 113 | +0.035 | [-0.027, +0.097] | 0.3850 | secondary | 0.3850 |
| qwen@venue_split vs mercury@venue_split | faithfulness | comparison | 38 | +0.017 | [-0.045, +0.077] | 0.6128 | secondary | 1.0000 |
| qwen@venue_split vs mercury@venue_split | faithfulness | all | 84 | -0.020 | [-0.068, +0.026] | 0.4135 | secondary | 0.8269 |
| qwen@venue_split vs mercury@venue_split | venue_attribution_error | comparison | 38 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| qwen@venue_split vs mercury@venue_split | venue_attribution_error | all | 84 | +0.038 | [+0.006, +0.079] | 0.1255 | secondary | 0.7529 |
| qwen@venue_split vs mercury@venue_split | coverage | comparison | 50 | +0.149 | [+0.077, +0.221] | 0.0002 | secondary | 0.0008 |
| qwen@venue_split vs mercury@venue_split | coverage | all | 90 | +0.111 | [+0.061, +0.161] | 0.0001 | secondary | 0.0006 |
| qwen@venue_split vs mercury@venue_split | behaviour_correct | comparison | 50 | -0.120 | [-0.220, -0.020] | 0.0718 | secondary | 0.1436 |
| qwen@venue_split vs mercury@venue_split | behaviour_correct | all | 114 | -0.061 | [-0.132, +0.009] | 0.1408 | secondary | 0.2816 |
| qwen@venue_split vs gemini@hybrid_rerank | faithfulness | comparison | 20 | -0.034 | [-0.083, +0.011] | 0.2538 | secondary | 1.0000 |
| qwen@venue_split vs gemini@hybrid_rerank | faithfulness | all | 58 | -0.049 | [-0.086, -0.017] | 0.0060 | secondary | 0.0360 |
| qwen@venue_split vs gemini@hybrid_rerank | venue_attribution_error | comparison | 20 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| qwen@venue_split vs gemini@hybrid_rerank | venue_attribution_error | all | 58 | +0.046 | [+0.000, +0.103] | 0.2507 | secondary | 1.0000 |
| qwen@venue_split vs gemini@hybrid_rerank | coverage | comparison | 50 | +0.338 | [+0.207, +0.468] | 0.0001 | secondary | 0.0005 |
| qwen@venue_split vs gemini@hybrid_rerank | coverage | all | 90 | +0.190 | [+0.104, +0.276] | 0.0001 | secondary | 0.0006 |
| qwen@venue_split vs gemini@hybrid_rerank | behaviour_correct | comparison | 50 | +0.260 | [+0.120, +0.400] | 0.0015 | secondary | 0.0060 |
| qwen@venue_split vs gemini@hybrid_rerank | behaviour_correct | all | 114 | +0.132 | [+0.061, +0.202] | 0.0004 | secondary | 0.0020 |
| mercury@venue_split vs gemini@hybrid_rerank | faithfulness | comparison | 21 | -0.024 | [-0.069, +0.017] | 0.3784 | secondary | 1.0000 |
| mercury@venue_split vs gemini@hybrid_rerank | faithfulness | all | 59 | -0.037 | [-0.067, -0.011] | 0.0132 | secondary | 0.0660 |
| mercury@venue_split vs gemini@hybrid_rerank | venue_attribution_error | comparison | 21 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| mercury@venue_split vs gemini@hybrid_rerank | venue_attribution_error | all | 59 | +0.000 | [+0.000, +0.000] | 1.0000 | secondary | 1.0000 |
| mercury@venue_split vs gemini@hybrid_rerank | coverage | comparison | 50 | +0.189 | [+0.078, +0.305] | 0.0019 | secondary | 0.0038 |
| mercury@venue_split vs gemini@hybrid_rerank | coverage | all | 90 | +0.079 | [+0.005, +0.157] | 0.0455 | secondary | 0.0910 |
| mercury@venue_split vs gemini@hybrid_rerank | behaviour_correct | comparison | 50 | +0.380 | [+0.240, +0.520] | 0.0001 | secondary | 0.0006 |
| mercury@venue_split vs gemini@hybrid_rerank | behaviour_correct | all | 114 | +0.193 | [+0.105, +0.281] | 0.0001 | secondary | 0.0006 |
