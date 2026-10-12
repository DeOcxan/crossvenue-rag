# Prompt v1 against v2: results/20260926T000001Z_judged and results/20260926T004836Z_judged

v2 - v1 per configuration, paired on questions; Holm across configurations within each metric. Faithfulness is not compared (its judge dimension failed calibration).

| config | metric | n | v1 | v2 | v2 - v1 | 95% CI | better | worse | test | p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| gemini@venue_split | wrong_refusal | 94 | 0.372 | 0.181 | -0.191 | [-0.277, -0.117] | 18 | 0 | exact McNemar | 0.0000 | 0.0000 |
| gemini@venue_split | venue_attribution_error | 62 | 0.000 | 0.022 | +0.022 | [+0.000, +0.059] | 0 | 2 | paired randomisation | 0.4980 | 1.0000 |
| qwen@venue_split | wrong_refusal | 94 | 0.255 | 0.064 | -0.191 | [-0.277, -0.117] | 18 | 0 | exact McNemar | 0.0000 | 0.0000 |
| qwen@venue_split | venue_attribution_error | 84 | 0.038 | 0.052 | +0.014 | [+0.000, +0.032] | 0 | 3 | paired randomisation | 0.2566 | 1.0000 |
| mercury@venue_split | wrong_refusal | 94 | 0.106 | 0.021 | -0.085 | [-0.149, -0.032] | 8 | 0 | exact McNemar | 0.0078 | 0.0078 |
| mercury@venue_split | venue_attribution_error | 100 | 0.003 | 0.020 | +0.017 | [+0.000, +0.043] | 0 | 2 | paired randomisation | 0.5055 | 1.0000 |
| gemini@hybrid_rerank | wrong_refusal | 94 | 0.415 | 0.202 | -0.213 | [-0.298, -0.138] | 20 | 0 | exact McNemar | 0.0000 | 0.0000 |
| gemini@hybrid_rerank | venue_attribution_error | 59 | 0.000 | 0.028 | +0.028 | [+0.000, +0.073] | 0 | 3 | paired randomisation | 0.2551 | 1.0000 |
