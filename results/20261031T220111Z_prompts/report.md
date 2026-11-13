# Prompt v1 against v2: `results/20261031T213610Z_judged` and `results/20261031T220102Z_judged`

- `index_version`: `2825a7160c4d09bf`

v2 − v1 per configuration, paired on questions; Holm across configurations within each metric. Faithfulness is not compared (its judge dimension failed calibration).

| config | metric | n | v1 | v2 | v2 − v1 | 95% CI | better | worse | test | p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| gemini@venue_split | `wrong_refusal` | 96 | 0.365 | 0.177 | −0.188 | [−0.271, −0.115] | 18 | 0 | exact McNemar | < 0.001 | < 0.001 |
| gemini@venue_split | `venue_attribution_error` | 63 | 0.000 | 0.021 | +0.021 | [+0.000, +0.058] | 0 | 2 | paired randomization | 0.500 | 1.000 |
| qwen@venue_split | `wrong_refusal` | 96 | 0.250 | 0.062 | −0.188 | [−0.271, −0.115] | 18 | 0 | exact McNemar | < 0.001 | < 0.001 |
| qwen@venue_split | `venue_attribution_error` | 86 | 0.037 | 0.050 | +0.014 | [+0.000, +0.031] | 0 | 3 | paired randomization | 0.249 | 0.998 |
| mercury@venue_split | `wrong_refusal` | 96 | 0.104 | 0.021 | −0.083 | [−0.146, −0.031] | 8 | 0 | exact McNemar | 0.008 | 0.008 |
| mercury@venue_split | `venue_attribution_error` | 100 | 0.003 | 0.020 | +0.017 | [+0.000, +0.043] | 0 | 2 | paired randomization | 0.506 | 1.000 |
| gemini@hybrid_rerank | `wrong_refusal` | 96 | 0.406 | 0.198 | −0.208 | [−0.292, −0.125] | 20 | 0 | exact McNemar | < 0.001 | < 0.001 |
| gemini@hybrid_rerank | `venue_attribution_error` | 59 | 0.000 | 0.028 | +0.028 | [+0.000, +0.073] | 0 | 3 | paired randomization | 0.255 | 0.998 |
