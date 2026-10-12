# Judge calibration: results/20260925T175128Z_answers

40 answers rated by artem; 95% percentile bootstrap intervals over answers, 10,000 resamples, seed 20260924. Left out of behaviour_correct against the human: qwen@venue_split:cv-0138.

## sonnet vs human

| dimension | n | agreement | kappa | Spearman | threshold |
|---|---:|---:|---:|---:|---:|
| supported | 91 | 0.901 [0.84, 0.96] | 0.165 [0.00, 0.49] | - | kappa >= 0.61 |
| venue_correct | 91 | 0.978 [0.94, 1.00] | 0.492 [0.00, 0.66], undefined in 1,270 resamples | - | kappa >= 0.61 |
| behaviour_correct | 39 | 0.897 [0.79, 0.97] | 0.780 [0.56, 0.95] | - | kappa >= 0.61 |
| coverage | 34 | - | - | 0.843 [0.67, 0.95] | spearman >= 0.7 |

## opus vs human

| dimension | n | agreement | kappa | Spearman | threshold |
|---|---:|---:|---:|---:|---:|
| supported | 91 | 0.890 [0.83, 0.95] | 0.238 [-0.05, 0.56] | - | kappa >= 0.61 |
| venue_correct | 91 | 0.989 [0.96, 1.00] | 0.662 [0.65, 0.66], undefined in 3,634 resamples | - | kappa >= 0.61 |
| behaviour_correct | 39 | 0.872 [0.76, 0.97] | 0.722 [0.49, 0.94] | - | kappa >= 0.61 |
| coverage | 34 | - | - | 0.899 [0.78, 0.96] | spearman >= 0.7 |

## sonnet vs opus

| dimension | n | agreement | kappa | Spearman | threshold |
|---|---:|---:|---:|---:|---:|
| supported | 91 | 0.967 [0.92, 1.00] | 0.389 [0.00, 1.00], undefined in 134 resamples | - | kappa >= 0.61 |
| venue_correct | 91 | 0.989 [0.96, 1.00] | 0.795 [0.00, 1.00], undefined in 1,270 resamples | - | kappa >= 0.61 |
| behaviour_correct | 40 | 0.975 [0.93, 1.00] | 0.942 [0.80, 1.00] | - | kappa >= 0.61 |
| coverage | 34 | - | - | 0.980 [0.93, 1.00] | spearman >= 0.7 |

## opus vs opus replicate 1

| dimension | n | agreement | kappa | Spearman | threshold |
|---|---:|---:|---:|---:|---:|
| supported | 91 | 0.989 [0.96, 1.00] | 0.852 [0.00, 1.00], undefined in 134 resamples | - | kappa >= 0.61 |
| venue_correct | 91 | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00], undefined in 3,634 resamples | - | kappa >= 0.61 |
| behaviour_correct | 40 | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | - | kappa >= 0.61 |
| coverage | 34 | - | - | 0.964 [0.91, 0.99] | spearman >= 0.7 |

## Selection

| judge | passes | fails | USD per answer |
|---|---|---|---:|
| sonnet | behaviour_correct, coverage | supported, venue_correct | 0.0164 |
| opus | venue_correct, behaviour_correct, coverage | supported | 0.0295 |

Selected: opus. opus meets the threshold on 3 dimensions, more than any other candidate.
