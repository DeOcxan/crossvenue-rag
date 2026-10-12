# Retrieval ablation

- run: `20261010T214504Z`
- index_version: `2825a7160c4d09bf`
- questions file: `evals/datasets/v0.2.2-gold.jsonl`
- questions: 114 (92 with gold labels)
- embedding: `BAAI/bge-base-en-v1.5`
- reranker: `BAAI/bge-reranker-base`
- chunking: max 450 tokens, overlap 60
- RRF k: 60

## Metrics

| method | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.459 | 0.161 | 0.359 | 0.359 | 0.320 | 0.189 | 0.331 | 0.507 | 0.159 | 0.403 | 0.404 | 59 | 0.407 | 0.661 | 0.074 | 0.2 | 0.3 |
| dense | 0.399 | 0.155 | 0.283 | 0.283 | 0.321 | 0.154 | 0.304 | 0.464 | 0.117 | 0.360 | 0.366 | 59 | 0.407 | 0.508 | 0.074 | 46 | 57 |
| hybrid | 0.451 | 0.142 | 0.304 | 0.304 | 0.418 | 0.222 | 0.375 | 0.592 | 0.166 | 0.436 | 0.437 | 59 | 0.441 | 0.576 | 0.035 | 45 | 54 |
| hybrid_rerank | 0.588 | 0.238 | 0.435 | 0.435 | 0.528 | 0.270 | 0.494 | 0.692 | 0.197 | 0.560 | 0.562 | 59 | 0.441 | 0.661 | 0.051 | 7,509 | 13,929 |
| venue_split | 0.543 | 0.203 | 0.380 | 0.380 | 0.516 | 0.263 | 0.464 | 0.665 | 0.189 | 0.525 | 0.526 | 59 | 0.831 | 0.898 | 0.034 | 7,774 | 13,592 |
| venue_split_oracle (oracle) † | 0.520 | 0.193 | 0.348 | 0.348 | 0.502 | 0.248 | 0.444 | 0.655 | 0.184 | 0.508 | 0.517 | 59 | 1.000 | 1.000 | 0.063 | 7,707 | 17,466 |

`venue_coverage@k`: share of questions naming 2 or more venues (`multi-venue`) whose top k hold a chunk from every one of them. It needs no gold labels.

† Not in the labeling pool (`retrieval.retriever.POOL_METHODS`). `unjudged@k` is the share of a method's top k that no judgment covered (pool plus augmentation, from the journal), each such chunk counted as a miss; it does not measure the bias itself. `ndcg_prime@k` is nDCG on the condensed list (Sakai 2007, full entry under References in `evals/methodology.md`): unjudged chunks are dropped from the ranking instead of scored as irrelevant, and its gap to nDCG@k is the change that dropping them makes. Both are `-` when the judgment history is not available. venue_coverage@k needs no gold and is unaffected. (oracle) runs venue_split with the question's labeled `venues` in place of the detected ones, which a deployed system does not have; it bounds what better venue detection can add to venue coverage, and it is not an upper bound on recall, precision or nDCG.

### By venue

| venue | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binance | bm25 | 42 | 34 | 0.391 | 0.067 | 0.265 | 0.265 | 0.246 | 0.188 | 0.265 | 0.501 | 0.185 | 0.359 | 0.361 | 27 | 0.222 | 0.444 | 0.071 | 0.2 |
| binance | dense | 42 | 34 | 0.382 | 0.122 | 0.265 | 0.265 | 0.284 | 0.176 | 0.286 | 0.474 | 0.144 | 0.357 | 0.371 | 27 | 0.407 | 0.518 | 0.076 | 48 |
| binance | hybrid | 42 | 34 | 0.418 | 0.117 | 0.294 | 0.294 | 0.330 | 0.229 | 0.330 | 0.537 | 0.188 | 0.400 | 0.401 | 27 | 0.333 | 0.407 | 0.024 | 44 |
| binance | hybrid_rerank | 42 | 34 | 0.578 | 0.159 | 0.382 | 0.382 | 0.508 | 0.300 | 0.471 | 0.769 | 0.244 | 0.576 | 0.581 | 27 | 0.296 | 0.556 | 0.059 | 7,532 |
| binance | venue_split | 42 | 34 | 0.530 | 0.137 | 0.353 | 0.353 | 0.491 | 0.294 | 0.446 | 0.687 | 0.224 | 0.522 | 0.524 | 27 | 0.778 | 0.815 | 0.029 | 8,698 |
| binance | venue_split_oracle | 42 | 34 | 0.492 | 0.122 | 0.294 | 0.294 | 0.468 | 0.265 | 0.411 | 0.665 | 0.209 | 0.490 | 0.503 | 27 | 1.000 | 1.000 | 0.079 | 9,975 |
| bingx | bm25 | 24 | 22 | 0.477 | 0.119 | 0.364 | 0.364 | 0.253 | 0.255 | 0.337 | 0.495 | 0.268 | 0.432 | 0.432 | 19 | 0.210 | 0.579 | 0.064 | 0.1 |
| bingx | dense | 24 | 22 | 0.391 | 0.106 | 0.273 | 0.273 | 0.222 | 0.182 | 0.257 | 0.373 | 0.164 | 0.316 | 0.318 | 19 | 0.263 | 0.368 | 0.077 | 51 |
| bingx | hybrid | 24 | 22 | 0.505 | 0.097 | 0.318 | 0.318 | 0.419 | 0.345 | 0.425 | 0.524 | 0.255 | 0.450 | 0.451 | 19 | 0.316 | 0.474 | 0.041 | 47 |
| bingx | hybrid_rerank | 24 | 22 | 0.691 | 0.204 | 0.545 | 0.545 | 0.499 | 0.382 | 0.551 | 0.646 | 0.291 | 0.597 | 0.601 | 19 | 0.263 | 0.526 | 0.036 | 6,952 |
| bingx | venue_split | 24 | 22 | 0.571 | 0.106 | 0.364 | 0.364 | 0.456 | 0.345 | 0.458 | 0.653 | 0.291 | 0.539 | 0.543 | 19 | 0.684 | 0.789 | 0.023 | 7,947 |
| bingx | venue_split_oracle | 24 | 22 | 0.522 | 0.099 | 0.318 | 0.318 | 0.382 | 0.282 | 0.392 | 0.596 | 0.264 | 0.485 | 0.509 | 19 | 1.000 | 1.000 | 0.114 | 10,347 |
| bybit | bm25 | 34 | 30 | 0.466 | 0.190 | 0.400 | 0.400 | 0.316 | 0.187 | 0.348 | 0.572 | 0.160 | 0.429 | 0.431 | 26 | 0.231 | 0.500 | 0.060 | 0.2 |
| bybit | dense | 34 | 30 | 0.434 | 0.253 | 0.367 | 0.367 | 0.364 | 0.147 | 0.353 | 0.509 | 0.113 | 0.410 | 0.411 | 26 | 0.308 | 0.385 | 0.073 | 47 |
| bybit | hybrid | 34 | 30 | 0.419 | 0.190 | 0.300 | 0.300 | 0.399 | 0.213 | 0.387 | 0.578 | 0.157 | 0.438 | 0.440 | 26 | 0.269 | 0.385 | 0.030 | 47 |
| bybit | hybrid_rerank | 34 | 30 | 0.690 | 0.289 | 0.500 | 0.500 | 0.660 | 0.333 | 0.617 | 0.819 | 0.223 | 0.669 | 0.674 | 26 | 0.385 | 0.615 | 0.050 | 7,532 |
| bybit | venue_split | 34 | 30 | 0.591 | 0.236 | 0.400 | 0.400 | 0.580 | 0.287 | 0.515 | 0.754 | 0.210 | 0.583 | 0.584 | 26 | 0.769 | 0.846 | 0.020 | 8,926 |
| bybit | venue_split_oracle | 34 | 30 | 0.558 | 0.231 | 0.367 | 0.367 | 0.533 | 0.247 | 0.471 | 0.713 | 0.190 | 0.544 | 0.560 | 26 | 1.000 | 1.000 | 0.087 | 10,873 |
| ccxt | bm25 | 6 | 5 | 0.725 | 0.217 | 0.600 | 0.600 | 0.483 | 0.280 | 0.488 | 0.683 | 0.200 | 0.574 | 0.574 | 5 | 0.600 | 0.800 | 0.080 | 0.2 |
| ccxt | dense | 6 | 5 | 0.600 | 0.183 | 0.600 | 0.600 | 0.183 | 0.120 | 0.266 | 0.350 | 0.120 | 0.347 | 0.347 | 5 | 0.600 | 0.800 | 0.060 | 43 |
| ccxt | hybrid | 6 | 5 | 0.532 | 0.133 | 0.400 | 0.400 | 0.383 | 0.240 | 0.383 | 0.667 | 0.200 | 0.508 | 0.515 | 5 | 0.400 | 0.800 | 0.040 | 47 |
| ccxt | hybrid_rerank | 6 | 5 | 0.800 | 0.183 | 0.600 | 0.600 | 0.583 | 0.360 | 0.565 | 0.767 | 0.240 | 0.654 | 0.659 | 5 | 0.400 | 0.400 | 0.060 | 7,289 |
| ccxt | venue_split | 6 | 5 | 0.750 | 0.183 | 0.600 | 0.600 | 0.650 | 0.400 | 0.600 | 0.833 | 0.260 | 0.689 | 0.694 | 5 | 0.600 | 0.600 | 0.040 | 7,523 |
| ccxt | venue_split_oracle | 6 | 5 | 0.557 | 0.133 | 0.400 | 0.400 | 0.550 | 0.320 | 0.450 | 0.667 | 0.200 | 0.505 | 0.574 | 5 | 1.000 | 1.000 | 0.240 | 9,476 |
| dydx | bm25 | 14 | 11 | 0.273 | 0.121 | 0.182 | 0.182 | 0.197 | 0.091 | 0.190 | 0.348 | 0.082 | 0.251 | 0.252 | 8 | 0.500 | 0.750 | 0.082 | 0.2 |
| dydx | dense | 14 | 11 | 0.405 | 0.121 | 0.273 | 0.273 | 0.295 | 0.127 | 0.271 | 0.432 | 0.109 | 0.334 | 0.336 | 8 | 0.375 | 0.500 | 0.054 | 47 |
| dydx | hybrid | 14 | 11 | 0.379 | 0.061 | 0.182 | 0.182 | 0.492 | 0.200 | 0.339 | 0.515 | 0.109 | 0.352 | 0.352 | 8 | 0.500 | 0.625 | 0.054 | 43 |
| dydx | hybrid_rerank | 14 | 11 | 0.483 | 0.242 | 0.364 | 0.364 | 0.447 | 0.164 | 0.386 | 0.599 | 0.118 | 0.449 | 0.450 | 8 | 0.625 | 0.875 | 0.036 | 6,952 |
| dydx | venue_split | 14 | 11 | 0.437 | 0.151 | 0.273 | 0.273 | 0.477 | 0.182 | 0.364 | 0.599 | 0.118 | 0.409 | 0.409 | 8 | 0.875 | 1.000 | 0.009 | 8,781 |
| dydx | venue_split_oracle | 14 | 11 | 0.483 | 0.182 | 0.364 | 0.364 | 0.447 | 0.164 | 0.363 | 0.599 | 0.118 | 0.423 | 0.429 | 8 | 1.000 | 1.000 | 0.036 | 7,568 |
| hyperliquid | bm25 | 27 | 24 | 0.453 | 0.198 | 0.333 | 0.333 | 0.361 | 0.150 | 0.340 | 0.500 | 0.108 | 0.398 | 0.398 | 19 | 0.579 | 0.737 | 0.104 | 0.2 |
| hyperliquid | dense | 27 | 24 | 0.436 | 0.156 | 0.333 | 0.333 | 0.361 | 0.158 | 0.329 | 0.455 | 0.108 | 0.371 | 0.372 | 19 | 0.316 | 0.421 | 0.079 | 46 |
| hyperliquid | hybrid | 27 | 24 | 0.453 | 0.125 | 0.292 | 0.292 | 0.441 | 0.200 | 0.365 | 0.646 | 0.138 | 0.440 | 0.441 | 19 | 0.421 | 0.526 | 0.033 | 44 |
| hyperliquid | hybrid_rerank | 27 | 24 | 0.572 | 0.260 | 0.417 | 0.417 | 0.521 | 0.217 | 0.473 | 0.705 | 0.158 | 0.555 | 0.556 | 19 | 0.421 | 0.579 | 0.046 | 5,836 |
| hyperliquid | venue_split | 27 | 24 | 0.533 | 0.226 | 0.375 | 0.375 | 0.528 | 0.217 | 0.460 | 0.677 | 0.150 | 0.524 | 0.525 | 19 | 0.737 | 0.789 | 0.029 | 7,860 |
| hyperliquid | venue_split_oracle | 27 | 24 | 0.524 | 0.229 | 0.375 | 0.375 | 0.514 | 0.200 | 0.439 | 0.663 | 0.142 | 0.503 | 0.520 | 19 | 1.000 | 1.000 | 0.083 | 8,660 |
| pancakeswap | bm25 | 18 | 15 | 0.744 | 0.247 | 0.667 | 0.667 | 0.442 | 0.267 | 0.473 | 0.526 | 0.167 | 0.515 | 0.516 | 11 | 0.545 | 0.818 | 0.113 | 0.2 |
| pancakeswap | dense | 18 | 15 | 0.407 | 0.150 | 0.333 | 0.333 | 0.250 | 0.160 | 0.276 | 0.330 | 0.107 | 0.310 | 0.310 | 11 | 0.455 | 0.545 | 0.087 | 44 |
| pancakeswap | hybrid | 18 | 15 | 0.450 | 0.117 | 0.333 | 0.333 | 0.369 | 0.213 | 0.334 | 0.573 | 0.167 | 0.420 | 0.420 | 11 | 0.636 | 0.818 | 0.040 | 43 |
| pancakeswap | hybrid_rerank | 18 | 15 | 0.553 | 0.163 | 0.467 | 0.467 | 0.429 | 0.280 | 0.417 | 0.576 | 0.193 | 0.487 | 0.487 | 11 | 0.545 | 0.818 | 0.033 | 7,378 |
| pancakeswap | venue_split | 18 | 15 | 0.572 | 0.163 | 0.467 | 0.467 | 0.462 | 0.307 | 0.444 | 0.576 | 0.187 | 0.492 | 0.492 | 11 | 0.818 | 1.000 | 0.027 | 7,520 |
| pancakeswap | venue_split_oracle | 18 | 15 | 0.497 | 0.113 | 0.333 | 0.333 | 0.451 | 0.293 | 0.407 | 0.587 | 0.193 | 0.470 | 0.485 | 11 | 1.000 | 1.000 | 0.047 | 7,707 |
| uniswap | bm25 | 18 | 16 | 0.512 | 0.172 | 0.438 | 0.438 | 0.318 | 0.175 | 0.324 | 0.349 | 0.100 | 0.339 | 0.340 | 13 | 0.615 | 0.846 | 0.113 | 0.2 |
| uniswap | dense | 18 | 16 | 0.360 | 0.125 | 0.250 | 0.250 | 0.297 | 0.150 | 0.275 | 0.453 | 0.094 | 0.326 | 0.327 | 13 | 0.538 | 0.615 | 0.113 | 46 |
| uniswap | hybrid | 18 | 16 | 0.406 | 0.099 | 0.250 | 0.250 | 0.406 | 0.175 | 0.310 | 0.604 | 0.144 | 0.391 | 0.391 | 13 | 0.692 | 0.846 | 0.025 | 44 |
| uniswap | hybrid_rerank | 18 | 16 | 0.440 | 0.172 | 0.312 | 0.312 | 0.323 | 0.175 | 0.325 | 0.458 | 0.144 | 0.388 | 0.389 | 13 | 0.615 | 0.846 | 0.056 | 7,358 |
| uniswap | venue_split | 18 | 16 | 0.458 | 0.172 | 0.312 | 0.312 | 0.354 | 0.200 | 0.351 | 0.458 | 0.138 | 0.393 | 0.394 | 13 | 0.846 | 1.000 | 0.050 | 8,037 |
| uniswap | venue_split_oracle | 18 | 16 | 0.388 | 0.125 | 0.188 | 0.188 | 0.344 | 0.188 | 0.316 | 0.469 | 0.144 | 0.372 | 0.387 | 13 | 1.000 | 1.000 | 0.069 | 8,013 |

A question naming two venues counts in both rows, so venue rows do not sum to the total. The corpus is unbalanced (bybit 3,146 chunks vs bingx 507), so the aggregate above is weighted toward whichever venues the question set names most.

### By qtype

| qtype | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison | bm25 | 53 | 51 | 0.445 | 0.138 | 0.353 | 0.353 | 0.286 | 0.188 | 0.316 | 0.463 | 0.159 | 0.375 | 0.377 | 51 | 0.431 | 0.686 | 0.086 | 0.2 |
| comparison | dense | 53 | 51 | 0.393 | 0.162 | 0.314 | 0.314 | 0.272 | 0.145 | 0.286 | 0.367 | 0.104 | 0.321 | 0.322 | 51 | 0.431 | 0.510 | 0.086 | 49 |
| comparison | hybrid | 53 | 51 | 0.419 | 0.102 | 0.275 | 0.275 | 0.360 | 0.212 | 0.333 | 0.548 | 0.169 | 0.393 | 0.394 | 51 | 0.471 | 0.588 | 0.031 | 44 |
| comparison | hybrid_rerank | 53 | 51 | 0.526 | 0.175 | 0.353 | 0.353 | 0.420 | 0.231 | 0.412 | 0.609 | 0.186 | 0.485 | 0.488 | 51 | 0.490 | 0.686 | 0.039 | 6,767 |
| comparison | venue_split | 53 | 51 | 0.444 | 0.113 | 0.255 | 0.255 | 0.400 | 0.220 | 0.358 | 0.561 | 0.172 | 0.421 | 0.421 | 51 | 0.941 | 0.961 | 0.008 | 9,930 |
| comparison | venue_split_oracle | 53 | 51 | 0.434 | 0.108 | 0.235 | 0.235 | 0.403 | 0.216 | 0.354 | 0.575 | 0.176 | 0.424 | 0.424 | 51 | 1.000 | 1.000 | 0.008 | 9,975 |
| fact | bm25 | 56 | 37 | 0.454 | 0.190 | 0.351 | 0.351 | 0.305 | 0.168 | 0.316 | 0.522 | 0.151 | 0.409 | 0.409 | 8 | 0.250 | 0.500 | 0.059 | 0.2 |
| fact | dense | 56 | 37 | 0.407 | 0.153 | 0.243 | 0.243 | 0.408 | 0.173 | 0.343 | 0.589 | 0.135 | 0.417 | 0.431 | 8 | 0.250 | 0.500 | 0.065 | 43 |
| fact | hybrid | 56 | 37 | 0.453 | 0.169 | 0.297 | 0.297 | 0.472 | 0.227 | 0.402 | 0.622 | 0.160 | 0.461 | 0.462 | 8 | 0.250 | 0.500 | 0.041 | 44 |
| fact | hybrid_rerank | 56 | 37 | 0.643 | 0.321 | 0.513 | 0.513 | 0.647 | 0.314 | 0.581 | 0.785 | 0.211 | 0.640 | 0.643 | 8 | 0.125 | 0.500 | 0.062 | 8,297 |
| fact | venue_split | 56 | 37 | 0.643 | 0.321 | 0.513 | 0.513 | 0.647 | 0.314 | 0.581 | 0.785 | 0.211 | 0.640 | 0.643 | 8 | 0.125 | 0.500 | 0.062 | 6,009 |
| fact | venue_split_oracle | 56 | 37 | 0.599 | 0.303 | 0.460 | 0.460 | 0.609 | 0.281 | 0.537 | 0.742 | 0.192 | 0.596 | 0.618 | 8 | 1.000 | 1.000 | 0.135 | 5,309 |
| multihop | bm25 | 5 | 4 | 0.688 | 0.188 | 0.500 | 0.500 | 0.875 | 0.400 | 0.669 | 0.938 | 0.225 | 0.704 | 0.704 | 0 | - | - | 0.050 | 0.2 |
| multihop | dense | 5 | 4 | 0.403 | 0.083 | 0.250 | 0.250 | 0.146 | 0.100 | 0.179 | 0.542 | 0.125 | 0.329 | 0.329 | 0 | - | - | 0.000 | 43 |
| multihop | hybrid | 5 | 4 | 0.833 | 0.396 | 0.750 | 0.750 | 0.667 | 0.300 | 0.664 | 0.875 | 0.200 | 0.748 | 0.750 | 0 | - | - | 0.025 | 50 |
| multihop | hybrid_rerank | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 8,752 |
| multihop | venue_split | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,760 |
| multihop | venue_split_oracle | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,016 |

### By category

| category | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adversarial | bm25 | 13 | 11 | 0.499 | 0.139 | 0.455 | 0.455 | 0.247 | 0.182 | 0.284 | 0.527 | 0.164 | 0.399 | 0.399 | 1 | 1.000 | 1.000 | 0.064 | 0.2 |
| adversarial | dense | 13 | 11 | 0.359 | 0.114 | 0.182 | 0.182 | 0.364 | 0.127 | 0.290 | 0.495 | 0.091 | 0.337 | 0.338 | 1 | 0.000 | 1.000 | 0.064 | 40 |
| adversarial | hybrid | 13 | 11 | 0.437 | 0.144 | 0.273 | 0.273 | 0.488 | 0.218 | 0.378 | 0.706 | 0.164 | 0.465 | 0.465 | 1 | 0.000 | 1.000 | 0.073 | 43 |
| adversarial | hybrid_rerank | 13 | 11 | 0.514 | 0.223 | 0.364 | 0.364 | 0.683 | 0.291 | 0.513 | 0.823 | 0.182 | 0.574 | 0.574 | 1 | 0.000 | 0.000 | 0.073 | 11,750 |
| adversarial | venue_split | 13 | 11 | 0.514 | 0.223 | 0.364 | 0.364 | 0.683 | 0.291 | 0.513 | 0.823 | 0.182 | 0.574 | 0.574 | 1 | 0.000 | 0.000 | 0.073 | 4,728 |
| adversarial | venue_split_oracle | 13 | 11 | 0.498 | 0.223 | 0.364 | 0.364 | 0.683 | 0.291 | 0.502 | 0.792 | 0.173 | 0.550 | 0.551 | 1 | 1.000 | 1.000 | 0.109 | 5,693 |
| ambiguous | bm25 | 6 | 6 | 0.630 | 0.167 | 0.500 | 0.500 | 0.306 | 0.233 | 0.365 | 0.514 | 0.200 | 0.462 | 0.463 | 6 | 0.000 | 0.333 | 0.083 | 0.2 |
| ambiguous | dense | 6 | 6 | 0.575 | 0.153 | 0.500 | 0.500 | 0.333 | 0.267 | 0.371 | 0.569 | 0.217 | 0.463 | 0.469 | 6 | 0.167 | 0.333 | 0.083 | 52 |
| ambiguous | hybrid | 6 | 6 | 0.464 | 0.111 | 0.333 | 0.333 | 0.431 | 0.333 | 0.394 | 0.597 | 0.217 | 0.455 | 0.460 | 6 | 0.167 | 0.333 | 0.017 | 47 |
| ambiguous | hybrid_rerank | 6 | 6 | 0.917 | 0.236 | 0.833 | 0.833 | 0.639 | 0.500 | 0.674 | 0.833 | 0.333 | 0.752 | 0.758 | 6 | 0.000 | 0.500 | 0.083 | 6,952 |
| ambiguous | venue_split | 6 | 6 | 0.917 | 0.236 | 0.833 | 0.833 | 0.639 | 0.500 | 0.674 | 0.833 | 0.333 | 0.752 | 0.758 | 6 | 0.000 | 0.500 | 0.083 | 7,603 |
| ambiguous | venue_split_oracle | 6 | 6 | 0.672 | 0.125 | 0.500 | 0.500 | 0.403 | 0.300 | 0.420 | 0.625 | 0.233 | 0.519 | 0.640 | 6 | 1.000 | 1.000 | 0.467 | 11,644 |
| comparison | bm25 | 50 | 50 | 0.454 | 0.141 | 0.360 | 0.360 | 0.292 | 0.192 | 0.322 | 0.473 | 0.162 | 0.383 | 0.384 | 50 | 0.440 | 0.700 | 0.088 | 0.2 |
| comparison | dense | 50 | 50 | 0.391 | 0.165 | 0.320 | 0.320 | 0.258 | 0.144 | 0.279 | 0.354 | 0.104 | 0.315 | 0.315 | 50 | 0.440 | 0.520 | 0.086 | 49 |
| comparison | hybrid | 50 | 50 | 0.423 | 0.104 | 0.280 | 0.280 | 0.347 | 0.212 | 0.331 | 0.539 | 0.170 | 0.393 | 0.393 | 50 | 0.480 | 0.600 | 0.032 | 44 |
| comparison | hybrid_rerank | 50 | 50 | 0.536 | 0.178 | 0.360 | 0.360 | 0.429 | 0.236 | 0.421 | 0.622 | 0.190 | 0.495 | 0.497 | 50 | 0.500 | 0.700 | 0.032 | 6,767 |
| comparison | venue_split | 50 | 50 | 0.453 | 0.115 | 0.260 | 0.260 | 0.408 | 0.224 | 0.365 | 0.573 | 0.176 | 0.430 | 0.430 | 50 | 0.960 | 0.980 | 0.000 | 9,937 |
| comparison | venue_split_oracle | 50 | 50 | 0.443 | 0.110 | 0.240 | 0.240 | 0.410 | 0.220 | 0.361 | 0.586 | 0.180 | 0.432 | 0.432 | 50 | 1.000 | 1.000 | 0.000 | 9,975 |
| fact | bm25 | 10 | 10 | 0.360 | 0.300 | 0.300 | 0.300 | 0.308 | 0.080 | 0.317 | 0.550 | 0.120 | 0.417 | 0.417 | 0 | - | - | 0.020 | 0.3 |
| fact | dense | 10 | 10 | 0.346 | 0.200 | 0.200 | 0.200 | 0.500 | 0.120 | 0.363 | 0.633 | 0.090 | 0.419 | 0.467 | 0 | - | - | 0.080 | 40 |
| fact | hybrid | 10 | 10 | 0.368 | 0.200 | 0.200 | 0.200 | 0.508 | 0.140 | 0.395 | 0.633 | 0.110 | 0.439 | 0.439 | 0 | - | - | 0.010 | 51 |
| fact | hybrid_rerank | 10 | 10 | 0.637 | 0.500 | 0.500 | 0.500 | 0.700 | 0.220 | 0.642 | 0.783 | 0.160 | 0.676 | 0.679 | 0 | - | - | 0.040 | 6,224 |
| fact | venue_split | 10 | 10 | 0.637 | 0.500 | 0.500 | 0.500 | 0.700 | 0.220 | 0.642 | 0.783 | 0.160 | 0.676 | 0.679 | 0 | - | - | 0.040 | 5,612 |
| fact | venue_split_oracle | 10 | 10 | 0.637 | 0.500 | 0.500 | 0.500 | 0.700 | 0.220 | 0.642 | 0.783 | 0.160 | 0.676 | 0.679 | 0 | - | - | 0.040 | 5,207 |
| multihop | bm25 | 5 | 4 | 0.688 | 0.188 | 0.500 | 0.500 | 0.875 | 0.400 | 0.669 | 0.938 | 0.225 | 0.704 | 0.704 | 0 | - | - | 0.050 | 0.2 |
| multihop | dense | 5 | 4 | 0.403 | 0.083 | 0.250 | 0.250 | 0.146 | 0.100 | 0.179 | 0.542 | 0.125 | 0.329 | 0.329 | 0 | - | - | 0.000 | 43 |
| multihop | hybrid | 5 | 4 | 0.833 | 0.396 | 0.750 | 0.750 | 0.667 | 0.300 | 0.664 | 0.875 | 0.200 | 0.748 | 0.750 | 0 | - | - | 0.025 | 50 |
| multihop | hybrid_rerank | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 8,752 |
| multihop | venue_split | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,760 |
| multihop | venue_split_oracle | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,016 |
| noisy | bm25 | 6 | 6 | 0.347 | 0.167 | 0.167 | 0.167 | 0.333 | 0.167 | 0.293 | 0.333 | 0.083 | 0.293 | 0.293 | 0 | - | - | 0.033 | 0.2 |
| noisy | dense | 6 | 6 | 0.367 | 0.167 | 0.167 | 0.167 | 0.375 | 0.200 | 0.319 | 0.708 | 0.183 | 0.464 | 0.464 | 0 | - | - | 0.033 | 51 |
| noisy | hybrid | 6 | 6 | 0.458 | 0.208 | 0.333 | 0.333 | 0.333 | 0.167 | 0.333 | 0.458 | 0.133 | 0.397 | 0.397 | 0 | - | - | 0.067 | 48 |
| noisy | hybrid_rerank | 6 | 6 | 0.575 | 0.375 | 0.500 | 0.500 | 0.542 | 0.233 | 0.512 | 0.667 | 0.167 | 0.578 | 0.579 | 0 | - | - | 0.050 | 12,250 |
| noisy | venue_split | 6 | 6 | 0.575 | 0.375 | 0.500 | 0.500 | 0.542 | 0.233 | 0.512 | 0.667 | 0.167 | 0.578 | 0.579 | 0 | - | - | 0.050 | 4,154 |
| noisy | venue_split_oracle | 6 | 6 | 0.575 | 0.375 | 0.500 | 0.500 | 0.542 | 0.233 | 0.512 | 0.667 | 0.167 | 0.578 | 0.579 | 0 | - | - | 0.050 | 4,432 |
| unanswerable | bm25 | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 0.2 |
| unanswerable | dense | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 42 |
| unanswerable | hybrid | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 41 |
| unanswerable | hybrid_rerank | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 7,096 |
| unanswerable | venue_split | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 6,357 |
| unanswerable | venue_split_oracle | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 1.000 | 1.000 | - | 5,123 |
| version_sensitive | bm25 | 6 | 5 | 0.369 | 0.100 | 0.200 | 0.200 | 0.333 | 0.200 | 0.293 | 0.583 | 0.180 | 0.405 | 0.405 | 0 | - | - | 0.120 | 0.2 |
| version_sensitive | dense | 6 | 5 | 0.500 | 0.100 | 0.200 | 0.200 | 0.567 | 0.240 | 0.468 | 0.667 | 0.160 | 0.516 | 0.516 | 0 | - | - | 0.060 | 51 |
| version_sensitive | hybrid | 6 | 5 | 0.600 | 0.150 | 0.400 | 0.400 | 0.683 | 0.360 | 0.562 | 0.717 | 0.200 | 0.576 | 0.576 | 0 | - | - | 0.020 | 48 |
| version_sensitive | hybrid_rerank | 6 | 5 | 0.567 | 0.150 | 0.400 | 0.400 | 0.467 | 0.360 | 0.464 | 0.633 | 0.240 | 0.530 | 0.535 | 0 | - | - | 0.140 | 9,833 |
| version_sensitive | venue_split | 6 | 5 | 0.567 | 0.150 | 0.400 | 0.400 | 0.467 | 0.360 | 0.464 | 0.633 | 0.240 | 0.530 | 0.535 | 0 | - | - | 0.140 | 4,982 |
| version_sensitive | venue_split_oracle | 6 | 5 | 0.567 | 0.150 | 0.400 | 0.400 | 0.467 | 0.360 | 0.464 | 0.633 | 0.240 | 0.530 | 0.535 | 0 | - | - | 0.140 | 3,949 |

## Paired comparisons

Same questions, both methods. `a − b` with a 95% paired bootstrap interval (10,000 resamples, seed 20260924); exact two-sided McNemar on the binary venue_coverage; two-sided paired randomization test (sign flip, 10,000 permutations, seed 20260924) on every metric. The primary comparison is venue_split hybrid_rerank venue_coverage@10 comparison; every other row is secondary, with its randomization p-value Holm-adjusted within its metric and scope.

| a vs b | metric | scope | n | a | b | a − b | 95% CI | a better / worse / tie | better is | McNemar p | randomization p | role | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rerank vs hybrid | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.576 | +0.085 | [−0.051, +0.237] | 12 / 7 / 40 | higher | 0.359 | 0.364 | secondary | 0.728 |
| hybrid_rerank vs hybrid | venue_coverage@10 | comparison | 50 | 0.700 | 0.600 | +0.100 | [−0.060, +0.260] | 11 / 6 / 33 | higher | 0.332 | 0.332 | secondary | 0.997 |
| hybrid_rerank vs hybrid | recall@10 | labeled | 92 | 0.692 | 0.592 | +0.100 | [+0.018, +0.182] | 37 / 17 / 38 | higher | - | 0.024 | secondary | 0.097 |
| hybrid_rerank vs hybrid | recall@10 | labeled comparison | 50 | 0.622 | 0.539 | +0.083 | [−0.033, +0.198] | 19 / 12 / 19 | higher | - | 0.184 | secondary | 0.597 |
| venue_split vs hybrid_rerank | venue_coverage@10 | multi-venue | 59 | 0.898 | 0.661 | +0.237 | [+0.136, +0.356] | 14 / 0 / 45 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split vs hybrid_rerank | venue_coverage@10 | comparison | 50 | 0.980 | 0.700 | +0.280 | [+0.160, +0.400] | 14 / 0 / 36 | higher | < 0.001 | < 0.001 | primary | - |
| venue_split vs hybrid_rerank | recall@10 | labeled | 92 | 0.665 | 0.692 | −0.027 | [−0.064, +0.006] | 4 / 10 / 78 | higher | - | 0.151 | secondary | 0.302 |
| venue_split vs hybrid_rerank | recall@10 | labeled comparison | 50 | 0.573 | 0.622 | −0.049 | [−0.117, +0.013] | 4 / 10 / 36 | higher | - | 0.149 | secondary | 0.597 |
| venue_split_oracle (oracle) † vs hybrid_rerank | venue_coverage@10 | multi-venue | 59 | 1.000 | 0.661 | +0.339 | [+0.220, +0.458] | 20 / 0 / 39 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split_oracle (oracle) † vs hybrid_rerank | venue_coverage@10 | comparison | 50 | 1.000 | 0.700 | +0.300 | [+0.180, +0.440] | 15 / 0 / 35 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split_oracle (oracle) † vs hybrid_rerank | recall@10 | labeled | 92 | 0.655 | 0.692 | −0.037 | [−0.080, +0.002] | 6 / 14 / 72 | higher | - | 0.082 | secondary | 0.247 |
| venue_split_oracle (oracle) † vs hybrid_rerank | recall@10 | labeled comparison | 50 | 0.586 | 0.622 | −0.036 | [−0.108, +0.032] | 6 / 10 / 34 | higher | - | 0.323 | secondary | 0.647 |
| venue_split_oracle (oracle) † vs venue_split | venue_coverage@10 | multi-venue | 59 | 1.000 | 0.898 | +0.102 | [+0.034, +0.186] | 6 / 0 / 53 | higher | 0.031 | 0.030 | secondary | 0.119 |
| venue_split_oracle (oracle) † vs venue_split | venue_coverage@10 | comparison | 50 | 1.000 | 0.980 | +0.020 | [+0.000, +0.060] | 1 / 0 / 49 | higher | 1.000 | 1.000 | secondary | 1.000 |
| venue_split_oracle (oracle) † vs venue_split | recall@10 | labeled | 92 | 0.655 | 0.665 | −0.010 | [−0.032, +0.010] | 2 / 4 / 86 | higher | - | 0.405 | secondary | 0.405 |
| venue_split_oracle (oracle) † vs venue_split | recall@10 | labeled comparison | 50 | 0.586 | 0.573 | +0.013 | [+0.000, +0.037] | 2 / 0 / 48 | higher | - | 0.498 | secondary | 0.647 |
| hybrid_rerank vs dense | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.508 | +0.153 | [+0.000, +0.305] | 16 / 7 / 36 | higher | 0.093 | 0.089 | secondary | 0.267 |
| hybrid_rerank vs dense | venue_coverage@10 | comparison | 50 | 0.700 | 0.520 | +0.180 | [+0.020, +0.340] | 14 / 5 / 31 | higher | 0.064 | 0.064 | secondary | 0.255 |
| hybrid_rerank vs dense | recall@10 | labeled | 92 | 0.692 | 0.464 | +0.228 | [+0.133, +0.320] | 42 / 12 / 38 | higher | - | < 0.001 | secondary | < 0.001 |
| hybrid_rerank vs dense | recall@10 | labeled comparison | 50 | 0.622 | 0.354 | +0.267 | [+0.143, +0.388] | 26 / 7 / 17 | higher | - | < 0.001 | secondary | 0.001 |
| hybrid_rerank vs bm25 | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.661 | +0.000 | [−0.136, +0.136] | 9 / 9 / 41 | higher | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs bm25 | venue_coverage@10 | comparison | 50 | 0.700 | 0.700 | +0.000 | [−0.160, +0.160] | 8 / 8 / 34 | higher | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs bm25 | recall@10 | labeled | 92 | 0.692 | 0.507 | +0.184 | [+0.091, +0.278] | 40 / 16 / 36 | higher | - | < 0.001 | secondary | < 0.001 |
| hybrid_rerank vs bm25 | recall@10 | labeled comparison | 50 | 0.622 | 0.473 | +0.149 | [+0.018, +0.278] | 22 / 10 / 18 | higher | - | 0.030 | secondary | 0.151 |

## Off-venue retrieval (exploratory, methodology section 9b)

`off_venue@k`: share of the top k whose venue the question does not name (ccxt apart, as `via_ccxt@k`); `off_venue@1` is the top-1 off-venue rate. Tests as in the paired comparisons, Holm-adjusted within each metric and scope. Lower is better for both metrics: `better` counts the questions where a has the smaller share. A row of a scope under 50 questions is descriptive instead: an interval and no p-value.

| a vs b | metric | scope | n | a | b | a − b | 95% CI | a better / worse / tie | better is | McNemar p | randomization p | role | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rerank vs bm25 | off_venue@10 | comparison | 50 | 0.036 | 0.214 | −0.178 | [−0.226, −0.134] | 36 / 0 / 14 | lower | - | < 0.001 | secondary | < 0.001 |
| hybrid_rerank vs dense | off_venue@10 | comparison | 50 | 0.036 | 0.016 | +0.020 | [−0.006, +0.048] | 4 / 10 / 36 | lower | - | 0.211 | secondary | 0.301 |
| hybrid_rerank vs hybrid | off_venue@10 | comparison | 50 | 0.036 | 0.060 | −0.024 | [−0.052, +0.004] | 15 / 6 / 29 | lower | - | 0.150 | secondary | 0.301 |
| hybrid_rerank vs bm25 | off_venue@10 | adversarial | 13 | 0.269 | 0.392 | −0.123 | [−0.246, +0.000] | 8 / 3 / 2 | lower | - | - | descriptive | - |
| hybrid_rerank vs dense | off_venue@10 | adversarial | 13 | 0.269 | 0.177 | +0.092 | [−0.015, +0.215] | 2 / 6 / 5 | lower | - | - | descriptive | - |
| hybrid_rerank vs hybrid | off_venue@10 | adversarial | 13 | 0.269 | 0.277 | −0.008 | [−0.123, +0.123] | 6 / 3 / 4 | lower | - | - | descriptive | - |
| hybrid_rerank vs bm25 | off_venue@1 | comparison | 50 | 0.020 | 0.180 | −0.160 | [−0.260, −0.060] | 8 / 0 / 42 | lower | 0.008 | 0.009 | secondary | 0.026 |
| hybrid_rerank vs dense | off_venue@1 | comparison | 50 | 0.020 | 0.000 | +0.020 | [+0.000, +0.060] | 0 / 1 / 49 | lower | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs hybrid | off_venue@1 | comparison | 50 | 0.020 | 0.000 | +0.020 | [+0.000, +0.060] | 0 / 1 / 49 | lower | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs bm25 | off_venue@1 | adversarial | 13 | 0.231 | 0.308 | −0.077 | [−0.385, +0.231] | 3 / 2 / 8 | lower | - | - | descriptive | - |
| hybrid_rerank vs dense | off_venue@1 | adversarial | 13 | 0.231 | 0.231 | +0.000 | [+0.000, +0.000] | 0 / 0 / 13 | lower | - | - | descriptive | - |
| hybrid_rerank vs hybrid | off_venue@1 | adversarial | 13 | 0.231 | 0.308 | −0.077 | [−0.231, +0.000] | 1 / 0 / 12 | lower | - | - | descriptive | - |

## Latency (p50, ms per step)

| method | encode | search | fuse | rerank | steps sum p50 |
|---|---|---|---|---|---|
| bm25 | 0.0 | 0.2 | 0.0 | 0.0 | 0.2 |
| dense | 41 | 3.4 | 0.0 | 0.0 | 46 |
| hybrid | 39 | 5.0 | 0.1 | 0.0 | 45 |
| hybrid_rerank | 56 | 8.2 | 0.1 | 7,439 | 7,509 |
| venue_split | 62 | 16 | 0.2 | 7,679 | 7,774 |
| venue_split_oracle | 63 | 21 | 0.2 | 7,582 | 7,707 |
