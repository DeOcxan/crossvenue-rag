# Retrieval ablation

- run: `20261031T202503Z`
- `index_version`: `2825a7160c4d09bf`
- questions file: `evals/datasets/v0.3-gold.jsonl`
- questions: 114 (89 with gold labels)
- embedding: `BAAI/bge-base-en-v1.5`
- reranker: `BAAI/bge-reranker-base`
- chunking: max 450 tokens, overlap 60
- RRF k: 60

## Metrics

| method | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.350 | 0.147 | 0.258 | 0.258 | 0.280 | 0.112 | 0.256 | 0.461 | 0.098 | 0.328 | 0.329 | 59 | 0.407 | 0.661 | 0.074 | 0.1 | 0.1 |
| dense | 0.364 | 0.157 | 0.258 | 0.258 | 0.326 | 0.128 | 0.288 | 0.446 | 0.089 | 0.336 | 0.342 | 59 | 0.407 | 0.508 | 0.075 | 28 | 56 |
| hybrid | 0.351 | 0.122 | 0.202 | 0.202 | 0.392 | 0.153 | 0.302 | 0.577 | 0.111 | 0.371 | 0.372 | 59 | 0.441 | 0.576 | 0.036 | 29 | 35 |
| hybrid_rerank | 0.507 | 0.238 | 0.382 | 0.382 | 0.488 | 0.191 | 0.426 | 0.663 | 0.137 | 0.498 | 0.499 | 59 | 0.441 | 0.661 | 0.047 | 3,624 | 5,847 |
| venue_split | 0.461 | 0.185 | 0.315 | 0.315 | 0.488 | 0.196 | 0.398 | 0.658 | 0.136 | 0.465 | 0.465 | 59 | 0.831 | 0.898 | 0.029 | 8,335 | 21,576 |
| venue_split_oracle (oracle) † | 0.438 | 0.175 | 0.270 | 0.270 | 0.476 | 0.186 | 0.384 | 0.645 | 0.129 | 0.450 | 0.458 | 59 | 1.000 | 1.000 | 0.055 | 8,228 | 19,242 |

`venue_coverage@k`: share of questions naming 2 or more venues (`multi-venue`) whose top k hold a chunk from every one of them. It needs no gold labels.

† Not in the labeling pool (`retrieval.methods.POOL_METHODS`). `unjudged@k` is the share of a method's top k that no judgment covered (pool plus augmentation, from the journal), each such chunk counted as a miss; it does not measure the bias itself. `ndcg_prime@k` is nDCG on the condensed list (Sakai 2007, full entry under References in `evals/methodology.md`): unjudged chunks are dropped from the ranking instead of scored as irrelevant, and its gap to nDCG@k is the change that dropping them makes. Both are `-` when the judgment history is not available. venue_coverage@k needs no gold and is unaffected. (oracle) runs venue_split with the question's labeled `venues` in place of the detected ones, which a deployed system does not have; it bounds what better venue detection can add to venue coverage, and it is not an upper bound on recall, precision or nDCG.

### By venue

| venue | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binance | bm25 | 42 | 33 | 0.258 | 0.072 | 0.182 | 0.182 | 0.161 | 0.091 | 0.161 | 0.363 | 0.097 | 0.240 | 0.240 | 27 | 0.222 | 0.444 | 0.073 | 0.1 |
| binance | dense | 42 | 33 | 0.353 | 0.146 | 0.242 | 0.242 | 0.341 | 0.145 | 0.294 | 0.457 | 0.100 | 0.342 | 0.356 | 27 | 0.407 | 0.518 | 0.079 | 28 |
| binance | hybrid | 42 | 33 | 0.306 | 0.093 | 0.182 | 0.182 | 0.298 | 0.145 | 0.238 | 0.547 | 0.124 | 0.335 | 0.336 | 27 | 0.333 | 0.407 | 0.024 | 29 |
| binance | hybrid_rerank | 42 | 33 | 0.517 | 0.201 | 0.364 | 0.364 | 0.493 | 0.224 | 0.428 | 0.745 | 0.164 | 0.526 | 0.527 | 27 | 0.296 | 0.556 | 0.058 | 3,761 |
| binance | venue_split | 42 | 33 | 0.444 | 0.131 | 0.273 | 0.273 | 0.499 | 0.230 | 0.390 | 0.687 | 0.151 | 0.457 | 0.458 | 27 | 0.778 | 0.815 | 0.027 | 8,926 |
| binance | venue_split_oracle | 42 | 33 | 0.402 | 0.116 | 0.212 | 0.212 | 0.454 | 0.200 | 0.348 | 0.666 | 0.139 | 0.428 | 0.440 | 27 | 1.000 | 1.000 | 0.079 | 10,034 |
| bingx | bm25 | 24 | 22 | 0.399 | 0.132 | 0.318 | 0.318 | 0.298 | 0.136 | 0.269 | 0.483 | 0.127 | 0.350 | 0.350 | 19 | 0.210 | 0.579 | 0.064 | 0.1 |
| bingx | dense | 24 | 22 | 0.343 | 0.129 | 0.273 | 0.273 | 0.258 | 0.127 | 0.246 | 0.398 | 0.104 | 0.307 | 0.308 | 19 | 0.263 | 0.368 | 0.077 | 30 |
| bingx | hybrid | 24 | 22 | 0.380 | 0.095 | 0.227 | 0.227 | 0.466 | 0.209 | 0.339 | 0.510 | 0.123 | 0.362 | 0.363 | 19 | 0.316 | 0.474 | 0.041 | 29 |
| bingx | hybrid_rerank | 24 | 22 | 0.625 | 0.222 | 0.500 | 0.500 | 0.513 | 0.273 | 0.488 | 0.607 | 0.168 | 0.526 | 0.529 | 19 | 0.263 | 0.526 | 0.036 | 3,470 |
| bingx | venue_split | 24 | 22 | 0.514 | 0.116 | 0.364 | 0.364 | 0.475 | 0.255 | 0.403 | 0.607 | 0.168 | 0.457 | 0.459 | 19 | 0.684 | 0.789 | 0.023 | 9,930 |
| bingx | venue_split_oracle | 24 | 22 | 0.442 | 0.095 | 0.273 | 0.273 | 0.408 | 0.209 | 0.339 | 0.541 | 0.141 | 0.398 | 0.417 | 19 | 1.000 | 1.000 | 0.114 | 9,906 |
| bybit | bm25 | 34 | 31 | 0.327 | 0.142 | 0.258 | 0.258 | 0.246 | 0.110 | 0.238 | 0.472 | 0.103 | 0.321 | 0.322 | 26 | 0.231 | 0.500 | 0.058 | 0.1 |
| bybit | dense | 34 | 31 | 0.403 | 0.269 | 0.355 | 0.355 | 0.368 | 0.123 | 0.348 | 0.470 | 0.084 | 0.390 | 0.390 | 26 | 0.308 | 0.385 | 0.077 | 28 |
| bybit | hybrid | 34 | 31 | 0.351 | 0.177 | 0.226 | 0.226 | 0.360 | 0.142 | 0.313 | 0.572 | 0.113 | 0.394 | 0.396 | 26 | 0.269 | 0.385 | 0.029 | 30 |
| bybit | hybrid_rerank | 34 | 31 | 0.573 | 0.292 | 0.452 | 0.452 | 0.541 | 0.219 | 0.498 | 0.766 | 0.152 | 0.581 | 0.583 | 26 | 0.385 | 0.615 | 0.052 | 3,661 |
| bybit | venue_split | 34 | 31 | 0.491 | 0.217 | 0.355 | 0.355 | 0.501 | 0.213 | 0.430 | 0.713 | 0.145 | 0.503 | 0.506 | 26 | 0.769 | 0.846 | 0.023 | 8,083 |
| bybit | venue_split_oracle | 34 | 31 | 0.444 | 0.202 | 0.290 | 0.290 | 0.470 | 0.187 | 0.393 | 0.666 | 0.126 | 0.463 | 0.477 | 26 | 1.000 | 1.000 | 0.087 | 10,352 |
| ccxt | bm25 | 6 | 4 | 0.156 | 0.000 | 0.000 | 0.000 | 0.250 | 0.150 | 0.183 | 0.375 | 0.100 | 0.232 | 0.232 | 5 | 0.600 | 0.800 | 0.050 | 0.1 |
| ccxt | dense | 6 | 4 | 0.750 | 0.292 | 0.750 | 0.750 | 0.292 | 0.150 | 0.388 | 0.542 | 0.150 | 0.502 | 0.502 | 5 | 0.600 | 0.800 | 0.025 | 31 |
| ccxt | hybrid | 6 | 4 | 0.375 | 0.083 | 0.250 | 0.250 | 0.292 | 0.150 | 0.268 | 0.458 | 0.125 | 0.345 | 0.350 | 5 | 0.400 | 0.800 | 0.050 | 31 |
| ccxt | hybrid_rerank | 6 | 4 | 0.750 | 0.292 | 0.750 | 0.750 | 0.458 | 0.250 | 0.479 | 0.625 | 0.175 | 0.560 | 0.567 | 5 | 0.400 | 0.400 | 0.075 | 3,314 |
| ccxt | venue_split | 6 | 4 | 0.750 | 0.292 | 0.750 | 0.750 | 0.458 | 0.250 | 0.479 | 0.625 | 0.175 | 0.560 | 0.567 | 5 | 0.600 | 0.600 | 0.050 | 7,303 |
| ccxt | venue_split_oracle | 6 | 4 | 0.550 | 0.208 | 0.500 | 0.500 | 0.375 | 0.200 | 0.361 | 0.458 | 0.125 | 0.403 | 0.475 | 5 | 1.000 | 1.000 | 0.200 | 9,978 |
| dydx | bm25 | 14 | 11 | 0.308 | 0.159 | 0.273 | 0.273 | 0.189 | 0.073 | 0.201 | 0.348 | 0.073 | 0.264 | 0.265 | 8 | 0.500 | 0.750 | 0.082 | 0.1 |
| dydx | dense | 14 | 11 | 0.427 | 0.144 | 0.364 | 0.364 | 0.318 | 0.145 | 0.297 | 0.371 | 0.091 | 0.322 | 0.323 | 8 | 0.375 | 0.500 | 0.054 | 27 |
| dydx | hybrid | 14 | 11 | 0.356 | 0.068 | 0.182 | 0.182 | 0.447 | 0.182 | 0.316 | 0.500 | 0.109 | 0.339 | 0.339 | 8 | 0.500 | 0.625 | 0.054 | 28 |
| dydx | hybrid_rerank | 14 | 11 | 0.439 | 0.250 | 0.364 | 0.364 | 0.394 | 0.127 | 0.352 | 0.545 | 0.109 | 0.420 | 0.421 | 8 | 0.625 | 0.875 | 0.036 | 3,660 |
| dydx | venue_split | 14 | 11 | 0.397 | 0.159 | 0.273 | 0.273 | 0.424 | 0.145 | 0.332 | 0.591 | 0.118 | 0.401 | 0.402 | 8 | 0.875 | 1.000 | 0.009 | 7,733 |
| dydx | venue_split_oracle | 14 | 11 | 0.397 | 0.159 | 0.273 | 0.273 | 0.424 | 0.145 | 0.332 | 0.568 | 0.109 | 0.392 | 0.394 | 8 | 1.000 | 1.000 | 0.036 | 7,777 |
| hyperliquid | bm25 | 27 | 22 | 0.399 | 0.193 | 0.273 | 0.273 | 0.367 | 0.127 | 0.326 | 0.481 | 0.091 | 0.373 | 0.373 | 19 | 0.579 | 0.737 | 0.104 | 0.1 |
| hyperliquid | dense | 27 | 22 | 0.458 | 0.144 | 0.364 | 0.364 | 0.337 | 0.164 | 0.317 | 0.394 | 0.100 | 0.342 | 0.343 | 19 | 0.316 | 0.421 | 0.077 | 27 |
| hyperliquid | hybrid | 27 | 22 | 0.390 | 0.091 | 0.227 | 0.227 | 0.356 | 0.164 | 0.286 | 0.629 | 0.132 | 0.385 | 0.386 | 19 | 0.421 | 0.526 | 0.036 | 28 |
| hyperliquid | hybrid_rerank | 27 | 22 | 0.520 | 0.246 | 0.409 | 0.409 | 0.458 | 0.182 | 0.419 | 0.648 | 0.145 | 0.505 | 0.507 | 19 | 0.421 | 0.579 | 0.045 | 3,391 |
| hyperliquid | venue_split | 27 | 22 | 0.464 | 0.186 | 0.318 | 0.318 | 0.485 | 0.191 | 0.400 | 0.659 | 0.141 | 0.473 | 0.475 | 19 | 0.737 | 0.789 | 0.027 | 8,335 |
| hyperliquid | venue_split_oracle | 27 | 22 | 0.436 | 0.171 | 0.273 | 0.273 | 0.470 | 0.182 | 0.379 | 0.640 | 0.132 | 0.450 | 0.465 | 19 | 1.000 | 1.000 | 0.068 | 8,753 |
| pancakeswap | bm25 | 18 | 15 | 0.533 | 0.163 | 0.400 | 0.400 | 0.349 | 0.160 | 0.332 | 0.512 | 0.127 | 0.409 | 0.411 | 11 | 0.545 | 0.818 | 0.113 | 0.1 |
| pancakeswap | dense | 18 | 15 | 0.378 | 0.147 | 0.267 | 0.267 | 0.248 | 0.133 | 0.262 | 0.392 | 0.093 | 0.317 | 0.318 | 11 | 0.455 | 0.545 | 0.087 | 28 |
| pancakeswap | hybrid | 18 | 15 | 0.367 | 0.113 | 0.200 | 0.200 | 0.399 | 0.173 | 0.312 | 0.566 | 0.120 | 0.377 | 0.377 | 11 | 0.636 | 0.818 | 0.040 | 28 |
| pancakeswap | hybrid_rerank | 18 | 15 | 0.431 | 0.108 | 0.333 | 0.333 | 0.330 | 0.173 | 0.282 | 0.537 | 0.153 | 0.377 | 0.378 | 11 | 0.545 | 0.818 | 0.033 | 3,491 |
| pancakeswap | venue_split | 18 | 15 | 0.460 | 0.108 | 0.333 | 0.333 | 0.363 | 0.200 | 0.314 | 0.603 | 0.167 | 0.417 | 0.417 | 11 | 0.818 | 1.000 | 0.027 | 8,357 |
| pancakeswap | venue_split_oracle | 18 | 15 | 0.417 | 0.078 | 0.200 | 0.200 | 0.393 | 0.213 | 0.320 | 0.590 | 0.160 | 0.405 | 0.417 | 11 | 1.000 | 1.000 | 0.047 | 8,109 |
| uniswap | bm25 | 18 | 15 | 0.378 | 0.097 | 0.267 | 0.267 | 0.260 | 0.120 | 0.231 | 0.357 | 0.087 | 0.276 | 0.279 | 13 | 0.615 | 0.846 | 0.120 | 0.1 |
| uniswap | dense | 18 | 15 | 0.362 | 0.147 | 0.267 | 0.267 | 0.270 | 0.133 | 0.266 | 0.503 | 0.093 | 0.345 | 0.346 | 13 | 0.538 | 0.615 | 0.113 | 28 |
| uniswap | hybrid | 18 | 15 | 0.398 | 0.147 | 0.267 | 0.267 | 0.410 | 0.147 | 0.318 | 0.643 | 0.113 | 0.407 | 0.407 | 13 | 0.692 | 0.846 | 0.027 | 29 |
| uniswap | hybrid_rerank | 18 | 15 | 0.348 | 0.097 | 0.200 | 0.200 | 0.297 | 0.107 | 0.242 | 0.470 | 0.113 | 0.323 | 0.324 | 13 | 0.615 | 0.846 | 0.033 | 3,596 |
| uniswap | venue_split | 18 | 15 | 0.376 | 0.097 | 0.200 | 0.200 | 0.330 | 0.133 | 0.274 | 0.537 | 0.127 | 0.363 | 0.363 | 13 | 0.846 | 1.000 | 0.027 | 7,208 |
| uniswap | venue_split_oracle | 18 | 15 | 0.333 | 0.067 | 0.067 | 0.067 | 0.360 | 0.147 | 0.280 | 0.523 | 0.120 | 0.350 | 0.363 | 13 | 1.000 | 1.000 | 0.047 | 8,109 |

A question naming two venues counts in both rows, so venue rows do not sum to the total. The corpus is unbalanced (bybit 3,146 chunks vs bingx 507), so the aggregate above is weighted toward whichever venues the question set names most.

### By qtype

| qtype | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison | bm25 | 53 | 50 | 0.274 | 0.095 | 0.180 | 0.180 | 0.218 | 0.088 | 0.191 | 0.373 | 0.076 | 0.252 | 0.253 | 51 | 0.431 | 0.686 | 0.088 | 0.1 |
| comparison | dense | 53 | 50 | 0.356 | 0.192 | 0.300 | 0.300 | 0.262 | 0.100 | 0.265 | 0.340 | 0.064 | 0.295 | 0.295 | 51 | 0.431 | 0.510 | 0.086 | 28 |
| comparison | hybrid | 53 | 50 | 0.320 | 0.107 | 0.180 | 0.180 | 0.335 | 0.132 | 0.261 | 0.528 | 0.102 | 0.334 | 0.335 | 51 | 0.471 | 0.588 | 0.032 | 28 |
| comparison | hybrid_rerank | 53 | 50 | 0.408 | 0.165 | 0.260 | 0.260 | 0.363 | 0.144 | 0.320 | 0.558 | 0.114 | 0.399 | 0.400 | 51 | 0.490 | 0.686 | 0.032 | 3,481 |
| comparison | venue_split | 53 | 50 | 0.326 | 0.072 | 0.140 | 0.140 | 0.363 | 0.152 | 0.270 | 0.550 | 0.112 | 0.340 | 0.340 | 51 | 0.941 | 0.961 | 0.000 | 8,357 |
| comparison | venue_split_oracle | 53 | 50 | 0.324 | 0.067 | 0.120 | 0.120 | 0.358 | 0.148 | 0.267 | 0.560 | 0.114 | 0.344 | 0.344 | 51 | 1.000 | 1.000 | 0.000 | 9,744 |
| fact | bm25 | 56 | 36 | 0.416 | 0.190 | 0.333 | 0.333 | 0.321 | 0.128 | 0.302 | 0.536 | 0.119 | 0.390 | 0.390 | 8 | 0.250 | 0.500 | 0.056 | 0.1 |
| fact | dense | 56 | 36 | 0.374 | 0.112 | 0.194 | 0.194 | 0.432 | 0.172 | 0.332 | 0.599 | 0.122 | 0.398 | 0.412 | 8 | 0.250 | 0.500 | 0.067 | 28 |
| fact | hybrid | 56 | 36 | 0.378 | 0.144 | 0.222 | 0.222 | 0.444 | 0.172 | 0.340 | 0.624 | 0.119 | 0.406 | 0.407 | 8 | 0.250 | 0.500 | 0.042 | 29 |
| fact | hybrid_rerank | 56 | 36 | 0.617 | 0.322 | 0.528 | 0.528 | 0.641 | 0.250 | 0.552 | 0.793 | 0.167 | 0.615 | 0.617 | 8 | 0.125 | 0.500 | 0.064 | 3,761 |
| fact | venue_split | 56 | 36 | 0.617 | 0.322 | 0.528 | 0.528 | 0.641 | 0.250 | 0.552 | 0.793 | 0.167 | 0.615 | 0.617 | 8 | 0.125 | 0.500 | 0.064 | 8,540 |
| fact | venue_split_oracle | 56 | 36 | 0.563 | 0.303 | 0.444 | 0.444 | 0.619 | 0.233 | 0.521 | 0.748 | 0.147 | 0.574 | 0.592 | 8 | 1.000 | 1.000 | 0.128 | 6,021 |
| multihop | bm25 | 5 | 3 | 0.833 | 0.500 | 0.667 | 0.667 | 0.833 | 0.333 | 0.782 | 1.000 | 0.200 | 0.855 | 0.855 | 0 | - | - | 0.067 | 0.1 |
| multihop | dense | 5 | 3 | 0.381 | 0.111 | 0.333 | 0.333 | 0.111 | 0.067 | 0.156 | 0.389 | 0.100 | 0.280 | 0.280 | 0 | - | - | 0.000 | 31 |
| multihop | hybrid | 5 | 3 | 0.556 | 0.111 | 0.333 | 0.333 | 0.722 | 0.267 | 0.524 | 0.833 | 0.167 | 0.571 | 0.573 | 0 | - | - | 0.033 | 31 |
| multihop | hybrid_rerank | 5 | 3 | 0.833 | 0.444 | 0.667 | 0.667 | 0.722 | 0.267 | 0.679 | 0.833 | 0.167 | 0.735 | 0.735 | 0 | - | - | 0.100 | 3,192 |
| multihop | venue_split | 5 | 3 | 0.833 | 0.444 | 0.667 | 0.667 | 0.722 | 0.267 | 0.679 | 0.833 | 0.167 | 0.735 | 0.735 | 0 | - | - | 0.100 | 4,695 |
| multihop | venue_split_oracle | 5 | 3 | 0.833 | 0.444 | 0.667 | 0.667 | 0.722 | 0.267 | 0.679 | 0.833 | 0.167 | 0.735 | 0.735 | 0 | - | - | 0.100 | 4,948 |

### By category

| category | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adversarial | bm25 | 13 | 10 | 0.366 | 0.150 | 0.300 | 0.300 | 0.300 | 0.100 | 0.258 | 0.567 | 0.100 | 0.356 | 0.356 | 1 | 1.000 | 1.000 | 0.050 | 0.1 |
| adversarial | dense | 13 | 10 | 0.335 | 0.100 | 0.100 | 0.100 | 0.483 | 0.120 | 0.338 | 0.583 | 0.070 | 0.366 | 0.368 | 1 | 0.000 | 1.000 | 0.050 | 27 |
| adversarial | hybrid | 13 | 10 | 0.302 | 0.100 | 0.100 | 0.100 | 0.517 | 0.140 | 0.319 | 0.767 | 0.110 | 0.405 | 0.405 | 1 | 0.000 | 1.000 | 0.080 | 28 |
| adversarial | hybrid_rerank | 13 | 10 | 0.515 | 0.283 | 0.400 | 0.400 | 0.700 | 0.220 | 0.506 | 0.850 | 0.140 | 0.568 | 0.568 | 1 | 0.000 | 0.000 | 0.080 | 3,566 |
| adversarial | venue_split | 13 | 10 | 0.515 | 0.283 | 0.400 | 0.400 | 0.700 | 0.220 | 0.506 | 0.850 | 0.140 | 0.568 | 0.568 | 1 | 0.000 | 0.000 | 0.080 | 13,143 |
| adversarial | venue_split_oracle | 13 | 10 | 0.515 | 0.283 | 0.400 | 0.400 | 0.700 | 0.220 | 0.506 | 0.850 | 0.140 | 0.568 | 0.568 | 1 | 1.000 | 1.000 | 0.080 | 6,025 |
| ambiguous | bm25 | 6 | 6 | 0.750 | 0.141 | 0.667 | 0.667 | 0.341 | 0.300 | 0.410 | 0.552 | 0.267 | 0.517 | 0.518 | 6 | 0.000 | 0.333 | 0.083 | 0.1 |
| ambiguous | dense | 6 | 6 | 0.700 | 0.172 | 0.667 | 0.667 | 0.481 | 0.400 | 0.508 | 0.675 | 0.267 | 0.596 | 0.597 | 6 | 0.167 | 0.333 | 0.083 | 31 |
| ambiguous | hybrid | 6 | 6 | 0.602 | 0.117 | 0.500 | 0.500 | 0.442 | 0.367 | 0.435 | 0.604 | 0.250 | 0.518 | 0.521 | 6 | 0.167 | 0.333 | 0.017 | 34 |
| ambiguous | hybrid_rerank | 6 | 6 | 1.000 | 0.238 | 1.000 | 1.000 | 0.567 | 0.467 | 0.641 | 0.871 | 0.367 | 0.777 | 0.785 | 6 | 0.000 | 0.500 | 0.083 | 3,664 |
| ambiguous | venue_split | 6 | 6 | 1.000 | 0.238 | 1.000 | 1.000 | 0.567 | 0.467 | 0.641 | 0.871 | 0.367 | 0.777 | 0.785 | 6 | 0.000 | 0.500 | 0.083 | 14,476 |
| ambiguous | venue_split_oracle | 6 | 6 | 0.672 | 0.125 | 0.500 | 0.500 | 0.438 | 0.367 | 0.454 | 0.596 | 0.250 | 0.533 | 0.635 | 6 | 1.000 | 1.000 | 0.467 | 19,242 |
| comparison | bm25 | 50 | 50 | 0.274 | 0.095 | 0.180 | 0.180 | 0.218 | 0.088 | 0.191 | 0.373 | 0.076 | 0.252 | 0.253 | 50 | 0.440 | 0.700 | 0.088 | 0.1 |
| comparison | dense | 50 | 50 | 0.356 | 0.192 | 0.300 | 0.300 | 0.262 | 0.100 | 0.265 | 0.340 | 0.064 | 0.295 | 0.295 | 50 | 0.440 | 0.520 | 0.086 | 28 |
| comparison | hybrid | 50 | 50 | 0.320 | 0.107 | 0.180 | 0.180 | 0.335 | 0.132 | 0.261 | 0.528 | 0.102 | 0.334 | 0.335 | 50 | 0.480 | 0.600 | 0.032 | 28 |
| comparison | hybrid_rerank | 50 | 50 | 0.408 | 0.165 | 0.260 | 0.260 | 0.363 | 0.144 | 0.320 | 0.558 | 0.114 | 0.399 | 0.400 | 50 | 0.500 | 0.700 | 0.032 | 3,481 |
| comparison | venue_split | 50 | 50 | 0.326 | 0.072 | 0.140 | 0.140 | 0.363 | 0.152 | 0.270 | 0.550 | 0.112 | 0.340 | 0.340 | 50 | 0.960 | 0.980 | 0.000 | 8,357 |
| comparison | venue_split_oracle | 50 | 50 | 0.324 | 0.067 | 0.120 | 0.120 | 0.358 | 0.148 | 0.267 | 0.560 | 0.114 | 0.344 | 0.344 | 50 | 1.000 | 1.000 | 0.000 | 9,744 |
| fact | bm25 | 10 | 9 | 0.400 | 0.278 | 0.333 | 0.333 | 0.389 | 0.089 | 0.346 | 0.667 | 0.089 | 0.449 | 0.449 | 0 | - | - | 0.022 | 0.1 |
| fact | dense | 10 | 9 | 0.273 | 0.111 | 0.111 | 0.111 | 0.389 | 0.111 | 0.265 | 0.556 | 0.089 | 0.333 | 0.385 | 0 | - | - | 0.089 | 42 |
| fact | hybrid | 10 | 9 | 0.354 | 0.167 | 0.222 | 0.222 | 0.500 | 0.133 | 0.358 | 0.611 | 0.078 | 0.390 | 0.390 | 0 | - | - | 0.011 | 29 |
| fact | hybrid_rerank | 10 | 9 | 0.541 | 0.389 | 0.444 | 0.444 | 0.667 | 0.200 | 0.568 | 0.722 | 0.111 | 0.590 | 0.590 | 0 | - | - | 0.033 | 3,014 |
| fact | venue_split | 10 | 9 | 0.541 | 0.389 | 0.444 | 0.444 | 0.667 | 0.200 | 0.568 | 0.722 | 0.111 | 0.590 | 0.590 | 0 | - | - | 0.033 | 3,603 |
| fact | venue_split_oracle | 10 | 9 | 0.541 | 0.389 | 0.444 | 0.444 | 0.667 | 0.200 | 0.568 | 0.722 | 0.111 | 0.590 | 0.590 | 0 | - | - | 0.033 | 6,443 |
| multihop | bm25 | 5 | 3 | 0.833 | 0.500 | 0.667 | 0.667 | 0.833 | 0.333 | 0.782 | 1.000 | 0.200 | 0.855 | 0.855 | 0 | - | - | 0.067 | 0.1 |
| multihop | dense | 5 | 3 | 0.381 | 0.111 | 0.333 | 0.333 | 0.111 | 0.067 | 0.156 | 0.389 | 0.100 | 0.280 | 0.280 | 0 | - | - | 0.000 | 31 |
| multihop | hybrid | 5 | 3 | 0.556 | 0.111 | 0.333 | 0.333 | 0.722 | 0.267 | 0.524 | 0.833 | 0.167 | 0.571 | 0.573 | 0 | - | - | 0.033 | 31 |
| multihop | hybrid_rerank | 5 | 3 | 0.833 | 0.444 | 0.667 | 0.667 | 0.722 | 0.267 | 0.679 | 0.833 | 0.167 | 0.735 | 0.735 | 0 | - | - | 0.100 | 3,192 |
| multihop | venue_split | 5 | 3 | 0.833 | 0.444 | 0.667 | 0.667 | 0.722 | 0.267 | 0.679 | 0.833 | 0.167 | 0.735 | 0.735 | 0 | - | - | 0.100 | 4,695 |
| multihop | venue_split_oracle | 5 | 3 | 0.833 | 0.444 | 0.667 | 0.667 | 0.722 | 0.267 | 0.679 | 0.833 | 0.167 | 0.735 | 0.735 | 0 | - | - | 0.100 | 4,948 |
| noisy | bm25 | 6 | 6 | 0.264 | 0.167 | 0.167 | 0.167 | 0.278 | 0.100 | 0.239 | 0.278 | 0.050 | 0.239 | 0.239 | 0 | - | - | 0.033 | 0.1 |
| noisy | dense | 6 | 6 | 0.325 | 0.167 | 0.167 | 0.167 | 0.444 | 0.167 | 0.320 | 0.722 | 0.133 | 0.430 | 0.430 | 0 | - | - | 0.033 | 28 |
| noisy | hybrid | 6 | 6 | 0.312 | 0.167 | 0.167 | 0.167 | 0.278 | 0.100 | 0.250 | 0.444 | 0.083 | 0.313 | 0.313 | 0 | - | - | 0.067 | 29 |
| noisy | hybrid_rerank | 6 | 6 | 0.542 | 0.417 | 0.500 | 0.500 | 0.556 | 0.167 | 0.520 | 0.667 | 0.117 | 0.574 | 0.574 | 0 | - | - | 0.050 | 3,661 |
| noisy | venue_split | 6 | 6 | 0.542 | 0.417 | 0.500 | 0.500 | 0.556 | 0.167 | 0.520 | 0.667 | 0.117 | 0.574 | 0.574 | 0 | - | - | 0.050 | 7,733 |
| noisy | venue_split_oracle | 6 | 6 | 0.542 | 0.417 | 0.500 | 0.500 | 0.556 | 0.167 | 0.520 | 0.667 | 0.117 | 0.574 | 0.574 | 0 | - | - | 0.050 | 4,685 |
| unanswerable | bm25 | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 0.1 |
| unanswerable | dense | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 28 |
| unanswerable | hybrid | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 30 |
| unanswerable | hybrid_rerank | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 3,884 |
| unanswerable | venue_split | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 6,934 |
| unanswerable | venue_split_oracle | 18 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 1.000 | 1.000 | - | 6,021 |
| version_sensitive | bm25 | 6 | 5 | 0.329 | 0.200 | 0.200 | 0.200 | 0.267 | 0.080 | 0.259 | 0.533 | 0.120 | 0.380 | 0.380 | 0 | - | - | 0.120 | 0.1 |
| version_sensitive | dense | 6 | 5 | 0.300 | 0.000 | 0.000 | 0.000 | 0.333 | 0.120 | 0.245 | 0.467 | 0.100 | 0.302 | 0.302 | 0 | - | - | 0.080 | 27 |
| version_sensitive | hybrid | 6 | 5 | 0.383 | 0.200 | 0.200 | 0.200 | 0.400 | 0.160 | 0.347 | 0.600 | 0.100 | 0.418 | 0.418 | 0 | - | - | 0.020 | 26 |
| version_sensitive | hybrid_rerank | 6 | 5 | 0.592 | 0.267 | 0.400 | 0.400 | 0.667 | 0.240 | 0.544 | 0.867 | 0.140 | 0.607 | 0.610 | 0 | - | - | 0.080 | 3,703 |
| version_sensitive | venue_split | 6 | 5 | 0.592 | 0.267 | 0.400 | 0.400 | 0.667 | 0.240 | 0.544 | 0.867 | 0.140 | 0.607 | 0.610 | 0 | - | - | 0.080 | 6,642 |
| version_sensitive | venue_split_oracle | 6 | 5 | 0.592 | 0.267 | 0.400 | 0.400 | 0.667 | 0.240 | 0.544 | 0.867 | 0.140 | 0.607 | 0.610 | 0 | - | - | 0.080 | 4,896 |

## Paired comparisons

Same questions, both methods. `a − b` with a 95% paired bootstrap interval (10,000 resamples, seed 20260924); exact two-sided McNemar on the binary venue_coverage; two-sided paired randomization test (sign flip, 10,000 permutations, seed 20260924) on every metric. The primary comparison is venue_split hybrid_rerank venue_coverage@10 comparison; every other row is secondary, with its randomization p-value Holm-adjusted within its metric and scope.

| a vs b | metric | scope | n | a | b | a − b | 95% CI | a better / worse / tie | better is | McNemar p | randomization p | role | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rerank vs hybrid | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.576 | +0.085 | [−0.051, +0.237] | 12 / 7 / 40 | higher | 0.359 | 0.364 | secondary | 0.728 |
| hybrid_rerank vs hybrid | venue_coverage@10 | comparison | 50 | 0.700 | 0.600 | +0.100 | [−0.060, +0.260] | 11 / 6 / 33 | higher | 0.332 | 0.332 | secondary | 0.997 |
| hybrid_rerank vs hybrid | recall@10 | labeled | 89 | 0.663 | 0.577 | +0.085 | [−0.013, +0.182] | 32 / 16 / 41 | higher | - | 0.090 | secondary | 0.361 |
| hybrid_rerank vs hybrid | recall@10 | labeled comparison | 50 | 0.558 | 0.528 | +0.030 | [−0.102, +0.162] | 17 / 12 / 21 | higher | - | 0.678 | secondary | 1.000 |
| venue_split vs hybrid_rerank | venue_coverage@10 | multi-venue | 59 | 0.898 | 0.661 | +0.237 | [+0.136, +0.356] | 14 / 0 / 45 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split vs hybrid_rerank | venue_coverage@10 | comparison | 50 | 0.980 | 0.700 | +0.280 | [+0.160, +0.400] | 14 / 0 / 36 | higher | < 0.001 | < 0.001 | primary | - |
| venue_split vs hybrid_rerank | recall@10 | labeled | 89 | 0.658 | 0.663 | −0.005 | [−0.053, +0.040] | 8 / 6 / 75 | higher | - | 0.881 | secondary | 1.000 |
| venue_split vs hybrid_rerank | recall@10 | labeled comparison | 50 | 0.550 | 0.558 | −0.008 | [−0.095, +0.072] | 8 / 6 / 36 | higher | - | 0.893 | secondary | 1.000 |
| venue_split_oracle (oracle) † vs hybrid_rerank | venue_coverage@10 | multi-venue | 59 | 1.000 | 0.661 | +0.339 | [+0.220, +0.458] | 20 / 0 / 39 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split_oracle (oracle) † vs hybrid_rerank | venue_coverage@10 | comparison | 50 | 1.000 | 0.700 | +0.300 | [+0.180, +0.440] | 15 / 0 / 35 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split_oracle (oracle) † vs hybrid_rerank | recall@10 | labeled | 89 | 0.645 | 0.663 | −0.018 | [−0.071, +0.033] | 9 / 11 / 69 | higher | - | 0.509 | secondary | 1.000 |
| venue_split_oracle (oracle) † vs hybrid_rerank | recall@10 | labeled comparison | 50 | 0.560 | 0.558 | +0.002 | [−0.087, +0.085] | 9 / 6 / 35 | higher | - | 1.000 | secondary | 1.000 |
| venue_split_oracle (oracle) † vs venue_split | venue_coverage@10 | multi-venue | 59 | 1.000 | 0.898 | +0.102 | [+0.034, +0.186] | 6 / 0 / 53 | higher | 0.031 | 0.030 | secondary | 0.119 |
| venue_split_oracle (oracle) † vs venue_split | venue_coverage@10 | comparison | 50 | 1.000 | 0.980 | +0.020 | [+0.000, +0.060] | 1 / 0 / 49 | higher | 1.000 | 1.000 | secondary | 1.000 |
| venue_split_oracle (oracle) † vs venue_split | recall@10 | labeled | 89 | 0.645 | 0.658 | −0.013 | [−0.035, +0.007] | 1 / 5 / 83 | higher | - | 0.314 | secondary | 0.942 |
| venue_split_oracle (oracle) † vs venue_split | recall@10 | labeled comparison | 50 | 0.560 | 0.550 | +0.010 | [+0.000, +0.030] | 1 / 0 / 49 | higher | - | 1.000 | secondary | 1.000 |
| hybrid_rerank vs dense | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.508 | +0.153 | [+0.000, +0.305] | 16 / 7 / 36 | higher | 0.093 | 0.089 | secondary | 0.267 |
| hybrid_rerank vs dense | venue_coverage@10 | comparison | 50 | 0.700 | 0.520 | +0.180 | [+0.020, +0.340] | 14 / 5 / 31 | higher | 0.064 | 0.064 | secondary | 0.255 |
| hybrid_rerank vs dense | recall@10 | labeled | 89 | 0.663 | 0.446 | +0.216 | [+0.115, +0.318] | 37 / 8 / 44 | higher | - | < 0.001 | secondary | 0.001 |
| hybrid_rerank vs dense | recall@10 | labeled comparison | 50 | 0.558 | 0.340 | +0.218 | [+0.087, +0.343] | 22 / 5 / 23 | higher | - | 0.002 | secondary | 0.010 |
| hybrid_rerank vs bm25 | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.661 | +0.000 | [−0.136, +0.136] | 9 / 9 / 41 | higher | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs bm25 | venue_coverage@10 | comparison | 50 | 0.700 | 0.700 | +0.000 | [−0.160, +0.160] | 8 / 8 / 34 | higher | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs bm25 | recall@10 | labeled | 89 | 0.663 | 0.460 | +0.202 | [+0.093, +0.311] | 40 / 14 / 35 | higher | - | < 0.001 | secondary | 0.004 |
| hybrid_rerank vs bm25 | recall@10 | labeled comparison | 50 | 0.558 | 0.373 | +0.185 | [+0.038, +0.323] | 21 / 7 / 22 | higher | - | 0.015 | secondary | 0.073 |

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
| bm25 | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 |
| dense | 22 | 6.2 | 0.0 | 0.0 | 28 |
| hybrid | 21 | 7.2 | 0.1 | 0.0 | 29 |
| hybrid_rerank | 29 | 8.8 | 0.1 | 3,587 | 3,624 |
| venue_split | 65 | 24 | 0.3 | 8,243 | 8,335 |
| venue_split_oracle | 76 | 24 | 0.3 | 8,126 | 8,228 |
