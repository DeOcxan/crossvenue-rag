# Retrieval ablation

- run: `20261010T222553Z`
- index_version: `2825a7160c4d09bf`
- questions file: `evals/datasets/v0.2.1-gold.jsonl`
- questions: 114 (90 with gold labels)
- embedding: `BAAI/bge-base-en-v1.5`
- reranker: `BAAI/bge-reranker-base`
- chunking: max 450 tokens, overlap 60
- RRF k: 60

## Metrics

| method | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.469 | 0.164 | 0.367 | 0.367 | 0.330 | 0.193 | 0.341 | 0.508 | 0.158 | 0.407 | 0.408 | 59 | 0.407 | 0.661 | 0.076 | 0.2 | 0.3 |
| dense | 0.402 | 0.147 | 0.278 | 0.278 | 0.327 | 0.160 | 0.307 | 0.472 | 0.121 | 0.364 | 0.370 | 59 | 0.407 | 0.508 | 0.076 | 34 | 42 |
| hybrid | 0.462 | 0.147 | 0.311 | 0.311 | 0.424 | 0.229 | 0.384 | 0.603 | 0.171 | 0.446 | 0.447 | 59 | 0.441 | 0.576 | 0.034 | 33 | 39 |
| hybrid_rerank | 0.574 | 0.231 | 0.422 | 0.422 | 0.508 | 0.260 | 0.479 | 0.691 | 0.197 | 0.552 | 0.555 | 59 | 0.441 | 0.661 | 0.050 | 4,765 | 8,292 |
| venue_split | 0.526 | 0.196 | 0.367 | 0.367 | 0.497 | 0.253 | 0.449 | 0.656 | 0.188 | 0.514 | 0.515 | 59 | 0.831 | 0.898 | 0.032 | 5,631 | 16,504 |
| venue_split_oracle (oracle) † | 0.510 | 0.188 | 0.344 | 0.344 | 0.485 | 0.240 | 0.432 | 0.649 | 0.183 | 0.501 | 0.509 | 59 | 1.000 | 1.000 | 0.062 | 7,785 | 13,894 |

`venue_coverage@k`: share of questions naming 2 or more venues (`multi-venue`) whose top k hold a chunk from every one of them. It needs no gold labels.

† Not in the labeling pool (`retrieval.retriever.POOL_METHODS`). `unjudged@k` is the share of a method's top k that no judgment covered (pool plus augmentation, from the journal), each such chunk counted as a miss; it does not measure the bias itself. `ndcg_prime@k` is nDCG on the condensed list (Sakai 2007, full entry under References in `evals/methodology.md`): unjudged chunks are dropped from the ranking instead of scored as irrelevant, and its gap to nDCG@k is the change that dropping them makes. Both are `-` when the judgment history is not available. venue_coverage@k needs no gold and is unaffected. (oracle) runs venue_split with the question's labeled `venues` in place of the detected ones, which a deployed system does not have; it bounds what better venue detection can add to venue coverage, and it is not an upper bound on recall, precision or nDCG.

### By venue

| venue | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binance | bm25 | 42 | 34 | 0.383 | 0.066 | 0.265 | 0.265 | 0.245 | 0.188 | 0.265 | 0.464 | 0.173 | 0.341 | 0.343 | 27 | 0.222 | 0.444 | 0.071 | 0.1 |
| binance | dense | 42 | 34 | 0.397 | 0.122 | 0.265 | 0.265 | 0.302 | 0.188 | 0.302 | 0.507 | 0.156 | 0.379 | 0.393 | 27 | 0.407 | 0.518 | 0.076 | 35 |
| binance | hybrid | 42 | 34 | 0.433 | 0.116 | 0.294 | 0.294 | 0.352 | 0.247 | 0.352 | 0.560 | 0.197 | 0.419 | 0.421 | 27 | 0.333 | 0.407 | 0.024 | 33 |
| binance | hybrid_rerank | 42 | 34 | 0.544 | 0.151 | 0.353 | 0.353 | 0.477 | 0.282 | 0.446 | 0.757 | 0.238 | 0.556 | 0.561 | 27 | 0.296 | 0.556 | 0.059 | 4,916 |
| binance | venue_split | 42 | 34 | 0.491 | 0.130 | 0.324 | 0.324 | 0.461 | 0.277 | 0.420 | 0.656 | 0.215 | 0.496 | 0.498 | 27 | 0.778 | 0.815 | 0.029 | 6,655 |
| binance | venue_split_oracle | 42 | 34 | 0.472 | 0.122 | 0.294 | 0.294 | 0.445 | 0.253 | 0.396 | 0.642 | 0.203 | 0.475 | 0.487 | 27 | 1.000 | 1.000 | 0.079 | 9,783 |
| bingx | bm25 | 24 | 21 | 0.492 | 0.123 | 0.381 | 0.381 | 0.263 | 0.267 | 0.353 | 0.482 | 0.267 | 0.432 | 0.432 | 19 | 0.210 | 0.579 | 0.067 | 0.1 |
| bingx | dense | 24 | 21 | 0.434 | 0.111 | 0.286 | 0.286 | 0.262 | 0.209 | 0.294 | 0.443 | 0.191 | 0.367 | 0.369 | 19 | 0.263 | 0.368 | 0.081 | 35 |
| bingx | hybrid | 24 | 21 | 0.553 | 0.100 | 0.333 | 0.333 | 0.476 | 0.391 | 0.481 | 0.586 | 0.281 | 0.503 | 0.504 | 19 | 0.316 | 0.474 | 0.043 | 35 |
| bingx | hybrid_rerank | 24 | 21 | 0.676 | 0.202 | 0.524 | 0.524 | 0.496 | 0.381 | 0.550 | 0.658 | 0.300 | 0.602 | 0.606 | 19 | 0.263 | 0.526 | 0.038 | 4,452 |
| bingx | venue_split | 24 | 21 | 0.551 | 0.099 | 0.333 | 0.333 | 0.452 | 0.343 | 0.453 | 0.659 | 0.295 | 0.538 | 0.541 | 19 | 0.684 | 0.789 | 0.024 | 7,732 |
| bingx | venue_split_oracle | 24 | 21 | 0.531 | 0.103 | 0.333 | 0.333 | 0.387 | 0.286 | 0.401 | 0.611 | 0.271 | 0.498 | 0.521 | 19 | 1.000 | 1.000 | 0.119 | 9,620 |
| bybit | bm25 | 34 | 29 | 0.472 | 0.197 | 0.414 | 0.414 | 0.326 | 0.193 | 0.360 | 0.548 | 0.152 | 0.424 | 0.426 | 26 | 0.231 | 0.500 | 0.062 | 0.2 |
| bybit | dense | 34 | 29 | 0.462 | 0.262 | 0.379 | 0.379 | 0.400 | 0.166 | 0.383 | 0.550 | 0.124 | 0.443 | 0.444 | 26 | 0.308 | 0.385 | 0.076 | 35 |
| bybit | hybrid | 34 | 29 | 0.451 | 0.197 | 0.310 | 0.310 | 0.436 | 0.234 | 0.419 | 0.621 | 0.169 | 0.472 | 0.474 | 26 | 0.269 | 0.385 | 0.028 | 34 |
| bybit | hybrid_rerank | 34 | 29 | 0.666 | 0.291 | 0.483 | 0.483 | 0.637 | 0.317 | 0.602 | 0.818 | 0.217 | 0.661 | 0.666 | 26 | 0.385 | 0.615 | 0.045 | 4,649 |
| bybit | venue_split | 34 | 29 | 0.558 | 0.236 | 0.379 | 0.379 | 0.554 | 0.269 | 0.496 | 0.735 | 0.203 | 0.567 | 0.568 | 26 | 0.769 | 0.846 | 0.014 | 9,715 |
| bybit | venue_split_oracle | 34 | 29 | 0.548 | 0.238 | 0.379 | 0.379 | 0.514 | 0.234 | 0.464 | 0.700 | 0.186 | 0.539 | 0.554 | 26 | 1.000 | 1.000 | 0.083 | 9,980 |
| ccxt | bm25 | 6 | 5 | 0.725 | 0.217 | 0.600 | 0.600 | 0.483 | 0.280 | 0.488 | 0.683 | 0.200 | 0.574 | 0.574 | 5 | 0.600 | 0.800 | 0.080 | 0.1 |
| ccxt | dense | 6 | 5 | 0.600 | 0.183 | 0.600 | 0.600 | 0.183 | 0.120 | 0.266 | 0.350 | 0.120 | 0.347 | 0.347 | 5 | 0.600 | 0.800 | 0.060 | 33 |
| ccxt | hybrid | 6 | 5 | 0.532 | 0.133 | 0.400 | 0.400 | 0.383 | 0.240 | 0.383 | 0.667 | 0.200 | 0.508 | 0.515 | 5 | 0.400 | 0.800 | 0.040 | 35 |
| ccxt | hybrid_rerank | 6 | 5 | 0.800 | 0.183 | 0.600 | 0.600 | 0.583 | 0.360 | 0.565 | 0.767 | 0.240 | 0.654 | 0.659 | 5 | 0.400 | 0.400 | 0.060 | 5,192 |
| ccxt | venue_split | 6 | 5 | 0.750 | 0.183 | 0.600 | 0.600 | 0.650 | 0.400 | 0.600 | 0.833 | 0.260 | 0.689 | 0.694 | 5 | 0.600 | 0.600 | 0.040 | 5,305 |
| ccxt | venue_split_oracle | 6 | 5 | 0.557 | 0.133 | 0.400 | 0.400 | 0.550 | 0.320 | 0.450 | 0.667 | 0.200 | 0.505 | 0.574 | 5 | 1.000 | 1.000 | 0.240 | 10,316 |
| dydx | bm25 | 14 | 11 | 0.273 | 0.121 | 0.182 | 0.182 | 0.197 | 0.073 | 0.187 | 0.348 | 0.073 | 0.249 | 0.250 | 8 | 0.500 | 0.750 | 0.082 | 0.2 |
| dydx | dense | 14 | 11 | 0.405 | 0.121 | 0.273 | 0.273 | 0.318 | 0.127 | 0.281 | 0.409 | 0.091 | 0.322 | 0.324 | 8 | 0.375 | 0.500 | 0.054 | 33 |
| dydx | hybrid | 14 | 11 | 0.379 | 0.076 | 0.182 | 0.182 | 0.485 | 0.182 | 0.337 | 0.485 | 0.091 | 0.337 | 0.337 | 8 | 0.500 | 0.625 | 0.054 | 32 |
| dydx | hybrid_rerank | 14 | 11 | 0.465 | 0.258 | 0.364 | 0.364 | 0.409 | 0.127 | 0.359 | 0.561 | 0.100 | 0.421 | 0.422 | 8 | 0.625 | 0.875 | 0.036 | 4,189 |
| dydx | venue_split | 14 | 11 | 0.419 | 0.167 | 0.273 | 0.273 | 0.439 | 0.145 | 0.342 | 0.561 | 0.100 | 0.387 | 0.388 | 8 | 0.875 | 1.000 | 0.009 | 6,644 |
| dydx | venue_split_oracle | 14 | 11 | 0.465 | 0.197 | 0.364 | 0.364 | 0.409 | 0.127 | 0.341 | 0.561 | 0.100 | 0.401 | 0.407 | 8 | 1.000 | 1.000 | 0.036 | 7,362 |
| hyperliquid | bm25 | 27 | 24 | 0.464 | 0.198 | 0.333 | 0.333 | 0.382 | 0.158 | 0.352 | 0.521 | 0.113 | 0.410 | 0.410 | 19 | 0.579 | 0.737 | 0.104 | 0.2 |
| hyperliquid | dense | 27 | 24 | 0.399 | 0.115 | 0.292 | 0.292 | 0.326 | 0.150 | 0.291 | 0.420 | 0.104 | 0.332 | 0.334 | 19 | 0.316 | 0.421 | 0.079 | 34 |
| hyperliquid | hybrid | 27 | 24 | 0.437 | 0.132 | 0.292 | 0.292 | 0.392 | 0.183 | 0.337 | 0.611 | 0.133 | 0.417 | 0.418 | 19 | 0.421 | 0.526 | 0.033 | 33 |
| hyperliquid | hybrid_rerank | 27 | 24 | 0.536 | 0.226 | 0.375 | 0.375 | 0.472 | 0.200 | 0.425 | 0.684 | 0.158 | 0.519 | 0.521 | 19 | 0.421 | 0.579 | 0.046 | 4,240 |
| hyperliquid | venue_split | 27 | 24 | 0.497 | 0.191 | 0.333 | 0.333 | 0.479 | 0.200 | 0.414 | 0.656 | 0.150 | 0.490 | 0.492 | 19 | 0.737 | 0.789 | 0.029 | 6,753 |
| hyperliquid | venue_split_oracle | 27 | 24 | 0.488 | 0.194 | 0.333 | 0.333 | 0.465 | 0.183 | 0.394 | 0.642 | 0.142 | 0.470 | 0.487 | 19 | 1.000 | 1.000 | 0.083 | 7,722 |
| pancakeswap | bm25 | 18 | 15 | 0.744 | 0.247 | 0.667 | 0.667 | 0.442 | 0.267 | 0.473 | 0.526 | 0.167 | 0.515 | 0.516 | 11 | 0.545 | 0.818 | 0.113 | 0.2 |
| pancakeswap | dense | 18 | 15 | 0.407 | 0.150 | 0.333 | 0.333 | 0.250 | 0.160 | 0.276 | 0.330 | 0.107 | 0.310 | 0.310 | 11 | 0.455 | 0.545 | 0.087 | 34 |
| pancakeswap | hybrid | 18 | 15 | 0.450 | 0.117 | 0.333 | 0.333 | 0.369 | 0.213 | 0.334 | 0.573 | 0.167 | 0.420 | 0.420 | 11 | 0.636 | 0.818 | 0.040 | 32 |
| pancakeswap | hybrid_rerank | 18 | 15 | 0.553 | 0.163 | 0.467 | 0.467 | 0.429 | 0.280 | 0.417 | 0.576 | 0.193 | 0.487 | 0.487 | 11 | 0.545 | 0.818 | 0.033 | 4,330 |
| pancakeswap | venue_split | 18 | 15 | 0.572 | 0.163 | 0.467 | 0.467 | 0.462 | 0.307 | 0.444 | 0.576 | 0.187 | 0.492 | 0.492 | 11 | 0.818 | 1.000 | 0.027 | 7,736 |
| pancakeswap | venue_split_oracle | 18 | 15 | 0.497 | 0.113 | 0.333 | 0.333 | 0.451 | 0.293 | 0.407 | 0.587 | 0.193 | 0.470 | 0.485 | 11 | 1.000 | 1.000 | 0.047 | 7,556 |
| uniswap | bm25 | 18 | 16 | 0.512 | 0.172 | 0.438 | 0.438 | 0.318 | 0.175 | 0.324 | 0.349 | 0.100 | 0.339 | 0.340 | 13 | 0.615 | 0.846 | 0.113 | 0.2 |
| uniswap | dense | 18 | 16 | 0.360 | 0.125 | 0.250 | 0.250 | 0.297 | 0.150 | 0.275 | 0.453 | 0.094 | 0.326 | 0.327 | 13 | 0.538 | 0.615 | 0.113 | 34 |
| uniswap | hybrid | 18 | 16 | 0.406 | 0.099 | 0.250 | 0.250 | 0.406 | 0.175 | 0.310 | 0.604 | 0.144 | 0.391 | 0.391 | 13 | 0.692 | 0.846 | 0.025 | 31 |
| uniswap | hybrid_rerank | 18 | 16 | 0.440 | 0.172 | 0.312 | 0.312 | 0.323 | 0.175 | 0.325 | 0.458 | 0.144 | 0.388 | 0.389 | 13 | 0.615 | 0.846 | 0.056 | 3,988 |
| uniswap | venue_split | 18 | 16 | 0.458 | 0.172 | 0.312 | 0.312 | 0.354 | 0.200 | 0.351 | 0.458 | 0.138 | 0.393 | 0.394 | 13 | 0.846 | 1.000 | 0.050 | 7,736 |
| uniswap | venue_split_oracle | 18 | 16 | 0.388 | 0.125 | 0.188 | 0.188 | 0.344 | 0.188 | 0.316 | 0.469 | 0.144 | 0.372 | 0.387 | 13 | 1.000 | 1.000 | 0.069 | 8,182 |

A question naming two venues counts in both rows, so venue rows do not sum to the total. The corpus is unbalanced (bybit 3,146 chunks vs bingx 507), so the aggregate above is weighted toward whichever venues the question set names most.

### By qtype

| qtype | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison | bm25 | 53 | 51 | 0.443 | 0.137 | 0.353 | 0.353 | 0.289 | 0.188 | 0.317 | 0.456 | 0.157 | 0.373 | 0.374 | 51 | 0.431 | 0.686 | 0.086 | 0.2 |
| comparison | dense | 53 | 51 | 0.393 | 0.162 | 0.314 | 0.314 | 0.275 | 0.145 | 0.287 | 0.366 | 0.104 | 0.321 | 0.322 | 51 | 0.431 | 0.510 | 0.086 | 35 |
| comparison | hybrid | 53 | 51 | 0.419 | 0.105 | 0.275 | 0.275 | 0.359 | 0.212 | 0.336 | 0.547 | 0.169 | 0.395 | 0.396 | 51 | 0.471 | 0.588 | 0.031 | 33 |
| comparison | hybrid_rerank | 53 | 51 | 0.522 | 0.178 | 0.353 | 0.353 | 0.406 | 0.224 | 0.403 | 0.608 | 0.184 | 0.480 | 0.483 | 51 | 0.490 | 0.686 | 0.039 | 4,240 |
| comparison | venue_split | 53 | 51 | 0.438 | 0.116 | 0.255 | 0.255 | 0.386 | 0.212 | 0.350 | 0.548 | 0.169 | 0.413 | 0.413 | 51 | 0.941 | 0.961 | 0.008 | 9,895 |
| comparison | venue_split_oracle | 53 | 51 | 0.428 | 0.111 | 0.235 | 0.235 | 0.389 | 0.208 | 0.346 | 0.561 | 0.172 | 0.416 | 0.416 | 51 | 1.000 | 1.000 | 0.008 | 9,620 |
| fact | bm25 | 56 | 35 | 0.482 | 0.201 | 0.371 | 0.371 | 0.328 | 0.177 | 0.337 | 0.535 | 0.151 | 0.423 | 0.423 | 8 | 0.250 | 0.500 | 0.063 | 0.2 |
| fact | dense | 56 | 35 | 0.416 | 0.133 | 0.229 | 0.229 | 0.424 | 0.189 | 0.349 | 0.620 | 0.146 | 0.429 | 0.444 | 8 | 0.250 | 0.500 | 0.069 | 34 |
| fact | hybrid | 56 | 35 | 0.482 | 0.179 | 0.314 | 0.314 | 0.491 | 0.246 | 0.422 | 0.653 | 0.171 | 0.486 | 0.486 | 8 | 0.250 | 0.500 | 0.040 | 32 |
| fact | hybrid_rerank | 56 | 35 | 0.615 | 0.303 | 0.486 | 0.486 | 0.624 | 0.303 | 0.560 | 0.789 | 0.214 | 0.631 | 0.635 | 8 | 0.125 | 0.500 | 0.060 | 5,071 |
| fact | venue_split | 56 | 35 | 0.615 | 0.303 | 0.486 | 0.486 | 0.624 | 0.303 | 0.560 | 0.789 | 0.214 | 0.631 | 0.635 | 8 | 0.125 | 0.500 | 0.060 | 4,660 |
| fact | venue_split_oracle | 56 | 35 | 0.587 | 0.291 | 0.457 | 0.457 | 0.591 | 0.274 | 0.524 | 0.751 | 0.197 | 0.594 | 0.616 | 8 | 1.000 | 1.000 | 0.137 | 5,850 |
| multihop | bm25 | 5 | 4 | 0.688 | 0.188 | 0.500 | 0.500 | 0.875 | 0.400 | 0.669 | 0.938 | 0.225 | 0.704 | 0.704 | 0 | - | - | 0.050 | 0.2 |
| multihop | dense | 5 | 4 | 0.403 | 0.083 | 0.250 | 0.250 | 0.146 | 0.100 | 0.179 | 0.542 | 0.125 | 0.329 | 0.329 | 0 | - | - | 0.000 | 34 |
| multihop | hybrid | 5 | 4 | 0.833 | 0.396 | 0.750 | 0.750 | 0.667 | 0.300 | 0.664 | 0.875 | 0.200 | 0.748 | 0.750 | 0 | - | - | 0.025 | 37 |
| multihop | hybrid_rerank | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,414 |
| multihop | venue_split | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 4,790 |
| multihop | venue_split_oracle | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,970 |

### By category

| category | method | questions | labeled | mrr | recall@1 | precision@1 | ndcg@1 | recall@5 | precision@5 | ndcg@5 | recall@10 | precision@10 | ndcg@10 | ndcg_prime@10 | multi-venue | venue_coverage@5 | venue_coverage@10 | unjudged@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adversarial | bm25 | 13 | 11 | 0.499 | 0.139 | 0.455 | 0.455 | 0.247 | 0.182 | 0.284 | 0.527 | 0.164 | 0.399 | 0.399 | 1 | 1.000 | 1.000 | 0.064 | 0.2 |
| adversarial | dense | 13 | 11 | 0.359 | 0.114 | 0.182 | 0.182 | 0.364 | 0.127 | 0.290 | 0.495 | 0.091 | 0.337 | 0.338 | 1 | 0.000 | 1.000 | 0.064 | 33 |
| adversarial | hybrid | 13 | 11 | 0.437 | 0.144 | 0.273 | 0.273 | 0.488 | 0.218 | 0.378 | 0.706 | 0.164 | 0.465 | 0.465 | 1 | 0.000 | 1.000 | 0.073 | 32 |
| adversarial | hybrid_rerank | 13 | 11 | 0.514 | 0.223 | 0.364 | 0.364 | 0.683 | 0.291 | 0.513 | 0.823 | 0.182 | 0.574 | 0.574 | 1 | 0.000 | 0.000 | 0.073 | 8,071 |
| adversarial | venue_split | 13 | 11 | 0.514 | 0.223 | 0.364 | 0.364 | 0.683 | 0.291 | 0.513 | 0.823 | 0.182 | 0.574 | 0.574 | 1 | 0.000 | 0.000 | 0.073 | 4,185 |
| adversarial | venue_split_oracle | 13 | 11 | 0.498 | 0.223 | 0.364 | 0.364 | 0.683 | 0.291 | 0.502 | 0.792 | 0.173 | 0.550 | 0.551 | 1 | 1.000 | 1.000 | 0.109 | 5,904 |
| ambiguous | bm25 | 6 | 6 | 0.602 | 0.167 | 0.500 | 0.500 | 0.306 | 0.233 | 0.365 | 0.389 | 0.150 | 0.397 | 0.398 | 6 | 0.000 | 0.333 | 0.083 | 0.2 |
| ambiguous | dense | 6 | 6 | 0.658 | 0.153 | 0.500 | 0.500 | 0.444 | 0.333 | 0.459 | 0.736 | 0.267 | 0.576 | 0.583 | 6 | 0.167 | 0.333 | 0.083 | 39 |
| ambiguous | hybrid | 6 | 6 | 0.547 | 0.111 | 0.333 | 0.333 | 0.542 | 0.400 | 0.482 | 0.708 | 0.250 | 0.544 | 0.549 | 6 | 0.167 | 0.333 | 0.017 | 33 |
| ambiguous | hybrid_rerank | 6 | 6 | 0.750 | 0.194 | 0.667 | 0.667 | 0.556 | 0.433 | 0.581 | 0.750 | 0.300 | 0.658 | 0.665 | 6 | 0.000 | 0.500 | 0.083 | 4,292 |
| ambiguous | venue_split | 6 | 6 | 0.750 | 0.194 | 0.667 | 0.667 | 0.556 | 0.433 | 0.581 | 0.750 | 0.300 | 0.658 | 0.665 | 6 | 0.000 | 0.500 | 0.083 | 4,453 |
| ambiguous | venue_split_oracle | 6 | 6 | 0.617 | 0.125 | 0.500 | 0.500 | 0.361 | 0.267 | 0.387 | 0.583 | 0.217 | 0.486 | 0.599 | 6 | 1.000 | 1.000 | 0.467 | 13,373 |
| comparison | bm25 | 50 | 50 | 0.452 | 0.140 | 0.360 | 0.360 | 0.295 | 0.192 | 0.324 | 0.465 | 0.160 | 0.380 | 0.382 | 50 | 0.440 | 0.700 | 0.088 | 0.2 |
| comparison | dense | 50 | 50 | 0.391 | 0.165 | 0.320 | 0.320 | 0.261 | 0.144 | 0.281 | 0.353 | 0.104 | 0.315 | 0.316 | 50 | 0.440 | 0.520 | 0.086 | 35 |
| comparison | hybrid | 50 | 50 | 0.423 | 0.107 | 0.280 | 0.280 | 0.346 | 0.212 | 0.334 | 0.538 | 0.170 | 0.394 | 0.395 | 50 | 0.480 | 0.600 | 0.032 | 33 |
| comparison | hybrid_rerank | 50 | 50 | 0.533 | 0.181 | 0.360 | 0.360 | 0.414 | 0.228 | 0.411 | 0.620 | 0.188 | 0.490 | 0.492 | 50 | 0.500 | 0.700 | 0.032 | 4,221 |
| comparison | venue_split | 50 | 50 | 0.446 | 0.118 | 0.260 | 0.260 | 0.394 | 0.216 | 0.357 | 0.559 | 0.172 | 0.421 | 0.421 | 50 | 0.960 | 0.980 | 0.000 | 9,941 |
| comparison | venue_split_oracle | 50 | 50 | 0.436 | 0.113 | 0.240 | 0.240 | 0.397 | 0.212 | 0.353 | 0.572 | 0.176 | 0.424 | 0.424 | 50 | 1.000 | 1.000 | 0.000 | 9,620 |
| fact | bm25 | 8 | 8 | 0.481 | 0.375 | 0.375 | 0.375 | 0.427 | 0.125 | 0.421 | 0.729 | 0.163 | 0.547 | 0.547 | 0 | - | - | 0.025 | 0.2 |
| fact | dense | 8 | 8 | 0.306 | 0.125 | 0.125 | 0.125 | 0.500 | 0.125 | 0.329 | 0.667 | 0.100 | 0.398 | 0.458 | 0 | - | - | 0.100 | 34 |
| fact | hybrid | 8 | 8 | 0.412 | 0.250 | 0.250 | 0.250 | 0.510 | 0.150 | 0.415 | 0.708 | 0.138 | 0.487 | 0.487 | 0 | - | - | 0.000 | 38 |
| fact | hybrid_rerank | 8 | 8 | 0.661 | 0.500 | 0.500 | 0.500 | 0.708 | 0.225 | 0.655 | 0.896 | 0.200 | 0.734 | 0.740 | 0 | - | - | 0.025 | 3,899 |
| fact | venue_split | 8 | 8 | 0.661 | 0.500 | 0.500 | 0.500 | 0.708 | 0.225 | 0.655 | 0.896 | 0.200 | 0.734 | 0.740 | 0 | - | - | 0.025 | 5,217 |
| fact | venue_split_oracle | 8 | 8 | 0.661 | 0.500 | 0.500 | 0.500 | 0.708 | 0.225 | 0.655 | 0.896 | 0.200 | 0.734 | 0.740 | 0 | - | - | 0.025 | 4,830 |
| multihop | bm25 | 5 | 4 | 0.688 | 0.188 | 0.500 | 0.500 | 0.875 | 0.400 | 0.669 | 0.938 | 0.225 | 0.704 | 0.704 | 0 | - | - | 0.050 | 0.2 |
| multihop | dense | 5 | 4 | 0.403 | 0.083 | 0.250 | 0.250 | 0.146 | 0.100 | 0.179 | 0.542 | 0.125 | 0.329 | 0.329 | 0 | - | - | 0.000 | 34 |
| multihop | hybrid | 5 | 4 | 0.833 | 0.396 | 0.750 | 0.750 | 0.667 | 0.300 | 0.664 | 0.875 | 0.200 | 0.748 | 0.750 | 0 | - | - | 0.025 | 37 |
| multihop | hybrid_rerank | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,414 |
| multihop | venue_split | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 4,790 |
| multihop | venue_split_oracle | 5 | 4 | 0.875 | 0.271 | 0.750 | 0.750 | 0.792 | 0.350 | 0.730 | 0.875 | 0.200 | 0.771 | 0.771 | 0 | - | - | 0.100 | 5,970 |
| noisy | bm25 | 6 | 6 | 0.347 | 0.167 | 0.167 | 0.167 | 0.306 | 0.133 | 0.274 | 0.306 | 0.067 | 0.274 | 0.274 | 0 | - | - | 0.033 | 0.2 |
| noisy | dense | 6 | 6 | 0.367 | 0.167 | 0.167 | 0.167 | 0.389 | 0.200 | 0.324 | 0.694 | 0.167 | 0.453 | 0.453 | 0 | - | - | 0.033 | 35 |
| noisy | hybrid | 6 | 6 | 0.458 | 0.208 | 0.333 | 0.333 | 0.347 | 0.167 | 0.339 | 0.431 | 0.117 | 0.379 | 0.379 | 0 | - | - | 0.067 | 30 |
| noisy | hybrid_rerank | 6 | 6 | 0.542 | 0.375 | 0.500 | 0.500 | 0.500 | 0.200 | 0.487 | 0.625 | 0.150 | 0.552 | 0.553 | 0 | - | - | 0.050 | 5,552 |
| noisy | venue_split | 6 | 6 | 0.542 | 0.375 | 0.500 | 0.500 | 0.500 | 0.200 | 0.487 | 0.625 | 0.150 | 0.552 | 0.553 | 0 | - | - | 0.050 | 6,235 |
| noisy | venue_split_oracle | 6 | 6 | 0.542 | 0.375 | 0.500 | 0.500 | 0.500 | 0.200 | 0.487 | 0.625 | 0.150 | 0.552 | 0.553 | 0 | - | - | 0.050 | 4,633 |
| unanswerable | bm25 | 20 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 0.1 |
| unanswerable | dense | 20 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 32 |
| unanswerable | hybrid | 20 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 32 |
| unanswerable | hybrid_rerank | 20 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 4,851 |
| unanswerable | venue_split | 20 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 0.500 | 0.500 | - | 4,726 |
| unanswerable | venue_split_oracle | 20 | 0 | - | - | - | - | - | - | - | - | - | - | - | 2 | 1.000 | 1.000 | - | 5,508 |
| version_sensitive | bm25 | 6 | 5 | 0.369 | 0.100 | 0.200 | 0.200 | 0.333 | 0.200 | 0.293 | 0.583 | 0.180 | 0.405 | 0.405 | 0 | - | - | 0.120 | 0.2 |
| version_sensitive | dense | 6 | 5 | 0.500 | 0.100 | 0.200 | 0.200 | 0.567 | 0.240 | 0.468 | 0.667 | 0.160 | 0.516 | 0.516 | 0 | - | - | 0.060 | 32 |
| version_sensitive | hybrid | 6 | 5 | 0.600 | 0.150 | 0.400 | 0.400 | 0.683 | 0.360 | 0.562 | 0.717 | 0.200 | 0.576 | 0.576 | 0 | - | - | 0.020 | 29 |
| version_sensitive | hybrid_rerank | 6 | 5 | 0.567 | 0.150 | 0.400 | 0.400 | 0.467 | 0.360 | 0.464 | 0.633 | 0.240 | 0.530 | 0.535 | 0 | - | - | 0.140 | 4,707 |
| version_sensitive | venue_split | 6 | 5 | 0.567 | 0.150 | 0.400 | 0.400 | 0.467 | 0.360 | 0.464 | 0.633 | 0.240 | 0.530 | 0.535 | 0 | - | - | 0.140 | 4,660 |
| version_sensitive | venue_split_oracle | 6 | 5 | 0.567 | 0.150 | 0.400 | 0.400 | 0.467 | 0.360 | 0.464 | 0.633 | 0.240 | 0.530 | 0.535 | 0 | - | - | 0.140 | 5,896 |

## Paired comparisons

Same questions, both methods. `a − b` with a 95% paired bootstrap interval (10,000 resamples, seed 20260924); exact two-sided McNemar on the binary venue_coverage; two-sided paired randomization test (sign flip, 10,000 permutations, seed 20260924) on every metric. The primary comparison is venue_split hybrid_rerank venue_coverage@10 comparison; every other row is secondary, with its randomization p-value Holm-adjusted within its metric and scope.

| a vs b | metric | scope | n | a | b | a − b | 95% CI | a better / worse / tie | better is | McNemar p | randomization p | role | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rerank vs hybrid | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.576 | +0.085 | [−0.051, +0.237] | 12 / 7 / 40 | higher | 0.359 | 0.364 | secondary | 0.728 |
| hybrid_rerank vs hybrid | venue_coverage@10 | comparison | 50 | 0.700 | 0.600 | +0.100 | [−0.060, +0.260] | 11 / 6 / 33 | higher | 0.332 | 0.332 | secondary | 0.997 |
| hybrid_rerank vs hybrid | recall@10 | labeled | 90 | 0.691 | 0.603 | +0.088 | [+0.002, +0.174] | 36 / 18 / 36 | higher | - | 0.051 | secondary | 0.202 |
| hybrid_rerank vs hybrid | recall@10 | labeled comparison | 50 | 0.620 | 0.538 | +0.083 | [−0.033, +0.198] | 19 / 12 / 19 | higher | - | 0.184 | secondary | 0.551 |
| venue_split vs hybrid_rerank | venue_coverage@10 | multi-venue | 59 | 0.898 | 0.661 | +0.237 | [+0.136, +0.356] | 14 / 0 / 45 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split vs hybrid_rerank | venue_coverage@10 | comparison | 50 | 0.980 | 0.700 | +0.280 | [+0.160, +0.400] | 14 / 0 / 36 | higher | < 0.001 | < 0.001 | primary | - |
| venue_split vs hybrid_rerank | recall@10 | labeled | 90 | 0.656 | 0.691 | −0.034 | [−0.079, +0.003] | 4 / 10 / 76 | higher | - | 0.111 | secondary | 0.238 |
| venue_split vs hybrid_rerank | recall@10 | labeled comparison | 50 | 0.559 | 0.620 | −0.062 | [−0.140, +0.006] | 4 / 10 / 36 | higher | - | 0.109 | secondary | 0.435 |
| venue_split_oracle (oracle) † vs hybrid_rerank | venue_coverage@10 | multi-venue | 59 | 1.000 | 0.661 | +0.339 | [+0.220, +0.458] | 20 / 0 / 39 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split_oracle (oracle) † vs hybrid_rerank | venue_coverage@10 | comparison | 50 | 1.000 | 0.700 | +0.300 | [+0.180, +0.440] | 15 / 0 / 35 | higher | < 0.001 | < 0.001 | secondary | < 0.001 |
| venue_split_oracle (oracle) † vs hybrid_rerank | recall@10 | labeled | 90 | 0.649 | 0.691 | −0.042 | [−0.090, +0.000] | 6 / 13 / 71 | higher | - | 0.079 | secondary | 0.238 |
| venue_split_oracle (oracle) † vs hybrid_rerank | recall@10 | labeled comparison | 50 | 0.572 | 0.620 | −0.048 | [−0.130, +0.025] | 6 / 10 / 34 | higher | - | 0.231 | secondary | 0.551 |
| venue_split_oracle (oracle) † vs venue_split | venue_coverage@10 | multi-venue | 59 | 1.000 | 0.898 | +0.102 | [+0.034, +0.186] | 6 / 0 / 53 | higher | 0.031 | 0.030 | secondary | 0.119 |
| venue_split_oracle (oracle) † vs venue_split | venue_coverage@10 | comparison | 50 | 1.000 | 0.980 | +0.020 | [+0.000, +0.060] | 1 / 0 / 49 | higher | 1.000 | 1.000 | secondary | 1.000 |
| venue_split_oracle (oracle) † vs venue_split | recall@10 | labeled | 90 | 0.649 | 0.656 | −0.007 | [−0.030, +0.013] | 2 / 3 / 85 | higher | - | 0.627 | secondary | 0.627 |
| venue_split_oracle (oracle) † vs venue_split | recall@10 | labeled comparison | 50 | 0.572 | 0.559 | +0.013 | [+0.000, +0.037] | 2 / 0 / 48 | higher | - | 0.498 | secondary | 0.551 |
| hybrid_rerank vs dense | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.508 | +0.153 | [+0.000, +0.305] | 16 / 7 / 36 | higher | 0.093 | 0.089 | secondary | 0.267 |
| hybrid_rerank vs dense | venue_coverage@10 | comparison | 50 | 0.700 | 0.520 | +0.180 | [+0.020, +0.340] | 14 / 5 / 31 | higher | 0.064 | 0.064 | secondary | 0.255 |
| hybrid_rerank vs dense | recall@10 | labeled | 90 | 0.691 | 0.472 | +0.218 | [+0.120, +0.316] | 42 / 13 / 35 | higher | - | < 0.001 | secondary | 0.002 |
| hybrid_rerank vs dense | recall@10 | labeled comparison | 50 | 0.620 | 0.353 | +0.267 | [+0.143, +0.388] | 26 / 7 / 17 | higher | - | < 0.001 | secondary | 0.001 |
| hybrid_rerank vs bm25 | venue_coverage@10 | multi-venue | 59 | 0.661 | 0.661 | +0.000 | [−0.136, +0.136] | 9 / 9 / 41 | higher | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs bm25 | venue_coverage@10 | comparison | 50 | 0.700 | 0.700 | +0.000 | [−0.160, +0.160] | 8 / 8 / 34 | higher | 1.000 | 1.000 | secondary | 1.000 |
| hybrid_rerank vs bm25 | recall@10 | labeled | 90 | 0.691 | 0.508 | +0.182 | [+0.087, +0.277] | 38 / 15 / 37 | higher | - | < 0.001 | secondary | 0.004 |
| hybrid_rerank vs bm25 | recall@10 | labeled comparison | 50 | 0.620 | 0.465 | +0.155 | [+0.021, +0.287] | 21 / 10 / 19 | higher | - | 0.029 | secondary | 0.145 |

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
| dense | 30 | 3.4 | 0.0 | 0.0 | 34 |
| hybrid | 27 | 6.0 | 0.1 | 0.0 | 33 |
| hybrid_rerank | 33 | 7.3 | 0.1 | 4,726 | 4,765 |
| venue_split | 44 | 12 | 0.1 | 5,587 | 5,631 |
| venue_split_oracle | 67 | 20 | 0.2 | 7,683 | 7,785 |
