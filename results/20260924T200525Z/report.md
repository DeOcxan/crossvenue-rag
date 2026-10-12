# Retrieval ablation

- run: `20260924T200525Z`
- questions: 114 (0 with gold labels)
- embedding: `BAAI/bge-base-en-v1.5`
- reranker: `BAAI/bge-reranker-base`
- chunking: max 450 tokens, overlap 60
- RRF k: 60

## Metrics

No gold labels yet, so no retrieval metrics. Run `crossvenue label` to mark relevant chunks, then re-run this evaluation.

## Latency (p50, ms per step)

| method | encode | search | fuse | rerank | total p50 |
|---|---|---|---|---|---|
| bm25 | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 |
| dense | 19.1 | 45.6 | 0.0 | 0.0 | 64.1 |
| hybrid | 18.0 | 5.5 | 0.1 | 0.0 | 24.1 |
| hybrid_rerank | 46.3 | 50.5 | 0.3 | 3375.4 | 3508.6 |
