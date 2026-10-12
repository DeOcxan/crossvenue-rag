# venue_split: one cross-encoder call per venue against one call for all venues

48 questions venue_split splits, 1 round(s), both modes per question back to back, order alternating; models loaded and warmed up before timing. Load average before timing: [13.1, 11.17, 8.59] on 10 CPUs; 1-minute load during timing: {'median': 4.09, 'max': 5.0}. Each pair was timed between two checks that found no other crossvenue, pytest or uvicorn process; 21 pair(s) with such a process at the second check were discarded and repeated.

| mode | n | total p50 ms | total p95 ms | total mean ms | rerank p50 ms | rerank p95 ms | calls per query | pairs per call | identical retrievals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per_venue | 48 | 5659.3 | 7486.9 | 5518.9 | 5292.1 | 7393.6 | 2.0 | 50.0 | 48 |
| one_call | 48 | 5208.5 | 7520.5 | 5308.8 | 5034.2 | 6861.1 | 1.0 | 100.0 | 48 |
