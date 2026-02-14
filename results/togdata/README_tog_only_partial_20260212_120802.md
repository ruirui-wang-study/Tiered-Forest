# ToG Partial Result Saved

- Source run: `python TieredForest-Benchmark/run_togdata_benchmark.py --dataset webqsp --split test --limit 1000 --backend wikidata`
- Run was manually stopped during `Tiered-Forest` phase.
- ToG phase had already finished and emitted summary:
  - `ToG => EM=68.20%, cost=$0.383834, avg_latency=0.00s`

Artifacts:
- Summary CSV: `
results\togdata\summary_webqsp_wikidata_tog_only_partial_20260212_120802.csv
`
- ToG cache snapshot: `
results\togdata\webqsp_tog_cache_snapshot_20260212_120802.json
`
- Source log: `TieredForest-Benchmark/logs/benchmark.log/togdata_webqsp_1000_20260212_094510.out.log`
