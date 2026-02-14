# ToG-data Experiment Analysis

## Core Findings
- Tiered-Forest accuracy improved from 49.6% to 68.9% (+19.3 pp).
- Optimized Tiered-Forest is +0.7 pp vs ToG (68.2%).
- Avg latency dropped from 1.79s to 0.91s.
- Avg cost/query changed from $0.000023 to $0.000130.

## Experiment Process
- Baseline (1000): Tiered-Forest 49.6%, ToG 68.2%.
- Diagnostic run (200, almost all Tier-3): Tiered-Forest 54.5%.
- Optimized run (200): Tiered-Forest 74.5%.
- Optimized full run (1000): Tiered-Forest 68.9% (exceeds ToG).

## Transition Analysis (Before -> After)
- Wrong -> Correct: 246
- Correct -> Wrong: 53
- Correct -> Correct: 443
- Wrong -> Wrong: 258

## Head-to-Head (After TF vs ToG)
- After TF wins: 113
- ToG wins: 106
- Tie: 781

## Generated Figures
- 01_main_metrics.png
- 02_accuracy_by_prefix.png
- 03_before_after_transition.png
- 04_head_to_head_vs_tog.png
- 05_experiment_timeline.png
