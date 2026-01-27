# Threshold Sensitivity Analysis Report

**Tested tau_high values**: [np.float64(0.7), np.float64(0.8), np.float64(0.85), np.float64(0.9)]

**Datasets**: MetaQA, WebQSP, Logistics

---

## Dataset: MetaQA

| tau_high | Accuracy | Tokens | Cost ($) | Fast-Pass% | Escalate% |
|----------|----------|--------|----------|------------|----------|
| 0.70 | 0.0% | 1,297 | $0.0003 | 100.0% | 0.0% |
| 0.80 | 0.0% | 1,242 | $0.0002 | 100.0% | 0.0% |
| 0.85 | 0.0% | 1,456 | $0.0003 | 100.0% | 0.0% |
| 0.90 | 0.0% | 1,237 | $0.0002 | 100.0% | 0.0% |

### Recommendations

- **Best Accuracy**: tau_high=0.70 (0.0%)
- **Best Cost**: tau_high=0.90 (1,237 tokens)

---

## Dataset: WebQSP

| tau_high | Accuracy | Tokens | Cost ($) | Fast-Pass% | Escalate% |
|----------|----------|--------|----------|------------|----------|
| 0.70 | 40.0% | 1,286 | $0.0018 | 93.3% | 6.7% |
| 0.80 | 46.7% | 1,190 | $0.0018 | 93.3% | 6.7% |
| 0.85 | 46.7% | 1,564 | $0.0035 | 86.7% | 13.3% |
| 0.90 | 60.0% | 1,244 | $0.0018 | 93.3% | 6.7% |

### Recommendations

- **Best Accuracy**: tau_high=0.90 (60.0%)
- **Best Cost**: tau_high=0.80 (1,190 tokens)

---

## Dataset: Logistics

| tau_high | Accuracy | Tokens | Cost ($) | Fast-Pass% | Escalate% |
|----------|----------|--------|----------|------------|----------|
| 0.70 | 20.0% | 1,848 | $0.0004 | 93.3% | 0.0% |
| 0.80 | 13.3% | 1,713 | $0.0003 | 93.3% | 0.0% |
| 0.85 | 20.0% | 1,854 | $0.0004 | 93.3% | 0.0% |
| 0.90 | 20.0% | 1,754 | $0.0004 | 93.3% | 0.0% |

### Recommendations

- **Best Accuracy**: tau_high=0.70 (20.0%)
- **Best Cost**: tau_high=0.80 (1,713 tokens)

---

## Overall Summary

|   tau_high |   accuracy |   total_tokens |   cost_usd |   fast_pass_rate |
|-----------:|-----------:|---------------:|-----------:|-----------------:|
|       0.7  |     0.2    |        1477    |     0.0008 |           0.9556 |
|       0.8  |     0.2    |        1381.67 |     0.0008 |           0.9556 |
|       0.85 |     0.2222 |        1624.67 |     0.0014 |           0.9333 |
|       0.9  |     0.2667 |        1411.67 |     0.0008 |           0.9556 |

---

**Report Generated**: 2026-01-27 20:41:45
