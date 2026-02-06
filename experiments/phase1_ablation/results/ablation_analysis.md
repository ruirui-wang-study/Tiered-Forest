# Ablation Study Analysis Report

**Total Experiments**: 24

**Variants Tested**: E0, A1, A2, A3, A4, A5, A6, A7

**Datasets**: MetaQA, WebQSP, Logistics

---

## Dataset: MetaQA

| Variant | Accuracy | Tokens | Cost ($) | Latency (s) | Tier1 Hit% | Fast-Pass% |
|---------|----------|--------|----------|-------------|------------|------------|
| **A3** | 66.7% | 1,948 | $0.0052 | 2.29 | 0% | 7% |
| **A2** | 66.7% | 978 | $0.0046 | 1.85 | 0% | 0% |
| **A6** | 66.7% | 1,889 | $0.0047 | 2.19 | 0% | 7% |
| **A5** | 66.7% | 994 | $0.0048 | 1.84 | 0% | 0% |
| **A4** | 60.0% | 2,040 | $0.0054 | 2.26 | 0% | 0% |
| **A7** | 60.0% | 2,147 | $0.0063 | 2.64 | 0% | 0% |
| **E0** | 53.3% | 1,897 | $0.0043 | 2.23 | 0% | 20% |
| **A1** | 53.3% | 1,998 | $0.0049 | 2.29 | 0% | 0% |

### Key Findings

**Baseline (E0)**: 53.3% accuracy, 1,897 tokens

- **A3**: Tokens +3%, Accuracy +13.3pp
- **A2**: Tokens -48%, Accuracy +13.3pp
- **A6**: Tokens -0%, Accuracy +13.3pp
- **A5**: Tokens -48%, Accuracy +13.3pp
- **A4**: Tokens +8%, Accuracy +6.7pp
- **A7**: Tokens +13%, Accuracy +6.7pp
- **A1**: Tokens +5%, Accuracy +0.0pp

---

## Dataset: WebQSP

| Variant | Accuracy | Tokens | Cost ($) | Latency (s) | Tier1 Hit% | Fast-Pass% |
|---------|----------|--------|----------|-------------|------------|------------|
| **A3** | 73.3% | 1,450 | $0.0021 | 1.62 | 0% | 0% |
| **E0** | 66.7% | 1,441 | $0.0021 | 1.60 | 0% | 7% |
| **A1** | 66.7% | 1,454 | $0.0023 | 1.44 | 0% | 7% |
| **A2** | 66.7% | 627 | $0.0020 | 1.17 | 0% | 0% |
| **A4** | 66.7% | 1,457 | $0.0022 | 2.08 | 0% | 0% |
| **A5** | 66.7% | 626 | $0.0020 | 1.16 | 0% | 0% |
| **A6** | 66.7% | 1,467 | $0.0022 | 2.12 | 0% | 0% |
| **A7** | 66.7% | 1,441 | $0.0021 | 1.81 | 0% | 7% |

### Key Findings

**Baseline (E0)**: 66.7% accuracy, 1,441 tokens

- **A3**: Tokens +1%, Accuracy +6.7pp
- **A1**: Tokens +1%, Accuracy +0.0pp
- **A2**: Tokens -56%, Accuracy +0.0pp
- **A4**: Tokens +1%, Accuracy +0.0pp
- **A5**: Tokens -57%, Accuracy +0.0pp
- **A6**: Tokens +2%, Accuracy +0.0pp
- **A7**: Tokens +0%, Accuracy +0.0pp

---

## Dataset: Logistics

| Variant | Accuracy | Tokens | Cost ($) | Latency (s) | Tier1 Hit% | Fast-Pass% |
|---------|----------|--------|----------|-------------|------------|------------|
| **E0** | 0.0% | 1,869 | $0.0019 | 1.50 | 7% | 0% |
| **A1** | 0.0% | 2,001 | $0.0020 | 1.25 | 0% | 0% |
| **A2** | 0.0% | 784 | $0.0017 | 0.84 | 7% | 0% |
| **A3** | 0.0% | 1,868 | $0.0019 | 1.34 | 7% | 0% |
| **A4** | 0.0% | 1,999 | $0.0020 | 1.25 | 0% | 0% |
| **A5** | 0.0% | 785 | $0.0017 | 0.88 | 7% | 0% |
| **A6** | 0.0% | 1,866 | $0.0019 | 1.04 | 7% | 0% |
| **A7** | 0.0% | 1,865 | $0.0019 | 1.18 | 7% | 0% |

### Key Findings

**Baseline (E0)**: 0.0% accuracy, 1,869 tokens

- **A1**: Tokens +7%, Accuracy +0.0pp
- **A2**: Tokens -58%, Accuracy +0.0pp
- **A3**: Tokens -0%, Accuracy +0.0pp
- **A4**: Tokens +7%, Accuracy +0.0pp
- **A5**: Tokens -58%, Accuracy +0.0pp
- **A6**: Tokens -0%, Accuracy +0.0pp
- **A7**: Tokens -0%, Accuracy +0.0pp

---

## Overall Summary

### Average Performance Across Datasets

| variant   |   accuracy |   total_tokens |   cost_usd |   latency_total |
|:----------|-----------:|---------------:|-----------:|----------------:|
| A1        |     0.4    |       1817.67  |     0.0031 |          1.6618 |
| A2        |     0.4444 |        796.333 |     0.0028 |          1.2863 |
| A3        |     0.4667 |       1755.33  |     0.0031 |          1.7491 |
| A4        |     0.4222 |       1832     |     0.0032 |          1.8665 |
| A5        |     0.4444 |        801.667 |     0.0028 |          1.2926 |
| A6        |     0.4444 |       1740.67  |     0.0029 |          1.7827 |
| A7        |     0.4222 |       1817.67  |     0.0034 |          1.878  |
| E0        |     0.4    |       1735.67  |     0.0027 |          1.7756 |

---

**Report Generated**: 2026-01-31 13:17:02
