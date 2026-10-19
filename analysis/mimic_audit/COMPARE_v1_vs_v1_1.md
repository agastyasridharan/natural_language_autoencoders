# Lock decision: v1 vs v1_1

Identical 128 held-out tasks. Discriminator = Sonnet 4.6, style only. Target = 50% (two-sided |acc-0.5| = 0).

## Discriminator

| | v1 | v1_1 |
|---|---|---|
| disc acc | 43.8% | 47.2% |
| two-sided \|acc-0.5\| | 0.062 | 0.028 |

## Style fraction signatures (mimic vs real; closer to real = better)

| signature | real | v1 | v1_1 |
|---|---|---|---|
| has_repeat | 39% | 11% | 12% |
| quote_unbalanced | 33% | 4% | 2% |
| ends_ellipsis | 67% | 99% | 98% |
| ends_midthought | 16% | 0% | 0% |
| has_slash_class | 68% | 100% | 100% |
| **total drift** | 0 | **1.39** | **1.39** |

## Numeric band check (OVER = caricature, disqualifying)

- v1 OVER: none
- v1_1 OVER: none

## Recommendation

**Lock: v1_1** (lowest two-sided discriminator distance with no OVER signatures; fraction drift as tie-break).
