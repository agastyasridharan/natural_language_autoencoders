# Cycle-consistency verdict report

run `gemma12b-cp7-fineweb-atoms`  claims 298  k 5

## Verdict distribution

| verdict | n | % |
|---|---|---|
| grounded | 13 | 4.4 |
| entangled | 13 | 4.4 |
| seeded | 84 | 28.2 |
| confabulated | 95 | 31.9 |
| unresolved | 93 | 31.2 |

unresolved by reason: power 46, validity 32, no text-passing decoys 15

## Gate flow (spec Step 7)

- 298 claims -> 93 unresolved (pre-checks), 205 reached the gates
- Gate 1 (specificity): 26 pass -> Gate 2, 179 fail -> Gate 3
- Gate 2 (ownership): 13 grounded, 13 entangled
- Gate 3 (shared signal): 84 seeded, 95 confabulated

## Atom -> cycle verdict shift

rows = published atom verdict, columns = cycle verdict

| atom \ cycle | grounded | entangled | seeded | confabulated | unresolved |
|---|---|---|---|---|---|
| confabulated | 1 | 8 | 32 | 89 | 50 |
| entangled | 0 | 2 | 20 | 3 | 13 |
| grounded | 12 | 3 | 32 | 3 | 30 |

## Pre-checks + grounding

- power passed 223/298, validity passed 239/298
- grounding estimate (M_BB - med D*, Gate-1 passers only (grounded+entangled; M_BB > 0), n=26): median 5.5115, p10 2.7292, p90 9.922

## kappa_0 re-fit (Gate 1 null on THIS run)

- calibration kappa_0 = 1.2 (applied)
- re-fit on 154 survivor D* families = 1.0  **DIVERGES > 0.1**
- decoy-vs-decoy exceedance rate at the calibration kappa_0 = 0.0436 (target <= 0.05, whole run incl. in-sample)
- **out-of-sample** (75 activations disjoint from the 25 calibration ones, 116 families): re-fit kappa_0 = 1.2, FP rate at calibration kappa_0 = 0.045 -> TRANSFERS (<= 0.05)
- _This re-fit is IN-SAMPLE when the run activations are the calibration split (e.g. the S4 small-N run reuses the 25 calibration activations): the LODO families and grid are then identical to calib_cycle.py's, so the FP rate is a consistency check, NOT out-of-sample validation. Genuine out-of-sample validation of kappa_0 requires a run on activations disjoint from calibration (S5's 375 new activations). WARNING: kappa_0_refit disagrees with the applied calibration kappa_0 by > 0.1 - the applied kappa_0 is stale for this set._

## Per-claim

| idx.ci | type | atom | cycle | gate | M_BB | gate1_bar | ground_est | surv | pw | vd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | detail | confabulated | seeded | 3 | 0.10 | 1.11 | 1.07 | 7 | P | P |
| 0.5 | theme | confabulated | unresolved |  | -0.43 | 1.57 | 1.90 | 5 | . | . |
| 0.7 | theme | confabulated | confabulated | 3 | -0.01 | 0.38 | 2.06 | 5 | P | P |
| 4.0 | theme | grounded | unresolved |  | 27.00 | 33.02 | 1.76 | 18 | . | P |
| 4.4 | theme | entangled | unresolved |  | 0.09 | 2.96 | -0.35 | 4 | . | . |
| 4.9 | detail | confabulated | confabulated | 3 | -0.26 | 0.21 | 1.24 | 5 | P | P |
| 8.6 | theme | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 8.10 | theme | confabulated | seeded | 3 | 0.34 | 1.58 | 0.22 | 5 | P | P |
| 8.13 | detail | confabulated | seeded | 3 | 0.46 | 5.62 | 2.74 | 7 | P | P |
| 12.0 | theme | confabulated | confabulated | 3 | -0.16 | 1.73 | 0.45 | 6 | . | P |
| 12.4 | theme | confabulated | unresolved |  | 0.07 | 1.42 | 0.84 | 5 | . | P |
| 12.8 | detail | confabulated | seeded | 3 | 0.07 | 6.49 | -3.35 | 5 | P | P |
| 16.0 | theme | grounded | seeded | 3 | 2.67 | 4.45 | 0.31 | 6 | P | P |
| 16.4 | theme | grounded | seeded | 3 | 2.65 | 6.19 | -1.45 | 7 | P | P |
| 16.13 | detail | grounded | unresolved |  | 4.43 | 0.67 | 7.41 | 4 | P | . |
| 20.3 | theme | confabulated | confabulated | 3 | -0.12 | -0.43 | 1.94 | 5 | P | P |
| 20.8 | detail | confabulated | confabulated | 3 | -0.24 | 1.35 | 1.19 | 5 | P | P |
| 20.9 | theme | confabulated | unresolved |  | -0.00 | 1.33 | 0.02 | 4 | P | . |
| 24.3 | theme | confabulated | seeded | 3 | 0.34 | 2.50 | 0.06 | 7 | P | P |
| 24.6 | detail | confabulated | unresolved |  | 0.00 | 4.99 | -3.00 | 2 | P | . |
| 24.14 | theme | grounded | unresolved |  | 7.14 |  |  |  | . | . |
| 28.0 | theme | grounded | seeded | 3 | 3.43 | 10.45 | -0.97 | 14 | P | P |
| 28.1 | theme | grounded | unresolved |  | 4.79 | 6.02 | 0.89 | 6 | . | P |
| 28.10 | detail | entangled | seeded | 3 | 4.45 | 7.86 | 0.01 | 5 | P | P |
| 32.4 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 32.7 | detail | confabulated | confabulated | 3 | -0.26 | 0.86 | 2.75 | 5 | P | P |
| 32.13 | theme | confabulated | confabulated | 3 | 0.04 | 3.08 | 1.00 | 10 | P | P |
| 36.1 | theme | confabulated | unresolved |  | -1.17 | -0.44 | 2.35 | 5 | P | . |
| 36.5 | theme | confabulated | confabulated | 3 | -0.00 | 1.57 | -0.17 | 11 | . | P |
| 36.6 | detail | entangled | seeded | 3 | 0.16 | 3.50 | -1.51 | 5 | P | P |
| 40.3 | theme | grounded | unresolved |  | 5.50 | 7.04 | 3.48 | 5 | . | P |
| 40.5 | theme | entangled | seeded | 3 | 0.83 | 1.89 | 0.49 | 6 | P | P |
| 40.12 | detail | grounded | unresolved |  | 6.84 |  |  |  | . | . |
| 44.1 | theme | grounded | unresolved |  | 5.34 | 6.65 | 1.07 | 5 | . | P |
| 44.4 | theme | confabulated | unresolved |  | -0.40 |  |  |  | . | . |
| 44.6 | detail | grounded | unresolved |  | 9.43 | 8.07 | 8.72 | 5 | P | . |
| 48.2 | theme | confabulated | unresolved |  | -2.49 |  |  |  | . | . |
| 48.5 | theme | confabulated | confabulated | 3 | -0.00 | 1.46 | -0.14 | 5 | . | P |
| 48.8 | detail | confabulated | confabulated | 3 | -0.17 | 1.72 | 1.36 | 7 | P | P |
| 52.2 | theme | confabulated | confabulated | 3 | -0.05 | 2.29 | 0.01 | 21 | P | P |
| 52.4 | theme | confabulated | confabulated | 3 | -0.06 | 4.16 | -1.71 | 21 | P | P |
| 52.13 | detail | grounded | unresolved |  | 1.76 | 4.44 | -0.60 | 5 | P | . |
| 56.3 | detail | confabulated | confabulated | 3 | 0.00 | -0.30 | 4.85 | 5 | P | P |
| 56.4 | theme | confabulated | unresolved |  | -0.02 |  |  |  | . | . |
| 56.6 | theme | confabulated | confabulated | 3 | -0.26 | 0.65 | 0.97 | 5 | P | P |
| 61.2 | theme | grounded | seeded | 3 | 10.25 | 14.84 | 3.24 | 6 | P | P |
| 61.7 | detail | confabulated | confabulated | 3 | -0.83 | 0.30 | 2.91 | 6 | P | P |
| 61.13 | theme | confabulated | confabulated | 3 | -1.20 | -1.39 | 4.92 | 5 | P | P |
| 65.0 | theme | confabulated | seeded | 3 | 0.66 | 1.91 | 0.62 | 5 | P | P |
| 65.3 | theme | confabulated | unresolved |  | 1.55 | 4.39 | -0.29 | 8 | . | P |
| 65.8 | detail | confabulated | entangled | 2 | 1.53 | 0.56 | 5.49 | 12 | P | P |
| 69.2 | detail | confabulated | unresolved |  | 0.37 | 0.18 | 1.76 | 2 | P | . |
| 69.6 | theme | confabulated | unresolved |  | 0.11 | 1.72 | -0.01 | 10 | . | P |
| 69.15 | theme | confabulated | confabulated | 3 | -1.55 | -3.12 | 4.50 | 6 | P | P |
| 73.1 | theme | grounded | seeded | 3 | 4.12 | 8.06 | 1.73 | 8 | P | P |
| 73.6 | theme | entangled | seeded | 3 | 0.95 | 2.70 | -0.12 | 5 | P | P |
| 73.13 | detail | grounded | seeded | 3 | 3.02 | 6.18 | 2.24 | 7 | P | P |
| 77.1 | theme | confabulated | confabulated | 3 | -1.03 | -0.26 | 1.13 | 5 | P | P |
| 77.4 | detail | grounded | unresolved |  | 5.25 |  |  | 0 | . | . |
| 77.6 | theme | confabulated | confabulated | 3 | 0.05 | 1.39 | -0.02 | 6 | P | P |
| 81.3 | theme | entangled | seeded | 3 | 0.88 | 6.74 | -2.92 | 8 | P | P |
| 81.8 | theme | entangled | unresolved |  | 0.07 | 1.38 | 0.12 | 7 | . | P |
| 81.12 | detail | entangled | unresolved |  | 1.06 | 2.96 | -0.40 | 5 | . | P |
| 85.3 | theme | confabulated | confabulated | 3 | -0.01 | 2.17 | -0.27 | 8 | P | P |
| 85.4 | theme | grounded | unresolved |  | 2.07 | 4.43 | 0.54 | 6 | . | P |
| 85.12 | detail | grounded | entangled | 2 | 4.74 | 3.74 | 5.03 | 8 | P | P |
| 89.0 | theme | grounded | grounded | 2 | 7.79 | 4.30 | 6.20 | 8 | P | P |
| 89.4 | detail | confabulated | entangled | 2 | 0.20 | -1.46 | 5.56 | 8 | P | P |
| 89.16 | theme | confabulated | confabulated | 3 | -1.41 | -1.43 | 2.50 | 6 | P | P |
| 93.4 | theme | confabulated | unresolved |  | 0.88 | 4.44 | -0.13 | 5 | P | . |
| 93.6 | theme | confabulated | confabulated | 3 | -0.03 | 1.71 | -0.38 | 5 | P | P |
| 93.9 | detail | confabulated | confabulated | 3 | 0.25 | 3.19 | -0.55 | 8 | P | P |
| 97.0 | theme | grounded | seeded | 3 | 5.92 | 16.03 | -1.87 | 5 | P | P |
| 97.1 | theme | confabulated | seeded | 3 | 0.03 | 2.60 | -0.66 | 5 | P | P |
| 97.5 | detail | confabulated | grounded | 2 | 0.00 | -0.12 | 2.12 | 5 | P | P |
| 101.0 | theme | grounded | seeded | 3 | 5.83 | 10.30 | -0.35 | 6 | P | P |
| 101.3 | theme | confabulated | unresolved |  | 0.67 | 2.70 | 0.09 | 5 | P | . |
| 101.6 | detail | grounded | unresolved |  | 5.33 | 14.44 | -1.95 | 6 | . | P |
| 105.2 | detail | grounded | confabulated | 3 | 1.08 | 6.41 | -1.63 | 6 | P | P |
| 105.3 | theme | grounded | seeded | 3 | 1.53 | 4.48 | -1.39 | 8 | P | P |
| 105.8 | theme | entangled | seeded | 3 | 0.40 | 2.65 | -0.15 | 6 | P | P |
| 109.1 | theme | grounded | seeded | 3 | 14.08 | 21.29 | -1.92 | 6 | P | P |
| 109.4 | theme | confabulated | confabulated | 3 | -0.17 | 2.58 | 0.35 | 10 | P | P |
| 109.11 | detail | grounded | seeded | 3 | 1.97 | 2.78 | 1.47 | 7 | P | P |
| 113.7 | theme | entangled | unresolved |  | 0.01 | 1.47 | -0.02 | 18 | . | P |
| 113.9 | theme | confabulated | unresolved |  | 0.04 | 2.60 | -0.30 | 5 | . | . |
| 113.12 | detail | grounded | unresolved |  | 3.56 | 7.54 | -1.64 | 6 | . | P |
| 117.0 | theme | entangled | seeded | 3 | 1.91 | 5.10 | 0.16 | 5 | P | P |
| 117.8 | theme | confabulated | seeded | 3 | 0.03 | 1.58 | 0.02 | 5 | P | P |
| 117.9 | detail | grounded | unresolved |  | 1.98 | 3.07 | 2.90 | 5 | . | P |
| 121.5 | theme | confabulated | confabulated | 3 | -1.13 | 0.13 | 0.42 | 6 | P | P |
| 121.8 | theme | entangled | unresolved |  | 0.22 | 1.58 | 0.24 | 13 | . | P |
| 125.0 | theme | grounded | unresolved |  | 1.70 | 16.44 | -0.92 | 4 | P | . |
| 125.1 | theme | confabulated | confabulated | 3 | -0.01 | 12.13 | -5.02 | 6 | P | P |
| 125.7 | detail | confabulated | confabulated | 3 | -0.14 | 4.69 | -1.04 | 6 | P | P |
| 129.0 | theme | grounded | unresolved |  | 3.52 |  |  |  | . | . |
| 129.2 | theme | grounded | unresolved |  | 1.61 | 2.52 | 0.87 | 7 | . | P |
| 129.13 | detail | confabulated | confabulated | 3 | -0.12 | -1.07 | 3.67 | 6 | P | P |
| 133.0 | theme | grounded | unresolved |  | 8.95 | 25.96 | -7.27 | 5 | P | . |
| 133.5 | detail | confabulated | seeded | 3 | 0.82 | 0.87 | 3.04 | 12 | P | P |
| 133.7 | theme | confabulated | confabulated | 3 | -0.08 | 2.37 | 0.44 | 7 | P | P |
| 137.0 | theme | confabulated | confabulated | 3 | -0.89 | 3.96 | -1.13 | 11 | P | P |
| 137.3 | theme | confabulated | unresolved |  | 0.10 | 4.69 | 0.44 | 6 | P | . |
| 137.11 | detail | confabulated | confabulated | 3 | -0.15 | 0.58 | 0.82 | 5 | P | P |
| 141.1 | theme | grounded | unresolved |  | 0.63 |  |  | 0 | . | . |
| 141.3 | detail | entangled | seeded | 3 | 0.95 | 2.01 | 0.59 | 5 | P | P |
| 141.16 | theme | confabulated | unresolved |  | -0.23 | 1.55 | -0.12 | 5 | P | . |
| 145.1 | theme | grounded | unresolved |  | 5.74 | 12.94 | -1.91 | 4 | P | . |
| 145.2 | theme | confabulated | confabulated | 3 | 0.17 | 2.69 | -0.36 | 6 | P | P |
| 145.8 | detail | confabulated | confabulated | 3 | 0.08 | 7.09 | -2.47 | 6 | P | P |
| 149.4 | theme | confabulated | confabulated | 3 | 0.50 | 5.16 | -2.42 | 5 | P | P |
| 149.7 | theme | confabulated | seeded | 3 | 0.05 | 4.17 | -1.66 | 19 | P | P |
| 149.12 | detail | confabulated | confabulated | 3 | -0.20 | 1.27 | 0.35 | 5 | P | P |
| 153.4 | theme | confabulated | confabulated | 3 | -0.13 | 0.71 | 0.63 | 5 | P | P |
| 153.5 | theme | grounded | unresolved |  | 5.40 | 15.41 | -2.34 | 9 | . | P |
| 153.8 | detail | grounded | unresolved |  | 11.43 | 14.45 | 3.23 | 7 | . | P |
| 157.7 | theme | confabulated | confabulated | 3 | -0.13 | 2.38 | -1.14 | 5 | P | P |
| 157.9 | theme | confabulated | confabulated | 3 | -0.58 | 1.25 | -0.12 | 5 | . | P |
| 157.12 | detail | entangled | confabulated | 3 | 1.25 | 3.52 | 1.89 | 5 | P | P |
| 161.0 | theme | grounded | seeded | 3 | 8.83 | 13.75 | -0.50 | 5 | P | P |
| 161.1 | detail | confabulated | unresolved |  | 2.72 |  |  | 0 | . | . |
| 161.4 | theme | confabulated | seeded | 3 | 0.33 | 1.76 | 0.13 | 6 | P | P |
| 165.0 | theme | confabulated | unresolved |  | -1.74 | 0.04 | 1.79 | 5 | P | . |
| 165.1 | detail | confabulated | seeded | 3 | 0.17 | 2.48 | -0.82 | 5 | P | P |
| 165.2 | theme | entangled | unresolved |  | 0.47 | -0.17 | 6.13 | 5 | P | . |
| 169.0 | theme | confabulated | confabulated | 3 | -0.28 | 2.08 | -0.68 | 6 | P | P |
| 169.3 | detail | grounded | seeded | 3 | 3.32 | 6.89 | 0.22 | 6 | P | P |
| 169.7 | theme | confabulated | unresolved |  | -0.05 |  |  |  | . | . |
| 173.6 | theme | confabulated | confabulated | 3 | -0.10 | 1.99 | -0.37 | 6 | . | P |
| 173.7 | detail | confabulated | confabulated | 3 | -0.10 | 1.12 | 0.63 | 7 | P | P |
| 173.8 | theme | confabulated | confabulated | 3 | -0.00 | 1.28 | 0.00 | 8 | P | P |
| 177.0 | theme | grounded | seeded | 3 | 19.66 | 29.95 | -5.11 | 5 | P | P |
| 177.2 | theme | grounded | grounded | 2 | 12.47 | 6.03 | 10.00 | 5 | P | P |
| 177.3 | detail | confabulated | unresolved |  | 0.00 | 2.35 | -0.25 | 3 | P | . |
| 181.1 | detail | confabulated | confabulated | 3 | -0.82 | 1.09 | -0.39 | 6 | P | P |
| 181.4 | theme | entangled | seeded | 3 | 0.16 | 1.77 | 0.16 | 10 | P | P |
| 181.6 | theme | confabulated | confabulated | 3 | -0.01 | 1.38 | 0.05 | 7 | . | P |
| 185.8 | theme | confabulated | unresolved |  | 0.01 | 0.96 | 0.76 | 5 | P | . |
| 185.9 | detail | confabulated | seeded | 3 | 0.08 | 4.28 | 2.56 | 18 | P | P |
| 185.10 | theme | confabulated | confabulated | 3 | -0.66 | 1.61 | 0.05 | 5 | P | P |
| 189.10 | detail | grounded | seeded | 3 | 4.98 | 7.18 | 0.13 | 5 | P | P |
| 189.14 | theme | confabulated | confabulated | 3 | -1.11 | 0.34 | 0.41 | 6 | P | P |
| 189.15 | theme | confabulated | confabulated | 3 | -0.42 | -1.02 | 4.12 | 5 | P | P |
| 193.2 | theme | confabulated | confabulated | 3 | -0.97 | 0.52 | -0.03 | 5 | P | P |
| 193.5 | theme | confabulated | seeded | 3 | 0.14 | 1.78 | 0.06 | 5 | P | P |
| 193.6 | detail | confabulated | seeded | 3 | 0.00 | 8.55 | -6.19 | 6 | P | P |
| 197.1 | theme | confabulated | confabulated | 3 | 0.46 | 0.75 | 1.39 | 5 | P | P |
| 197.5 | theme | entangled | unresolved |  | 0.72 | 1.83 | 0.67 | 5 | . | . |
| 197.8 | detail | confabulated | confabulated | 3 | -0.00 | 1.32 | 0.07 | 5 | . | P |
| 201.3 | theme | entangled | seeded | 3 | 0.71 | 3.11 | -0.36 | 5 | P | P |
| 201.10 | detail | grounded | grounded | 2 | 16.01 | 8.78 | 14.09 | 8 | P | P |
| 201.13 | theme | confabulated | confabulated | 3 | 0.07 | 3.99 | 1.02 | 5 | P | P |
| 205.1 | theme | confabulated | confabulated | 3 | -0.18 | 0.98 | 0.31 | 5 | P | P |
| 205.4 | theme | confabulated | seeded | 3 | 0.58 | 2.18 | 0.30 | 7 | P | P |
| 205.14 | detail | confabulated | confabulated | 3 | -0.30 | 1.50 | -0.35 | 5 | P | P |
| 209.3 | detail | confabulated | seeded | 3 | 0.52 | 2.10 | 0.29 | 8 | P | P |
| 209.5 | theme | confabulated | confabulated | 3 | -0.28 | 2.60 | -1.04 | 7 | P | P |
| 209.9 | theme | confabulated | confabulated | 3 | -0.00 | 1.79 | -0.20 | 6 | . | P |
| 213.0 | detail | grounded | unresolved |  | 18.88 | 18.94 | 5.27 | 6 | . | P |
| 213.4 | theme | confabulated | confabulated | 3 | 0.44 | 5.51 | -1.27 | 6 | P | P |
| 213.6 | theme | confabulated | unresolved |  | -0.00 | 1.88 | -0.19 | 4 | P | . |
| 217.0 | theme | confabulated | entangled | 2 | 0.51 | -1.17 | 5.53 | 5 | P | P |
| 217.7 | theme | confabulated | unresolved |  | -0.02 | 5.16 | -1.42 | 5 | . | . |
| 217.12 | detail | grounded | seeded | 3 | 9.92 | 12.14 | 1.87 | 5 | P | P |
| 221.0 | theme | entangled | seeded | 3 | 1.24 | 2.72 | 5.31 | 6 | P | P |
| 221.1 | theme | grounded | seeded | 3 | 8.18 | 8.44 | 3.17 | 7 | P | P |
| 221.8 | detail | confabulated | confabulated | 3 | -0.42 | 1.04 | 0.34 | 5 | . | P |
| 225.5 | detail | confabulated | confabulated | 3 | 0.11 | 1.71 | 0.11 | 5 | P | P |
| 225.9 | theme | confabulated | confabulated | 3 | -0.00 | 1.76 | -0.43 | 5 | . | P |
| 225.16 | theme | confabulated | unresolved |  | 0.03 | 2.05 | 2.96 | 16 | . | P |
| 229.4 | theme | confabulated | unresolved |  | 1.73 | 5.20 | -1.09 | 5 | P | . |
| 229.6 | theme | confabulated | unresolved |  | 0.16 | 1.73 | -0.18 | 18 | . | P |
| 229.13 | detail | confabulated | seeded | 3 | 0.65 | 2.37 | 0.08 | 5 | P | P |
| 233.4 | theme | grounded | seeded | 3 | 2.85 | 3.64 | 1.01 | 8 | P | P |
| 233.7 | theme | confabulated | seeded | 3 | 0.28 | 3.17 | -0.19 | 5 | P | P |
| 233.10 | detail | grounded | entangled | 2 | 18.45 | 12.91 | 8.99 | 5 | P | P |
| 237.1 | theme | confabulated | confabulated | 3 | -1.12 | -0.76 | 1.39 | 5 | P | P |
| 237.4 | theme | grounded | unresolved |  | 3.55 | 7.58 | -0.29 | 5 | P | . |
| 237.7 | detail | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 241.1 | theme | grounded | confabulated | 3 | 2.82 | 3.97 | 3.06 | 8 | P | P |
| 241.3 | theme | confabulated | confabulated | 3 | -0.64 | 0.53 | 0.11 | 5 | P | P |
| 241.12 | detail | grounded | seeded | 3 | 7.23 | 17.14 | -1.27 | 6 | P | P |
| 245.1 | theme | entangled | entangled | 2 | 1.24 | -0.28 | 6.60 | 6 | P | P |
| 245.5 | detail | confabulated | confabulated | 3 | 0.00 | 4.00 | 1.48 | 6 | P | P |
| 245.14 | theme | confabulated | confabulated | 3 | -0.79 | -0.44 | 5.00 | 8 | P | P |
| 249.6 | theme | confabulated | seeded | 3 | 0.00 | 1.52 | -0.19 | 5 | P | P |
| 249.7 | detail | grounded | grounded | 2 | 29.52 | 29.23 | 7.75 | 14 | P | P |
| 249.13 | theme | entangled | confabulated | 3 | 0.55 | 3.65 | -0.11 | 5 | P | P |
| 253.0 | theme | confabulated | seeded | 3 | 0.71 | 4.01 | 2.10 | 7 | P | P |
| 253.2 | theme | confabulated | entangled | 2 | 0.15 | -2.29 | 8.09 | 9 | P | P |
| 253.5 | detail | confabulated | seeded | 3 | 2.32 | 2.51 | 1.96 | 7 | P | P |
| 257.2 | detail | entangled | seeded | 3 | 0.00 | 4.73 | -1.35 | 5 | P | P |
| 257.5 | theme | confabulated | unresolved |  | -0.05 | 1.87 | 2.66 | 4 | P | . |
| 257.7 | theme | entangled | seeded | 3 | 0.76 | 3.40 | -0.17 | 14 | P | P |
| 261.1 | theme | confabulated | unresolved |  | 0.25 | 2.94 | -0.77 | 6 | . | P |
| 261.6 | theme | confabulated | unresolved |  | 0.05 | 1.20 | 0.19 | 6 | . | P |
| 261.10 | detail | confabulated | seeded | 3 | 0.10 | 1.50 | 1.42 | 11 | P | P |
| 265.0 | theme | confabulated | entangled | 2 | 0.62 | -2.28 | 6.10 | 6 | P | P |
| 265.3 | theme | confabulated | confabulated | 3 | -0.42 | -0.03 | 1.43 | 6 | P | P |
| 265.9 | detail | confabulated | confabulated | 3 | -0.05 | 1.15 | 0.10 | 5 | P | P |
| 269.3 | theme | entangled | seeded | 3 | 1.01 | 12.70 | -5.31 | 5 | P | P |
| 269.5 | theme | confabulated | confabulated | 3 | -0.00 | 1.74 | -0.42 | 7 | P | P |
| 269.8 | detail | entangled | confabulated | 3 | 0.02 | 7.07 | -1.94 | 8 | P | P |
| 273.0 | theme | entangled | seeded | 3 | 0.48 | 4.26 | -1.18 | 7 | P | P |
| 273.5 | theme | entangled | unresolved |  | 3.08 | 4.38 | 1.93 | 5 | P | . |
| 273.13 | detail | entangled | unresolved |  | 3.64 | 11.31 | -1.60 | 6 | . | P |
| 277.2 | theme | grounded | seeded | 3 | 7.06 | 14.99 | -2.86 | 6 | P | P |
| 277.7 | theme | confabulated | seeded | 3 | 0.59 | 1.73 | 1.95 | 8 | P | P |
| 277.12 | detail | confabulated | confabulated | 3 | 0.31 | 3.46 | 0.36 | 6 | P | P |
| 281.0 | theme | grounded | seeded | 3 | 16.49 | 24.72 | 6.52 | 7 | P | P |
| 281.4 | theme | confabulated | unresolved |  | 0.03 | 1.93 | -0.21 | 9 | . | P |
| 281.9 | detail | confabulated | confabulated | 3 | 0.05 | 0.55 | 0.83 | 5 | P | P |
| 285.4 | theme | confabulated | seeded | 3 | 0.08 | 2.02 | -0.30 | 6 | P | P |
| 285.6 | theme | confabulated | confabulated | 3 | -0.18 | 1.66 | 0.06 | 6 | P | P |
| 285.7 | detail | confabulated | unresolved |  | 0.45 | 2.17 | -0.21 | 14 | . | P |
| 289.0 | theme | confabulated | confabulated | 3 | -0.15 | 1.64 | -0.35 | 8 | . | P |
| 289.6 | theme | confabulated | unresolved |  | 0.06 | 1.79 | -0.10 | 11 | . | P |
| 289.11 | detail | confabulated | unresolved |  | -0.33 |  |  |  | . | . |
| 293.6 | detail | confabulated | unresolved |  | 0.03 | 7.74 | -6.51 | 1 | P | . |
| 293.7 | theme | confabulated | unresolved |  | -0.39 |  |  |  | . | . |
| 293.9 | theme | entangled | seeded | 3 | 0.47 | 3.12 | -0.37 | 6 | P | P |
| 297.4 | theme | confabulated | unresolved |  | 0.15 |  |  |  | . | . |
| 297.8 | detail | confabulated | unresolved |  | 0.21 | 1.62 | 0.51 | 5 | P | . |
| 297.11 | theme | confabulated | seeded | 3 | 0.12 | 1.37 | 0.16 | 5 | P | P |
| 301.0 | theme | confabulated | unresolved |  | 1.89 | 22.12 | -15.47 | 5 | P | . |
| 301.7 | detail | entangled | seeded | 3 | 2.92 | 6.28 | 0.79 | 7 | P | P |
| 301.11 | theme | grounded | seeded | 3 | 4.71 | 17.38 | -8.08 | 6 | P | P |
| 305.1 | detail | confabulated | seeded | 3 | 0.65 | 1.66 | 0.29 | 5 | P | P |
| 305.3 | theme | entangled | entangled | 2 | 0.78 | -0.22 | 3.21 | 8 | P | P |
| 305.13 | theme | confabulated | entangled | 2 | 0.07 | -0.91 | 4.64 | 13 | P | P |
| 309.0 | theme | grounded | seeded | 3 | 27.92 | 33.97 | 1.69 | 8 | P | P |
| 309.3 | detail | confabulated | unresolved |  | 3.47 | 7.13 | -0.09 | 6 | . | P |
| 309.4 | theme | confabulated | confabulated | 3 | -0.03 | 1.30 | 0.22 | 5 | P | P |
| 313.0 | theme | grounded | seeded | 3 | 5.80 | 12.89 | -2.02 | 5 | P | P |
| 313.1 | detail | grounded | seeded | 3 | 6.98 | 18.24 | -2.00 | 5 | P | P |
| 313.6 | theme | confabulated | unresolved |  | 0.18 | 1.63 | 0.03 | 6 | . | P |
| 317.3 | theme | confabulated | seeded | 3 | 0.96 | 5.49 | -1.40 | 5 | P | P |
| 317.4 | theme | grounded | unresolved |  | 0.91 | 4.00 | -0.63 | 5 | P | . |
| 317.14 | detail | confabulated | confabulated | 3 | -1.89 | -0.32 | 1.58 | 6 | . | P |
| 321.5 | theme | confabulated | confabulated | 3 | -1.18 | -0.78 | 0.93 | 6 | P | P |
| 321.11 | detail | grounded | grounded | 2 | 7.51 | 3.57 | 5.65 | 7 | P | P |
| 321.15 | theme | confabulated | confabulated | 3 | -2.39 | -4.77 | 6.81 | 6 | P | P |
| 325.0 | theme | confabulated | unresolved |  | 0.37 | 3.22 | 1.37 | 9 | . | . |
| 325.3 | detail | grounded | unresolved |  | 5.34 |  |  |  | . | . |
| 325.5 | detail | confabulated | confabulated | 3 | 0.20 | 2.28 | -0.48 | 5 | P | P |
| 329.0 | theme | confabulated | confabulated | 3 | -0.16 | -2.43 | 7.25 | 20 | P | P |
| 329.10 | detail | confabulated | confabulated | 3 | -0.20 | 1.48 | 1.17 | 13 | . | P |
| 329.13 | theme | confabulated | seeded | 3 | 0.11 | 0.86 | 0.65 | 6 | P | P |
| 333.2 | theme | confabulated | confabulated | 3 | -0.46 | 0.93 | 0.87 | 10 | P | P |
| 333.9 | theme | confabulated | unresolved |  | -0.09 |  |  |  | . | . |
| 333.13 | detail | confabulated | confabulated | 3 | -0.39 | 0.28 | 1.61 | 5 | P | P |
| 337.0 | theme | grounded | grounded | 2 | 35.18 | 30.59 | 9.84 | 5 | P | P |
| 337.3 | theme | grounded | grounded | 2 | 8.52 | 7.64 | 4.91 | 5 | P | P |
| 341.1 | theme | grounded | grounded | 2 | 5.49 | 5.36 | 4.48 | 6 | P | P |
| 341.7 | theme | entangled | unresolved |  | 0.07 | 1.39 | 0.39 | 6 | . | P |
| 341.12 | detail | grounded | grounded | 2 | 3.79 | 1.56 | 4.31 | 7 | P | P |
| 345.4 | theme | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 345.6 | theme | entangled | seeded | 3 | 0.09 | 1.26 | 0.14 | 5 | P | P |
| 345.7 | detail | entangled | unresolved |  | 1.07 | 11.90 | -4.76 | 2 | P | . |
| 349.0 | theme | confabulated | unresolved |  | 0.03 | 1.16 | 0.24 | 5 | . | P |
| 349.2 | theme | confabulated | confabulated | 3 | -0.18 | 2.49 | 0.27 | 9 | P | P |
| 349.16 | detail | confabulated | confabulated | 3 | -0.09 | 1.28 | 0.70 | 5 | P | P |
| 353.1 | theme | grounded | unresolved |  | 12.86 | 19.28 | 2.76 | 5 | P | . |
| 353.4 | theme | entangled | seeded | 3 | 1.50 | 5.82 | -1.67 | 7 | P | P |
| 353.12 | detail | confabulated | confabulated | 3 | -1.44 | -2.47 | 4.59 | 5 | P | P |
| 357.6 | theme | entangled | unresolved |  | 0.58 |  |  |  | . | . |
| 357.10 | detail | grounded | grounded | 2 | 11.67 | 9.24 | 4.51 | 5 | P | P |
| 357.13 | theme | grounded | seeded | 3 | 2.54 | 7.77 | -2.19 | 6 | P | P |
| 361.0 | theme | grounded | grounded | 2 | 16.82 | 13.59 | 11.81 | 11 | P | P |
| 361.1 | theme | grounded | seeded | 3 | 4.91 | 10.39 | -0.45 | 9 | P | P |
| 361.10 | detail | confabulated | confabulated | 3 | -0.09 | 1.21 | 0.50 | 6 | P | P |
| 365.3 | detail | confabulated | confabulated | 3 | 0.13 | 2.49 | -1.03 | 5 | P | P |
| 365.4 | theme | grounded | seeded | 3 | 3.64 | 9.27 | 0.23 | 7 | P | P |
| 365.15 | theme | confabulated | seeded | 3 | 0.31 | 4.12 | -0.84 | 6 | P | P |
| 369.0 | theme | grounded | unresolved |  | 11.60 | 15.54 | 1.46 | 6 | P | . |
| 369.4 | theme | confabulated | unresolved |  | 0.00 | 1.46 | -0.15 | 8 | . | P |
| 369.8 | detail | confabulated | unresolved |  | 0.50 | 2.61 | 0.09 | 5 | P | . |
| 373.0 | theme | entangled | seeded | 3 | 0.94 | 2.96 | 0.03 | 5 | P | P |
| 373.2 | detail | confabulated | confabulated | 3 | -0.02 | 1.29 | 0.18 | 7 | P | P |
| 373.3 | theme | grounded | confabulated | 3 | 1.12 | 3.31 | -0.06 | 5 | P | P |
| 377.2 | theme | grounded | seeded | 3 | 3.38 | 16.34 | -9.70 | 5 | P | P |
| 377.5 | detail | confabulated | confabulated | 3 | -0.08 | 6.65 | -2.96 | 7 | P | P |
| 377.6 | theme | confabulated | confabulated | 3 | -0.14 | 1.21 | 0.19 | 6 | . | P |
| 381.5 | theme | entangled | unresolved |  | 0.14 | 3.13 | -1.23 | 5 | . | P |
| 381.6 | detail | grounded | seeded | 3 | 10.15 | 14.52 | -1.14 | 7 | P | P |
| 381.7 | theme | confabulated | seeded | 3 | 0.13 | 1.75 | -0.30 | 5 | P | P |
| 385.0 | theme | grounded | seeded | 3 | 30.64 | 47.92 | -3.23 | 9 | P | P |
| 385.5 | theme | confabulated | confabulated | 3 | -0.01 | 2.19 | -0.37 | 6 | P | P |
| 385.6 | detail | confabulated | unresolved |  | 0.41 |  |  |  | . | . |
| 389.0 | theme | confabulated | entangled | 2 | 0.46 | -0.92 | 4.25 | 12 | P | P |
| 389.2 | theme | grounded | entangled | 2 | 3.11 | 1.56 | 4.34 | 6 | P | P |
| 389.9 | detail | confabulated | entangled | 2 | 1.53 | 1.31 | 2.14 | 17 | P | P |
| 393.4 | theme | grounded | grounded | 2 | 2.46 | 2.45 | 2.25 | 6 | P | P |
| 393.11 | detail | confabulated | confabulated | 3 | -0.07 | 1.25 | 0.41 | 7 | P | P |
| 393.14 | theme | confabulated | seeded | 3 | 0.40 | 4.73 | -1.94 | 5 | P | P |
| 397.0 | theme | grounded | unresolved |  | 2.37 | 8.86 | -0.54 | 5 | P | . |
| 397.1 | detail | grounded | seeded | 3 | 2.08 | 2.77 | 1.35 | 8 | P | P |
| 397.2 | theme | grounded | unresolved |  | 3.04 | 2.76 | 2.84 | 7 | . | P |
