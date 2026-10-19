# Cycle-consistency verdict report

run `llama70b-l4-fineweb-atoms`  claims 298  k 5

## Verdict distribution

| verdict | n | % |
|---|---|---|
| grounded | 18 | 6.0 |
| entangled | 12 | 4.0 |
| seeded | 93 | 31.2 |
| confabulated | 90 | 30.2 |
| unresolved | 85 | 28.5 |

unresolved by reason: validity 38, power 25, no text-passing decoys 17, claim span unlocated (fc) 5

## Gate flow (spec Step 7)

- 298 claims -> 85 unresolved (pre-checks), 213 reached the gates
- Gate 1 (specificity): 30 pass -> Gate 2, 183 fail -> Gate 3
- Gate 2 (ownership): 18 grounded, 12 entangled
- Gate 3 (shared signal): 93 seeded, 90 confabulated

## Atom -> cycle verdict shift

rows = published atom verdict, columns = cycle verdict

| atom \ cycle | grounded | entangled | seeded | confabulated | unresolved |
|---|---|---|---|---|---|
| confabulated | 0 | 0 | 6 | 72 | 18 |
| entangled | 0 | 6 | 51 | 15 | 31 |
| grounded | 18 | 6 | 36 | 3 | 36 |

## Pre-checks + grounding

- power passed 248/298, validity passed 225/298
- grounding estimate (M_BB - med D*, Gate-1 passers only (grounded+entangled; M_BB > 0), n=30): median 5.1757, p10 2.4827, p90 10.6681

## kappa_0 re-fit (Gate 1 null on THIS run)

- calibration kappa_0 = 1.1 (applied)
- re-fit on 158 survivor D* families = 1.0  **DIVERGES > 0.1**
- decoy-vs-decoy exceedance rate at the calibration kappa_0 = 0.0435 (target <= 0.05, whole run incl. in-sample)
- **out-of-sample** (75 activations disjoint from the 25 calibration ones, 112 families): re-fit kappa_0 = 1.0, FP rate at calibration kappa_0 = 0.0426 -> TRANSFERS (<= 0.05)
- _This re-fit is IN-SAMPLE when the run activations are the calibration split (e.g. the S4 small-N run reuses the 25 calibration activations): the LODO families and grid are then identical to calib_cycle.py's, so the FP rate is a consistency check, NOT out-of-sample validation. Genuine out-of-sample validation of kappa_0 requires a run on activations disjoint from calibration (S5's 375 new activations). WARNING: kappa_0_refit disagrees with the applied calibration kappa_0 by > 0.1 - the applied kappa_0 is stale for this set._

## Per-claim

| idx.ci | type | atom | cycle | gate | M_BB | gate1_bar | ground_est | surv | pw | vd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | theme | grounded | seeded | 3 | 6.96 | 12.92 | 0.18 | 7 | P | P |
| 0.3 | theme | confabulated | confabulated | 3 | -0.02 | 1.07 | 0.22 | 6 | P | P |
| 0.10 | detail | grounded | grounded | 2 | 5.33 | 2.76 | 4.07 | 7 | P | P |
| 4.0 | theme | entangled | seeded | 3 | 0.24 | 3.44 | -0.91 | 6 | P | P |
| 4.4 | theme | confabulated | confabulated | 3 | 0.04 | 1.30 | -0.09 | 5 | P | P |
| 4.10 | detail | grounded | unresolved |  | 1.69 | 4.16 | -0.08 | 5 | P | . |
| 8.0 | theme | grounded | unresolved |  | 7.65 |  |  |  | . | . |
| 8.1 | theme | grounded | seeded | 3 | 5.20 | 7.50 | 3.60 | 9 | P | P |
| 8.10 | detail | confabulated | confabulated | 3 | -0.75 | 0.48 | 0.80 | 7 | P | P |
| 12.0 | theme | confabulated | confabulated | 3 | -0.50 | 0.33 | 0.99 | 6 | P | P |
| 12.4 | theme | confabulated | seeded | 3 | 0.06 | 2.48 | -0.40 | 16 | P | P |
| 12.7 | detail | grounded | unresolved |  | 1.78 | 4.55 | -0.22 | 9 | . | P |
| 16.2 | detail | entangled | unresolved |  | 7.08 |  |  | 0 | . | . |
| 16.3 | theme | entangled | unresolved |  | 0.19 |  |  | 0 | . | . |
| 16.6 | theme | entangled | unresolved |  | 0.08 | 3.13 | -0.20 | 5 | P | . |
| 20.0 | theme | entangled | seeded | 3 | 0.71 | 1.99 | 0.54 | 6 | P | P |
| 20.7 | detail | grounded | unresolved |  | 4.18 | 3.91 | 3.69 | 4 | P | . |
| 20.12 | theme | confabulated | confabulated | 3 | -0.45 | 1.07 | -0.20 | 5 | P | P |
| 24.2 | theme | confabulated | seeded | 3 | 0.02 | 1.33 | -0.02 | 7 | P | P |
| 24.5 | detail | confabulated | confabulated | 3 | -0.04 | 2.39 | -0.26 | 6 | P | P |
| 24.9 | theme | entangled | confabulated | 3 | 0.22 | 1.36 | 0.37 | 8 | P | P |
| 28.0 | theme | grounded | unresolved |  | 2.70 |  |  | 0 | . | . |
| 28.5 | theme | confabulated | confabulated | 3 | -0.27 | 1.01 | 0.14 | 9 | P | P |
| 28.9 | detail | entangled | seeded | 3 | 1.33 | 2.81 | 0.79 | 7 | P | P |
| 32.2 | theme | entangled | seeded | 3 | 0.86 | 2.91 | 0.64 | 6 | P | P |
| 32.3 | theme | confabulated | confabulated | 3 | 0.00 | 1.35 | 0.13 | 8 | P | P |
| 32.4 | detail | confabulated | confabulated | 3 | 0.04 | 1.67 | 0.22 | 10 | P | P |
| 36.0 | theme | grounded | unresolved |  | 19.40 | 4.15 | 17.64 | 3 | P | . |
| 36.3 | theme | confabulated | confabulated | 3 | -0.10 | 1.00 | 0.76 | 5 | P | P |
| 36.6 | detail | entangled | seeded | 3 | 0.86 | 1.83 | 0.75 | 7 | P | P |
| 40.3 | theme | confabulated | confabulated | 3 | -0.02 | 5.69 | 1.29 | 8 | P | P |
| 40.8 | detail | grounded | unresolved |  | 11.10 | 12.51 | 2.21 | 20 | . | P |
| 40.9 | theme | entangled | unresolved |  | 0.76 | 2.21 | 0.42 | 4 | P | . |
| 44.2 | theme | confabulated | confabulated | 3 | -1.13 | -0.47 | 1.04 | 5 | P | P |
| 44.4 | detail | entangled | confabulated | 3 | 0.04 | 1.25 | 0.54 | 5 | P | P |
| 44.10 | theme | entangled | seeded | 3 | 0.91 | 3.15 | 0.01 | 6 | P | P |
| 48.2 | theme | entangled | unresolved |  | 0.42 | 3.96 | -0.38 | 6 | . | . |
| 48.6 | theme | entangled | seeded | 3 | 0.48 | 4.67 | -1.28 | 6 | P | P |
| 52.1 | theme | confabulated | confabulated | 3 | -0.06 | 2.89 | -0.16 | 7 | P | P |
| 52.2 | theme | entangled | seeded | 3 | 0.56 | 0.83 | 3.26 | 5 | P | P |
| 52.5 | detail | confabulated | confabulated | 3 | -0.28 | 2.79 | -0.37 | 9 | P | P |
| 56.2 | theme | grounded | seeded | 3 | 1.42 | 6.89 | -1.36 | 5 | P | P |
| 56.4 | detail | confabulated | unresolved |  | -0.01 | 1.81 | -0.22 | 5 | P | . |
| 56.7 | theme | entangled | confabulated | 3 | 0.01 | 1.26 | -0.13 | 5 | P | P |
| 61.5 | theme | confabulated | confabulated | 3 | -0.27 | -0.97 | 6.78 | 6 | P | P |
| 61.8 | theme | entangled | seeded | 3 | 2.66 | 18.53 | -4.22 | 5 | P | P |
| 61.11 | detail | entangled | entangled | 2 | 16.17 | 9.36 | 10.48 | 5 | P | P |
| 65.0 | detail | grounded | unresolved |  | 10.73 | 4.55 | 10.45 | 5 | P | . |
| 65.1 | theme | confabulated | unresolved |  | -0.00 | 3.50 | -1.48 | 3 | P | . |
| 65.2 | detail | grounded | unresolved |  | 38.45 |  |  | 0 | . | . |
| 69.4 | theme | grounded | unresolved |  | 15.06 | 11.57 | 4.59 | 1 | P | . |
| 69.11 | detail | grounded | unresolved |  | 3.02 | 7.64 | 0.05 | 5 | P | . |
| 69.13 | theme | confabulated | unresolved |  | 0.01 | -3.96 | 5.07 | 1 | P | . |
| 73.1 | theme | confabulated | confabulated | 3 | -0.46 | 0.36 | 0.79 | 5 | P | P |
| 73.6 | theme | confabulated | unresolved |  | -0.16 |  |  |  | . | . |
| 73.8 | detail | entangled | seeded | 3 | 0.40 | 3.04 | -0.30 | 6 | P | P |
| 77.1 | theme | entangled | unresolved |  | 0.85 | 0.30 | 2.29 | 4 | P | . |
| 77.10 | detail | grounded | unresolved |  | 3.16 |  |  |  | . | . |
| 77.11 | theme | confabulated | unresolved |  | -0.29 |  |  |  | . | . |
| 81.3 | theme | entangled | unresolved |  | 0.08 | 3.16 | 0.12 | 5 | P | . |
| 81.8 | detail | confabulated | unresolved |  | -0.13 | 2.88 | -0.38 | 6 | P | . |
| 81.10 | theme | confabulated | confabulated | 3 | -0.31 | 4.95 | -1.29 | 10 | P | P |
| 85.1 | theme | entangled | unresolved |  | 0.79 | 8.17 | -0.58 | 5 | P | . |
| 85.5 | detail | confabulated | confabulated | 3 | -0.29 | -0.64 | 2.60 | 8 | P | P |
| 85.8 | theme | grounded | seeded | 3 | 16.68 | 19.18 | 10.07 | 5 | P | P |
| 89.0 | theme | entangled | confabulated | 3 | 0.09 | 4.26 | -1.40 | 7 | P | P |
| 89.2 | theme | grounded | seeded | 3 | 1.09 | 7.28 | -1.80 | 9 | P | P |
| 89.3 | detail | entangled | confabulated | 3 | 0.07 | 2.58 | 0.26 | 8 | P | P |
| 93.3 | theme | confabulated | confabulated | 3 | 0.04 | 0.16 | 3.60 | 8 | P | P |
| 93.4 | theme | entangled | confabulated | 3 | 0.30 | 2.37 | 0.49 | 6 | P | P |
| 93.8 | detail | confabulated | confabulated | 3 | -0.00 | 1.83 | -0.28 | 10 | P | P |
| 97.1 | theme | entangled | seeded | 3 | 1.09 | 1.44 | 1.29 | 7 | P | P |
| 97.4 | theme | entangled | seeded | 3 | 2.15 | 5.18 | -0.55 | 6 | P | P |
| 97.10 | detail | confabulated | seeded | 3 | 0.18 | 0.82 | 0.86 | 7 | P | P |
| 101.3 | theme | grounded | unresolved |  | 4.99 | 10.33 | 0.98 | 6 | P | . |
| 101.11 | detail | entangled | confabulated | 3 | 3.92 | 6.45 | 3.86 | 9 | P | P |
| 101.13 | theme | entangled | seeded | 3 | 0.13 | 1.58 | 0.06 | 5 | P | P |
| 105.2 | theme | grounded | unresolved |  | 3.25 | 2.78 | 3.17 | 5 | P | . |
| 105.4 | detail | entangled | seeded | 3 | 0.00 | 1.30 | 0.19 | 17 | P | P |
| 105.8 | theme | entangled | unresolved |  |  |  |  |  | . | . |
| 109.8 | detail | confabulated | unresolved |  | -0.00 | 1.22 | -0.07 | 2 | P | . |
| 109.20 | theme | entangled | unresolved |  | 0.15 |  |  | 0 | . | . |
| 109.62 | theme | entangled | unresolved |  |  |  |  |  | . | . |
| 113.5 | detail | confabulated | confabulated | 3 | 0.15 | 1.40 | 0.33 | 14 | P | P |
| 113.8 | theme | confabulated | confabulated | 3 | -0.09 | 1.11 | 0.26 | 5 | P | P |
| 113.9 | theme | entangled | confabulated | 3 | 0.04 | 2.29 | -0.36 | 9 | P | P |
| 117.0 | theme | grounded | grounded | 2 | 2.69 | 2.51 | 1.73 | 5 | P | P |
| 117.5 | theme | grounded | entangled | 2 | 6.09 | 4.15 | 4.38 | 6 | P | P |
| 117.10 | detail | grounded | entangled | 2 | 3.25 | 2.13 | 2.90 | 5 | P | P |
| 121.1 | theme | entangled | seeded | 3 | 2.65 | 4.52 | 1.24 | 5 | P | P |
| 121.2 | detail | entangled | seeded | 3 | 0.65 | 13.56 | -8.08 | 6 | P | P |
| 121.4 | theme | entangled | confabulated | 3 | 0.15 | 1.80 | 0.12 | 8 | P | P |
| 125.0 | theme | confabulated | confabulated | 3 | -0.04 | 1.19 | 0.85 | 6 | P | P |
| 125.1 | theme | grounded | seeded | 3 | 1.53 | 3.02 | 1.59 | 7 | P | P |
| 125.5 | detail | confabulated | confabulated | 3 | -0.27 | 1.20 | -0.28 | 6 | P | P |
| 129.0 | theme | grounded | unresolved |  | 16.00 |  |  |  | . | . |
| 129.2 | theme | grounded | seeded | 3 | 1.32 | 6.47 | -2.15 | 5 | P | P |
| 129.12 | detail | confabulated | confabulated | 3 | -1.09 | 0.25 | 0.19 | 7 | P | P |
| 133.1 | theme | entangled | entangled | 2 | 0.62 | -0.62 | 2.95 | 7 | P | P |
| 133.6 | detail | entangled | seeded | 3 | 0.19 | 1.30 | 0.34 | 6 | P | P |
| 133.8 | theme | entangled | seeded | 3 | 0.54 | 1.49 | 0.37 | 5 | P | P |
| 137.0 | theme | grounded | unresolved |  | 4.28 | 7.41 | 0.64 | 22 | . | P |
| 137.6 | detail | grounded | seeded | 3 | 1.14 | 5.60 | -0.50 | 9 | P | P |
| 137.7 | theme | grounded | grounded | 2 | 8.50 | 4.69 | 6.52 | 7 | P | P |
| 141.1 | theme | confabulated | confabulated | 3 | -0.12 | 1.22 | -0.11 | 5 | P | P |
| 141.9 | detail | grounded | grounded | 2 | 10.48 | 8.99 | 3.74 | 8 | P | P |
| 141.17 | theme | grounded | seeded | 3 | 1.16 | 5.63 | -2.74 | 5 | P | P |
| 145.1 | theme | grounded | unresolved |  | 7.65 | 1.01 | 10.99 | 3 | P | . |
| 145.2 | theme | entangled | seeded | 3 | 1.36 | 3.99 | 0.46 | 6 | P | P |
| 145.9 | detail | grounded | confabulated | 3 | 1.45 | 1.63 | 1.53 | 7 | P | P |
| 149.4 | theme | entangled | unresolved |  | 0.65 | 3.13 | -0.48 | 5 | P | . |
| 149.7 | theme | entangled | unresolved |  | 0.11 | 1.11 | 0.36 | 5 | . | P |
| 153.1 | theme | confabulated | confabulated | 3 | -0.31 | 1.39 | 0.20 | 7 | P | P |
| 153.2 | theme | grounded | unresolved |  | 3.55 |  |  | 0 | . | . |
| 153.9 | detail | grounded | seeded | 3 | 1.14 | 6.21 | -1.60 | 5 | P | P |
| 157.6 | theme | entangled | unresolved |  | 0.09 |  |  |  | . | . |
| 157.10 | theme | entangled | unresolved |  | 0.15 |  |  |  | . | . |
| 157.12 | detail | grounded | grounded | 2 | 11.39 | 8.81 | 7.81 | 8 | P | P |
| 161.0 | theme | grounded | seeded | 3 | 6.71 | 7.94 | 4.47 | 7 | P | P |
| 161.1 | theme | entangled | seeded | 3 | 1.92 | 3.49 | 0.26 | 5 | P | P |
| 161.2 | detail | entangled | seeded | 3 | 0.16 | 7.03 | -3.28 | 6 | P | P |
| 165.1 | theme | grounded | seeded | 3 | 6.43 | 14.86 | -3.75 | 5 | P | P |
| 165.5 | detail | confabulated | confabulated | 3 | -0.05 | 2.61 | 0.30 | 10 | P | P |
| 165.8 | theme | confabulated | confabulated | 3 | -0.23 | 1.55 | -0.25 | 8 | P | P |
| 169.0 | theme | grounded | grounded | 2 | 31.30 | 16.19 | 20.97 | 6 | P | P |
| 169.2 | detail | entangled | seeded | 3 | 0.85 | 4.59 | -2.01 | 5 | P | P |
| 169.5 | theme | entangled | seeded | 3 | 0.31 | 1.96 | 0.31 | 6 | P | P |
| 173.0 | theme | grounded | seeded | 3 | 6.05 | 11.52 | 0.99 | 7 | P | P |
| 173.9 | detail | entangled | unresolved |  | 0.92 |  |  |  | . | . |
| 173.11 | theme | confabulated | confabulated | 3 | -0.42 | 1.06 | 0.47 | 6 | P | P |
| 177.0 | theme | entangled | unresolved |  | 0.26 | -5.19 | 19.74 | 4 | P | . |
| 177.1 | theme | grounded | seeded | 3 | 6.04 | 9.73 | 2.60 | 5 | P | P |
| 177.2 | detail | grounded | grounded | 2 | 2.79 | 2.49 | 2.18 | 6 | P | P |
| 181.9 | theme | confabulated | unresolved |  |  |  |  |  | . | . |
| 181.12 | theme | confabulated | unresolved |  |  |  |  |  | . | . |
| 181.15 | detail | confabulated | confabulated | 3 | -0.01 | 1.12 | 0.07 | 6 | P | P |
| 185.0 | theme | confabulated | confabulated | 3 | -0.22 | -1.27 | 2.61 | 5 | P | P |
| 185.6 | detail | entangled | seeded | 3 | 0.73 | 1.58 | 0.56 | 5 | P | P |
| 185.12 | theme | grounded | seeded | 3 | 2.80 | 7.75 | -0.07 | 5 | P | P |
| 189.2 | theme | entangled | seeded | 3 | 1.61 | 8.75 | -4.73 | 5 | P | P |
| 189.9 | detail | grounded | unresolved |  | 1.93 | 5.67 | 1.19 | 5 | P | . |
| 189.11 | theme | confabulated | seeded | 3 | 0.13 | 0.32 | 1.57 | 6 | P | P |
| 193.2 | theme | confabulated | confabulated | 3 | -1.03 | 1.13 | -0.80 | 6 | P | P |
| 193.4 | theme | grounded | grounded | 2 | 4.81 | 2.15 | 4.46 | 7 | P | P |
| 193.8 | detail | grounded | unresolved |  | 5.25 | 14.33 | -1.35 | 5 | P | . |
| 197.1 | theme | entangled | unresolved |  | 0.26 |  |  |  | . | . |
| 197.3 | detail | grounded | seeded | 3 | 1.04 | 2.65 | 0.41 | 6 | P | P |
| 197.10 | theme | entangled | confabulated | 3 | 0.06 | 1.28 | 0.20 | 6 | P | P |
| 201.2 | theme | grounded | seeded | 3 | 8.95 | 11.31 | 9.31 | 9 | P | P |
| 201.3 | theme | grounded | seeded | 3 | 0.85 | 3.81 | -0.77 | 6 | P | P |
| 201.6 | detail | grounded | grounded | 2 | 6.42 | 3.70 | 10.53 | 7 | P | P |
| 205.1 | theme | confabulated | unresolved |  | -0.18 | 1.26 | -0.17 | 5 | . | . |
| 205.4 | detail | entangled | unresolved |  | 0.00 | 1.95 | -0.56 | 4 | P | . |
| 205.6 | theme | grounded | unresolved |  | 1.76 |  |  |  | . | . |
| 209.0 | theme | confabulated | confabulated | 3 | -1.04 | -3.96 | 5.45 | 10 | P | P |
| 209.3 | detail | grounded | unresolved |  | 10.47 |  |  |  | . | . |
| 209.5 | theme | entangled | unresolved |  | 0.00 | 1.11 | 0.14 | 7 | . | P |
| 213.1 | theme | grounded | seeded | 3 | 4.31 | 4.87 | 2.25 | 6 | P | P |
| 213.2 | theme | confabulated | confabulated | 3 | -1.31 | -0.85 | 3.25 | 9 | P | P |
| 213.4 | detail | grounded | grounded | 2 | 22.58 | 18.57 | 11.94 | 6 | P | P |
| 217.0 | theme | entangled | seeded | 3 | 1.30 | 3.34 | -0.26 | 9 | P | P |
| 217.2 | theme | grounded | seeded | 3 | 1.15 | 2.48 | 1.24 | 5 | P | P |
| 217.10 | detail | grounded | unresolved |  | 6.62 | 5.44 | 5.52 | 5 | P | . |
| 221.0 | theme | entangled | unresolved |  | 0.82 |  |  |  | . | . |
| 221.1 | theme | grounded | entangled | 2 | 4.99 | 1.27 | 6.13 | 6 | P | P |
| 221.4 | detail | confabulated | confabulated | 3 | -0.10 | 1.03 | 0.09 | 7 | P | P |
| 225.0 | theme | grounded | unresolved |  | 14.02 | 15.38 | 3.93 | 9 | . | P |
| 225.10 | detail | confabulated | confabulated | 3 | -0.75 | 0.91 | 0.07 | 5 | . | P |
| 225.13 | theme | grounded | grounded | 2 | 5.31 | 5.21 | 8.20 | 9 | P | P |
| 229.2 | theme | confabulated | confabulated | 3 | -0.59 | -0.00 | 1.15 | 5 | P | P |
| 229.3 | theme | confabulated | confabulated | 3 | -1.15 | 3.05 | -0.18 | 6 | P | P |
| 229.5 | detail | confabulated | confabulated | 3 | -1.38 | 0.13 | 0.29 | 7 | . | P |
| 233.1 | theme | entangled | seeded | 3 | 1.11 | 2.11 | 0.82 | 7 | P | P |
| 233.2 | theme | grounded | unresolved |  | 4.54 |  |  | 0 | . | . |
| 233.6 | detail | confabulated | confabulated | 3 | -1.01 | -0.08 | 1.12 | 5 | P | P |
| 237.1 | detail | confabulated | confabulated | 3 | -0.00 | 0.99 | 0.47 | 5 | P | P |
| 237.2 | detail | entangled | unresolved |  | 0.58 |  |  |  | . | . |
| 237.3 | theme | entangled | unresolved |  | 1.50 | 4.48 | 0.85 | 5 | P | . |
| 241.1 | theme | grounded | unresolved |  |  |  |  |  | . | . |
| 241.2 | theme | confabulated | confabulated | 3 | -0.17 | 1.01 | 0.56 | 7 | P | P |
| 241.12 | detail | grounded | seeded | 3 | 6.03 | 8.32 | -0.55 | 5 | P | P |
| 245.1 | theme | confabulated | confabulated | 3 | -0.34 | 0.78 | 1.47 | 9 | P | P |
| 245.5 | theme | grounded | seeded | 3 | 0.84 | 3.33 | -0.32 | 6 | P | P |
| 245.7 | detail | confabulated | confabulated | 3 | -4.05 | 2.47 | -2.84 | 6 | P | P |
| 249.2 | theme | entangled | seeded | 3 | 0.20 | 3.16 | -0.43 | 11 | P | P |
| 249.6 | theme | confabulated | unresolved |  | -0.06 | 1.15 | -0.07 | 4 | P | . |
| 249.8 | detail | grounded | unresolved |  | 10.20 |  |  |  | . | . |
| 253.1 | theme | grounded | seeded | 3 | 2.28 | 3.24 | 1.03 | 6 | P | P |
| 253.3 | theme | confabulated | confabulated | 3 | -0.74 | -0.24 | 1.13 | 5 | P | P |
| 253.12 | detail | confabulated | confabulated | 3 | -0.03 | 0.44 | 3.74 | 15 | P | P |
| 257.2 | theme | grounded | grounded | 2 | 4.48 | 2.99 | 5.29 | 11 | P | P |
| 257.4 | theme | grounded | grounded | 2 | 9.34 | 5.48 | 5.93 | 10 | P | P |
| 257.7 | detail | entangled | seeded | 3 | 0.00 | 1.01 | 0.54 | 6 | P | P |
| 261.1 | theme | grounded | unresolved |  | 1.75 |  |  | 0 | . | . |
| 261.5 | theme | confabulated | confabulated | 3 | 0.05 | 1.17 | 0.95 | 11 | P | P |
| 261.9 | detail | confabulated | confabulated | 3 | -0.45 | 1.00 | 0.01 | 5 | P | P |
| 265.2 | theme | confabulated | confabulated | 3 | 0.01 | 1.19 | 0.14 | 5 | P | P |
| 265.3 | detail | entangled | unresolved |  | 0.23 |  |  | 0 | . | . |
| 265.4 | theme | entangled | seeded | 3 | 0.50 | 2.50 | -0.00 | 9 | P | P |
| 269.2 | theme | grounded | grounded | 2 | 4.63 | 2.24 | 5.06 | 5 | P | P |
| 269.7 | theme | grounded | seeded | 3 | 1.05 | 3.86 | -0.68 | 9 | P | P |
| 269.9 | detail | confabulated | seeded | 3 | 0.01 | 1.37 | 0.22 | 8 | P | P |
| 273.0 | theme | confabulated | confabulated | 3 | -0.02 | 1.33 | -0.00 | 5 | P | P |
| 273.4 | theme | grounded | unresolved |  | 6.78 | 7.63 | 0.25 | 1 | P | . |
| 273.6 | detail | entangled | seeded | 3 | 0.77 | 2.63 | -0.20 | 8 | P | P |
| 277.1 | theme | confabulated | confabulated | 3 | -1.63 | -0.56 | 1.08 | 5 | P | P |
| 277.2 | theme | entangled | entangled | 2 | 2.85 | 2.06 | 2.42 | 7 | P | P |
| 277.9 | detail | confabulated | confabulated | 3 | -0.09 | 1.44 | 0.03 | 5 | P | P |
| 281.4 | theme | grounded | seeded | 3 | 4.04 | 9.10 | 0.99 | 6 | P | P |
| 281.5 | theme | grounded | seeded | 3 | 8.03 | 20.16 | -6.08 | 8 | P | P |
| 281.10 | detail | entangled | seeded | 3 | 0.51 | 4.82 | -2.37 | 5 | P | P |
| 285.2 | theme | entangled | unresolved |  |  |  |  |  | . | . |
| 285.3 | theme | grounded | seeded | 3 | 2.35 | 5.75 | 0.41 | 7 | P | P |
| 285.8 | detail | entangled | confabulated | 3 | 0.04 | 1.05 | 0.28 | 7 | P | P |
| 289.0 | theme | grounded | grounded | 2 | 15.62 | 12.73 | 6.17 | 6 | P | P |
| 289.6 | detail | entangled | seeded | 3 | 1.16 | 3.17 | -0.39 | 6 | P | P |
| 289.7 | theme | entangled | entangled | 2 | 4.59 | 3.99 | 3.41 | 8 | P | P |
| 293.3 | theme | confabulated | confabulated | 3 | -0.33 | -0.01 | 2.62 | 6 | P | P |
| 293.4 | detail | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 293.5 | theme | entangled | seeded | 3 | 0.14 | 1.33 | 0.04 | 6 | P | P |
| 297.2 | theme | grounded | seeded | 3 | 2.49 | 8.92 | -0.93 | 7 | P | P |
| 297.4 | theme | confabulated | confabulated | 3 | -0.19 | 1.71 | 0.44 | 10 | P | P |
| 297.6 | detail | entangled | seeded | 3 | 0.51 | 1.53 | 0.53 | 5 | P | P |
| 301.0 | theme | grounded | unresolved |  | 9.34 |  |  | 0 | . | . |
| 301.5 | detail | grounded | unresolved |  | 2.69 | 4.94 | -0.21 | 3 | P | . |
| 301.6 | theme | entangled | seeded | 3 | 0.43 | 1.69 | 3.52 | 8 | P | P |
| 305.0 | theme | entangled | seeded | 3 | 0.73 | 1.56 | 1.27 | 10 | P | P |
| 305.2 | theme | entangled | seeded | 3 | 0.36 | 3.75 | 0.27 | 6 | P | P |
| 305.10 | detail | grounded | unresolved |  | 18.26 | 18.97 | 5.61 | 5 | . | P |
| 309.1 | theme | confabulated | confabulated | 3 | 0.14 | 1.88 | 0.67 | 6 | P | P |
| 309.9 | detail | entangled | seeded | 3 | 0.12 | 1.30 | 2.68 | 5 | P | P |
| 309.12 | theme | confabulated | confabulated | 3 | -0.20 | 1.63 | -0.03 | 6 | P | P |
| 313.0 | theme | confabulated | confabulated | 3 | -0.17 | 2.97 | 1.49 | 11 | . | P |
| 313.3 | theme | confabulated | confabulated | 3 | -0.30 | 3.36 | -0.48 | 5 | P | P |
| 313.8 | detail | confabulated | confabulated | 3 | -0.04 | 1.41 | 0.07 | 6 | P | P |
| 317.1 | theme | confabulated | confabulated | 3 | -1.19 | -0.13 | 1.41 | 5 | P | P |
| 317.2 | theme | grounded | seeded | 3 | 1.99 | 6.84 | -0.49 | 5 | P | P |
| 317.5 | detail | entangled | confabulated | 3 | 0.52 | 4.00 | -1.46 | 6 | P | P |
| 321.3 | theme | entangled | seeded | 3 | 1.75 | 6.02 | 0.41 | 6 | P | P |
| 321.7 | detail | entangled | seeded | 3 | 1.85 | 6.59 | -1.90 | 5 | P | P |
| 321.10 | theme | entangled | seeded | 3 | 2.42 | 8.64 | -2.82 | 6 | P | P |
| 325.6 | theme | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 325.8 | detail | entangled | unresolved |  | 1.69 | 4.21 | 0.67 | 6 | . | P |
| 325.11 | theme | confabulated | unresolved |  | -0.33 | -0.24 | 4.43 | 2 | P | . |
| 329.0 | theme | grounded | confabulated | 3 | 1.38 | 1.78 | 2.54 | 7 | P | P |
| 329.10 | detail | grounded | seeded | 3 | 2.42 | 3.50 | 1.64 | 5 | P | P |
| 329.13 | theme | confabulated | confabulated | 3 | -0.15 | 1.75 | -0.24 | 7 | P | P |
| 333.2 | theme | confabulated | confabulated | 3 | -0.21 | 1.47 | -0.32 | 7 | P | P |
| 333.8 | theme | entangled | unresolved |  | 0.36 | 3.62 | -1.00 | 5 | P | . |
| 333.15 | detail | confabulated | unresolved |  | -0.28 | 1.66 | -0.35 | 4 | P | . |
| 337.3 | theme | entangled | seeded | 3 | 1.28 | 6.28 | -1.39 | 5 | P | P |
| 337.10 | theme | confabulated | seeded | 3 | 0.02 | 2.33 | -0.99 | 5 | P | P |
| 337.13 | detail | grounded | seeded | 3 | 8.99 | 9.95 | 3.64 | 7 | P | P |
| 341.1 | theme | entangled | seeded | 3 | 0.15 | 2.18 | -0.32 | 5 | P | P |
| 341.9 | theme | entangled | unresolved |  | 0.13 | 1.27 | 0.05 | 5 | . | P |
| 341.12 | detail | entangled | seeded | 3 | 0.22 | 2.22 | 0.07 | 7 | P | P |
| 345.2 | theme | entangled | seeded | 3 | 0.29 | 2.41 | -0.17 | 8 | P | P |
| 345.4 | theme | confabulated | unresolved |  | -0.14 | 1.83 | -0.16 | 5 | P | . |
| 345.9 | detail | grounded | grounded | 2 | 12.14 | 4.58 | 10.01 | 5 | P | P |
| 349.0 | theme | grounded | seeded | 3 | 1.49 | 5.55 | -1.48 | 6 | P | P |
| 349.4 | theme | entangled | seeded | 3 | 0.08 | 1.30 | -0.02 | 5 | P | P |
| 349.8 | detail | grounded | grounded | 2 | 7.32 | 2.11 | 8.49 | 7 | P | P |
| 353.0 | theme | confabulated | confabulated | 3 | -0.22 | 1.18 | 0.68 | 5 | P | P |
| 353.2 | theme | entangled | confabulated | 3 | 0.80 | 1.29 | 2.77 | 9 | P | P |
| 353.12 | detail | entangled | unresolved |  | 0.74 | 3.61 | -0.41 | 5 | P | . |
| 357.3 | theme | confabulated | confabulated | 3 | -0.26 | 0.77 | 1.01 | 5 | P | P |
| 357.4 | theme | grounded | unresolved |  | 1.14 | 3.25 | -0.27 | 5 | P | . |
| 357.10 | detail | entangled | unresolved |  | 0.13 | 2.38 | -0.05 | 10 | . | P |
| 361.5 | theme | entangled | seeded | 3 | 0.87 | 2.54 | 0.08 | 5 | P | P |
| 361.6 | theme | grounded | unresolved |  | 1.16 | 4.66 | 0.11 | 5 | P | . |
| 361.12 | detail | grounded | seeded | 3 | 4.30 | 8.48 | -1.82 | 7 | P | P |
| 365.0 | theme | grounded | seeded | 3 | 2.02 | 3.80 | 0.82 | 9 | P | P |
| 365.3 | theme | confabulated | confabulated | 3 | -0.14 | -0.03 | 1.72 | 5 | P | P |
| 365.10 | detail | entangled | seeded | 3 | 0.33 | 1.88 | 0.48 | 6 | P | P |
| 369.0 | theme | grounded | grounded | 2 | 8.93 | 5.99 | 6.17 | 9 | P | P |
| 369.3 | theme | confabulated | confabulated | 3 | 0.04 | 2.03 | -0.31 | 6 | P | P |
| 369.7 | detail | confabulated | confabulated | 3 | -0.02 | 0.49 | 0.72 | 5 | P | P |
| 373.0 | theme | grounded | seeded | 3 | 2.63 | 6.46 | 0.47 | 6 | P | P |
| 373.2 | theme | entangled | confabulated | 3 | 0.70 | 3.14 | 1.98 | 13 | P | P |
| 373.5 | detail | grounded | entangled | 2 | 4.88 | 4.60 | 3.54 | 6 | P | P |
| 377.5 | theme | grounded | unresolved |  | 9.28 | 12.99 | 0.89 | 6 | . | P |
| 377.6 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 377.10 | detail | confabulated | confabulated | 3 | -0.08 | 1.86 | -0.10 | 5 | P | P |
| 381.4 | theme | confabulated | confabulated | 3 | -0.21 | -0.09 | 1.40 | 5 | P | P |
| 381.6 | theme | entangled | seeded | 3 | 0.05 | 1.05 | 0.72 | 9 | P | P |
| 381.8 | detail | entangled | entangled | 2 | 1.75 | 0.95 | 2.49 | 8 | P | P |
| 385.3 | theme | entangled | confabulated | 3 | 2.59 | 15.39 | -4.58 | 13 | P | P |
| 385.5 | theme | entangled | unresolved |  | 0.42 | 1.93 | 0.39 | 7 | . | P |
| 385.8 | detail | grounded | entangled | 2 | 18.15 | 5.22 | 17.65 | 6 | P | P |
| 389.3 | theme | entangled | entangled | 2 | 1.16 | -0.40 | 5.01 | 5 | P | P |
| 389.6 | theme | grounded | confabulated | 3 | 3.36 | 12.33 | -6.65 | 6 | P | P |
| 389.12 | detail | confabulated | confabulated | 3 | -0.35 | 1.76 | -0.41 | 5 | P | P |
| 393.2 | theme | confabulated | confabulated | 3 | -0.38 | 1.56 | 0.03 | 5 | P | P |
| 393.3 | theme | grounded | seeded | 3 | 7.44 | 8.31 | 3.11 | 5 | P | P |
| 393.9 | detail | grounded | entangled | 2 | 5.05 | 4.38 | 3.39 | 7 | P | P |
| 397.1 | theme | grounded | unresolved |  | 9.00 | 11.04 | -0.47 | 3 | P | . |
| 397.4 | detail | entangled | seeded | 3 | 0.29 | 1.87 | -0.07 | 6 | P | P |
| 397.5 | theme | confabulated | unresolved |  | -0.77 | 1.01 | -0.32 | 4 | P | . |
