# Cycle-consistency verdict report

run `gemma27b-g4-fineweb-atoms`  claims 299  k 5

## Verdict distribution

| verdict | n | % |
|---|---|---|
| grounded | 12 | 4.0 |
| entangled | 5 | 1.7 |
| seeded | 87 | 29.1 |
| confabulated | 102 | 34.1 |
| unresolved | 93 | 31.1 |

unresolved by reason: validity 41, power 40, no text-passing decoys 10, claim span unlocated (fc) 2

## Gate flow (spec Step 7)

- 299 claims -> 93 unresolved (pre-checks), 206 reached the gates
- Gate 1 (specificity): 17 pass -> Gate 2, 189 fail -> Gate 3
- Gate 2 (ownership): 12 grounded, 5 entangled
- Gate 3 (shared signal): 87 seeded, 102 confabulated

## Atom -> cycle verdict shift

rows = published atom verdict, columns = cycle verdict

| atom \ cycle | grounded | entangled | seeded | confabulated | unresolved |
|---|---|---|---|---|---|
| confabulated | 0 | 3 | 25 | 84 | 39 |
| entangled | 0 | 2 | 29 | 10 | 26 |
| grounded | 12 | 0 | 33 | 8 | 28 |

## Pre-checks + grounding

- power passed 235/299, validity passed 228/299
- grounding estimate (M_BB - med D*, Gate-1 passers only (grounded+entangled; M_BB > 0), n=17): median 6.4402, p10 3.769, p90 12.4911

## kappa_0 re-fit (Gate 1 null on THIS run)

- calibration kappa_0 = 1.2 (applied)
- re-fit on 154 survivor D* families = 1.1
- decoy-vs-decoy exceedance rate at the calibration kappa_0 = 0.0468 (target <= 0.05, whole run incl. in-sample)
- **out-of-sample** (75 activations disjoint from the 25 calibration ones, 114 families): re-fit kappa_0 = 1.2, FP rate at calibration kappa_0 = 0.0479 -> TRANSFERS (<= 0.05)
- _This re-fit is IN-SAMPLE when the run activations are the calibration split (e.g. the S4 small-N run reuses the 25 calibration activations): the LODO families and grid are then identical to calib_cycle.py's, so the FP rate is a consistency check, NOT out-of-sample validation. Genuine out-of-sample validation of kappa_0 requires a run on activations disjoint from calibration (S5's 375 new activations). kappa_0_refit agrees with the applied calibration kappa_0._

## Per-claim

| idx.ci | type | atom | cycle | gate | M_BB | gate1_bar | ground_est | surv | pw | vd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 | theme | confabulated | confabulated | 3 | -1.54 | -1.09 | 4.59 | 8 | P | P |
| 0.6 | theme | confabulated | seeded | 3 | 0.03 | 1.57 | -0.13 | 8 | P | P |
| 0.11 | detail | entangled | seeded | 3 | 0.17 | 8.39 | -2.87 | 6 | P | P |
| 4.0 | theme | grounded | grounded | 2 | 4.13 | 4.10 | 3.44 | 5 | P | P |
| 4.3 | theme | confabulated | confabulated | 3 | -0.29 | 1.03 | 1.17 | 5 | P | P |
| 4.12 | detail | entangled | unresolved |  | 0.71 | 3.16 | -0.45 | 5 | P | . |
| 8.0 | theme | entangled | unresolved |  | 0.64 | 3.38 | -0.39 | 5 | . | . |
| 8.5 | detail | grounded | grounded | 2 | 10.17 | 2.73 | 13.75 | 5 | P | P |
| 8.8 | theme | entangled | unresolved |  | 0.77 | 1.83 | 0.77 | 5 | . | . |
| 12.0 | theme | entangled | unresolved |  | 0.45 | 4.36 | -1.60 | 5 | P | . |
| 12.5 | theme | confabulated | seeded | 3 | 0.03 | 2.10 | -0.34 | 6 | P | P |
| 12.9 | detail | entangled | unresolved |  | 0.94 | 2.60 | 0.07 | 6 | . | P |
| 16.5 | theme | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 16.9 | theme | confabulated | seeded | 3 | 0.17 | 2.28 | -0.52 | 6 | P | P |
| 16.14 | detail | confabulated | confabulated | 3 | -0.25 | 1.84 | -0.12 | 9 | P | P |
| 20.0 | theme | grounded | seeded | 3 | 1.65 | 3.47 | 0.79 | 6 | P | P |
| 20.6 | theme | confabulated | confabulated | 3 | 0.15 | 3.27 | 1.24 | 6 | P | P |
| 20.7 | detail | grounded | unresolved |  | 17.03 | 15.75 | 11.84 | 6 | . | P |
| 24.3 | theme | entangled | entangled | 2 | 2.25 | -0.22 | 5.83 | 7 | P | P |
| 24.5 | detail | grounded | grounded | 2 | 9.50 | 2.23 | 9.36 | 6 | P | P |
| 24.6 | theme | confabulated | confabulated | 3 | -0.06 | 1.14 | 0.56 | 10 | P | P |
| 28.0 | theme | grounded | unresolved |  | 2.53 | 10.70 | 2.24 | 5 | P | . |
| 28.1 | theme | grounded | grounded | 2 | 5.02 | 4.62 | 4.03 | 6 | P | P |
| 28.10 | detail | entangled | seeded | 3 | 4.41 | 6.63 | 0.58 | 5 | P | P |
| 32.3 | detail | confabulated | confabulated | 3 | -0.20 | 1.14 | 1.50 | 8 | P | P |
| 32.4 | theme | confabulated | confabulated | 3 | -0.23 | 1.51 | -0.05 | 10 | P | P |
| 32.8 | theme | confabulated | seeded | 3 | 0.00 | 1.41 | 0.22 | 5 | P | P |
| 36.1 | theme | confabulated | unresolved |  | 0.01 | 6.44 | -2.63 | 6 | P | . |
| 36.3 | theme | grounded | confabulated | 3 | 3.16 | 5.85 | 1.37 | 5 | P | P |
| 36.7 | detail | confabulated | confabulated | 3 | -0.62 | 0.59 | 0.93 | 5 | P | P |
| 40.4 | theme | entangled | unresolved |  | 0.00 | 1.45 | -0.25 | 1 | P | . |
| 40.6 | theme | confabulated | confabulated | 3 | -0.68 | 0.73 | 0.29 | 5 | P | P |
| 40.10 | detail | grounded | grounded | 2 | 19.76 | 11.15 | 11.65 | 6 | P | P |
| 44.1 | theme | entangled | seeded | 3 | 0.54 | 2.53 | 0.13 | 5 | P | P |
| 44.4 | theme | grounded | unresolved |  | 7.39 | 11.77 | 4.06 | 5 | P | . |
| 44.6 | detail | confabulated | entangled | 2 | 0.08 | -1.16 | 3.99 | 6 | P | P |
| 48.2 | theme | confabulated | confabulated | 3 | -1.76 | 0.78 | -1.13 | 5 | P | P |
| 48.8 | detail | entangled | confabulated | 3 | 0.03 | 1.88 | -0.21 | 8 | P | P |
| 48.13 | theme | confabulated | confabulated | 3 | -0.98 | -1.65 | 3.36 | 6 | P | P |
| 52.1 | theme | confabulated | entangled | 2 | 0.51 | -0.88 | 8.74 | 5 | P | P |
| 52.2 | theme | confabulated | confabulated | 3 | -0.43 | 1.31 | 0.71 | 7 | . | P |
| 52.17 | detail | confabulated | confabulated | 3 | -0.09 | 1.23 | -0.08 | 5 | P | P |
| 56.0 | theme | entangled | unresolved |  | 3.55 | 5.84 | -1.09 | 1 | P | . |
| 56.4 | detail | grounded | confabulated | 3 | 1.67 | 4.16 | -0.36 | 5 | P | P |
| 56.6 | detail | confabulated | confabulated | 3 | 0.11 | 5.28 | 2.45 | 22 | P | P |
| 61.8 | theme | confabulated | seeded | 3 | 0.01 | 1.58 | 2.22 | 6 | P | P |
| 61.11 | theme | confabulated | confabulated | 3 | -0.67 | 0.41 | 0.83 | 7 | P | P |
| 61.13 | detail | confabulated | confabulated | 3 | -1.78 | 1.24 | -1.42 | 10 | P | P |
| 65.3 | theme | confabulated | confabulated | 3 | -0.07 | 3.48 | -1.23 | 7 | P | P |
| 65.4 | theme | confabulated | confabulated | 3 | -0.64 | 0.69 | 0.41 | 8 | P | P |
| 65.5 | detail | entangled | seeded | 3 | 0.45 | 9.06 | -3.20 | 9 | P | P |
| 69.4 | theme | confabulated | unresolved |  | -0.00 | 1.52 | -0.17 | 2 | . | . |
| 69.6 | theme | confabulated | unresolved |  | 0.06 | 3.75 | -0.70 | 5 | P | . |
| 69.10 | detail | confabulated | confabulated | 3 | -0.05 | -0.29 | 2.05 | 5 | P | P |
| 73.1 | theme | entangled | seeded | 3 | 1.00 | 2.39 | 0.35 | 6 | P | P |
| 73.6 | theme | grounded | unresolved |  | 3.60 |  |  |  | . | . |
| 73.17 | detail | grounded | seeded | 3 | 4.29 | 9.61 | -1.07 | 5 | P | P |
| 77.0 | theme | entangled | seeded | 3 | 0.90 | 3.61 | -0.29 | 5 | P | P |
| 77.2 | theme | entangled | unresolved |  | 1.45 | 5.49 | 0.47 | 5 | P | . |
| 77.11 | detail | grounded | seeded | 3 | 3.19 | 9.61 | -0.16 | 5 | P | P |
| 81.2 | theme | confabulated | seeded | 3 | 0.34 | 2.37 | 0.00 | 7 | P | P |
| 81.6 | theme | confabulated | confabulated | 3 | 0.34 | 1.69 | 0.71 | 7 | P | P |
| 81.13 | detail | grounded | grounded | 2 | 6.79 | 4.79 | 5.07 | 5 | P | P |
| 85.3 | theme | entangled | seeded | 3 | 0.80 | 2.88 | 0.27 | 6 | P | P |
| 85.5 | theme | confabulated | seeded | 3 | 0.35 | 6.83 | -3.04 | 8 | P | P |
| 85.12 | detail | grounded | grounded | 2 | 8.92 | 7.68 | 5.15 | 7 | P | P |
| 89.0 | theme | confabulated | confabulated | 3 | -0.32 | -0.07 | 4.16 | 5 | P | P |
| 89.6 | theme | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 89.7 | detail | grounded | seeded | 3 | 11.56 | 17.39 | -0.80 | 5 | P | P |
| 93.4 | theme | confabulated | confabulated | 3 | -0.30 | 0.89 | 2.69 | 8 | P | P |
| 93.5 | theme | grounded | seeded | 3 | 4.58 | 11.68 | -1.50 | 11 | P | P |
| 93.17 | detail | grounded | grounded | 2 | 9.09 | 8.27 | 8.09 | 6 | P | P |
| 97.0 | theme | grounded | confabulated | 3 | 13.69 | 15.68 | 15.07 | 9 | P | P |
| 97.8 | theme | confabulated | confabulated | 3 | -0.37 | 1.30 | -0.27 | 5 | P | P |
| 97.16 | detail | entangled | entangled | 2 | 1.29 | -10.03 | 14.99 | 8 | P | P |
| 101.0 | theme | grounded | seeded | 3 | 7.38 | 10.23 | 2.26 | 6 | P | P |
| 101.7 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 101.11 | detail | grounded | seeded | 3 | 7.98 | 15.15 | -0.38 | 6 | P | P |
| 105.3 | theme | entangled | seeded | 3 | 0.94 | 3.18 | 1.75 | 7 | P | P |
| 105.8 | theme | entangled | unresolved |  | 0.05 | 5.07 | -3.82 | 1 | P | . |
| 105.15 | detail | entangled | confabulated | 3 | 3.04 | 6.18 | 0.53 | 10 | P | P |
| 109.0 | theme | confabulated | unresolved |  | 1.14 | 2.84 | 0.82 | 5 | P | . |
| 109.4 | theme | grounded | unresolved |  | 17.04 | 35.02 | -7.98 | 2 | P | . |
| 109.8 | detail | grounded | seeded | 3 | 5.40 | 6.26 | 4.50 | 6 | P | P |
| 113.7 | theme | confabulated | unresolved |  | 0.00 | 0.83 | 0.37 | 1 | P | . |
| 113.9 | theme | entangled | seeded | 3 | 0.48 | 1.37 | 0.94 | 6 | P | P |
| 113.14 | detail | confabulated | confabulated | 3 | -0.56 | -2.36 | 5.80 | 7 | P | P |
| 117.3 | theme | grounded | unresolved |  | 4.05 | 12.04 | -6.79 | 1 | P | . |
| 117.4 | theme | entangled | unresolved |  | 0.71 | 3.84 | 0.91 | 8 | . | P |
| 117.11 | detail | confabulated | seeded | 3 | 1.14 | 5.37 | -2.29 | 5 | P | P |
| 121.3 | theme | grounded | unresolved |  | 1.05 | 2.57 | 1.25 | 4 | P | . |
| 121.4 | theme | entangled | unresolved |  | 0.71 | 1.10 | 3.32 | 5 | P | . |
| 121.12 | detail | grounded | seeded | 3 | 2.43 | 4.13 | 0.67 | 8 | P | P |
| 125.0 | theme | confabulated | confabulated | 3 | 0.07 | 9.68 | -1.26 | 12 | P | P |
| 125.1 | theme | grounded | seeded | 3 | 1.31 | 3.24 | 0.48 | 5 | P | P |
| 125.15 | detail | confabulated | unresolved |  |  |  |  |  | . | . |
| 129.1 | theme | confabulated | unresolved |  | -0.20 | 0.72 | 1.71 | 5 | P | . |
| 129.4 | theme | grounded | seeded | 3 | 2.78 | 26.82 | -7.41 | 14 | P | P |
| 129.9 | detail | grounded | seeded | 3 | 6.20 | 7.65 | 3.56 | 6 | P | P |
| 133.3 | theme | confabulated | seeded | 3 | 0.40 | 3.52 | -0.19 | 9 | P | P |
| 133.4 | theme | entangled | unresolved |  | 0.00 | 1.13 | 0.09 | 2 | . | . |
| 133.9 | detail | entangled | confabulated | 3 | 0.00 | 1.89 | -0.26 | 6 | P | P |
| 137.0 | theme | entangled | seeded | 3 | 0.58 | 10.65 | 0.64 | 8 | P | P |
| 137.6 | theme | confabulated | seeded | 3 | 0.25 | 2.65 | 0.37 | 5 | P | P |
| 137.9 | detail | grounded | seeded | 3 | 1.35 | 4.91 | 2.33 | 6 | P | P |
| 141.1 | theme | confabulated | confabulated | 3 | -0.16 | 0.34 | 1.79 | 7 | P | P |
| 141.8 | detail | confabulated | confabulated | 3 | -0.37 | 11.27 | -3.24 | 7 | P | P |
| 141.12 | theme | confabulated | seeded | 3 | 0.22 | 3.13 | 2.24 | 5 | P | P |
| 145.3 | theme | grounded | unresolved |  | 1.55 | 1.81 | 1.32 | 2 | P | . |
| 145.6 | theme | entangled | unresolved |  | 0.39 | 3.62 | -0.34 | 6 | . | P |
| 145.7 | detail | entangled | unresolved |  | 0.87 | 4.68 | -1.84 | 2 | P | . |
| 149.4 | theme | confabulated | confabulated | 3 | -0.76 | 0.63 | 0.00 | 5 | P | P |
| 149.6 | theme | grounded | seeded | 3 | 9.93 | 11.44 | 2.69 | 7 | P | P |
| 149.12 | detail | grounded | confabulated | 3 | 1.49 | 4.14 | 1.25 | 8 | P | P |
| 153.4 | theme | grounded | seeded | 3 | 7.27 | 7.38 | 2.84 | 7 | P | P |
| 153.5 | theme | confabulated | unresolved |  | -0.00 | 1.60 | -0.23 | 3 | . | . |
| 153.8 | detail | confabulated | seeded | 3 | 1.00 | 2.37 | 3.80 | 10 | P | P |
| 157.7 | theme | confabulated | confabulated | 3 | -0.12 | 1.49 | 0.01 | 7 | . | P |
| 157.9 | theme | confabulated | confabulated | 3 | -0.00 | 1.50 | -0.13 | 7 | . | P |
| 157.12 | detail | grounded | seeded | 3 | 1.87 | 6.07 | -0.49 | 8 | P | P |
| 161.4 | theme | confabulated | unresolved |  | -0.73 | -0.91 | 3.58 | 5 | P | . |
| 161.5 | theme | confabulated | unresolved |  | -0.21 | 0.37 | 1.75 | 5 | P | . |
| 161.13 | detail | entangled | unresolved |  | 1.06 | 3.79 | -0.19 | 5 | . | P |
| 165.1 | theme | confabulated | seeded | 3 | 0.34 | 0.70 | 1.52 | 8 | P | P |
| 165.2 | detail | grounded | unresolved |  | 11.21 | 5.49 | 7.85 | 4 | P | . |
| 165.8 | theme | grounded | confabulated | 3 | 1.51 | 4.69 | -0.31 | 10 | P | P |
| 169.0 | theme | entangled | seeded | 3 | 1.75 | 3.68 | 0.36 | 6 | P | P |
| 169.8 | theme | grounded | unresolved |  | 4.01 |  |  | 0 | . | . |
| 169.12 | detail | entangled | unresolved |  | 0.49 | 0.12 | 2.08 | 6 | . | P |
| 173.2 | detail | confabulated | confabulated | 3 | -0.31 | 1.69 | -0.08 | 8 | P | P |
| 173.6 | theme | confabulated | confabulated | 3 | -0.36 | 0.73 | 0.81 | 7 | P | P |
| 173.8 | theme | confabulated | unresolved |  | 0.07 | 1.48 | 0.05 | 9 | . | P |
| 177.0 | theme | confabulated | confabulated | 3 | -0.40 | 1.67 | 0.22 | 7 | P | P |
| 177.9 | theme | confabulated | unresolved |  | 0.03 | 1.34 | 0.09 | 5 | . | P |
| 177.12 | detail | grounded | unresolved |  | 11.81 | 7.64 | 7.10 | 2 | P | . |
| 181.1 | theme | confabulated | confabulated | 3 | -0.50 | 0.41 | 1.37 | 6 | P | P |
| 181.3 | theme | confabulated | confabulated | 3 | -0.73 | 4.34 | -1.95 | 5 | P | P |
| 181.12 | detail | confabulated | confabulated | 3 | -0.33 | 1.25 | 1.27 | 7 | P | P |
| 185.5 | theme | grounded | confabulated | 3 | 2.37 | 2.70 | 2.93 | 6 | P | P |
| 185.6 | theme | confabulated | unresolved |  | 0.00 | 1.38 | 0.49 | 3 | P | . |
| 185.14 | detail | grounded | unresolved |  | 2.18 | 1.66 | 2.55 | 11 | . | P |
| 189.3 | theme | grounded | unresolved |  | 14.42 | 9.13 | 10.06 | 4 | P | . |
| 189.9 | detail | entangled | seeded | 3 | 0.33 | 1.55 | 0.20 | 5 | P | P |
| 189.12 | theme | confabulated | confabulated | 3 | -0.76 | -0.57 | 6.50 | 13 | P | P |
| 193.2 | theme | confabulated | unresolved |  | 1.10 | 3.24 | 0.16 | 8 | P | . |
| 193.5 | theme | confabulated | unresolved |  | 0.06 |  |  |  | . | . |
| 193.8 | detail | confabulated | seeded | 3 | 0.05 | 3.16 | -0.63 | 8 | P | P |
| 197.1 | theme | confabulated | seeded | 3 | 0.02 | 2.15 | 0.07 | 6 | P | P |
| 197.3 | theme | confabulated | confabulated | 3 | -0.05 | 1.20 | 1.35 | 10 | P | P |
| 197.4 | detail | entangled | seeded | 3 | 0.21 | 1.86 | 0.03 | 6 | P | P |
| 201.2 | theme | confabulated | confabulated | 3 | -0.10 | 1.24 | 0.12 | 7 | P | P |
| 201.9 | theme | entangled | confabulated | 3 | -0.01 | 2.77 | -0.76 | 9 | . | P |
| 201.13 | detail | grounded | seeded | 3 | 2.90 | 6.50 | -0.52 | 5 | P | P |
| 205.2 | theme | confabulated | confabulated | 3 | -0.01 | 3.37 | 0.06 | 6 | . | P |
| 205.4 | theme | confabulated | seeded | 3 | 0.03 | 2.68 | -0.10 | 6 | P | P |
| 205.10 | detail | confabulated | confabulated | 3 | -0.24 | 2.47 | -0.66 | 8 | . | P |
| 209.0 | theme | grounded | grounded | 2 | 11.37 | 8.85 | 6.44 | 9 | P | P |
| 209.5 | theme | confabulated | confabulated | 3 | -0.01 | 1.59 | 0.18 | 12 | . | P |
| 209.11 | detail | confabulated | seeded | 3 | 0.18 | 3.30 | -0.47 | 7 | P | P |
| 213.2 | theme | confabulated | entangled | 2 | 1.56 | -0.34 | 6.22 | 5 | P | P |
| 213.6 | detail | confabulated | unresolved |  | -0.00 | 3.06 | -1.86 | 1 | P | . |
| 213.7 | theme | confabulated | confabulated | 3 | -0.60 | -0.64 | 1.94 | 6 | P | P |
| 217.0 | theme | entangled | seeded | 3 | 0.78 | 3.94 | -1.37 | 5 | P | P |
| 217.5 | theme | confabulated | seeded | 3 | 0.12 | 3.51 | -0.50 | 5 | P | P |
| 217.17 | detail | grounded | seeded | 3 | 10.84 | 11.95 | 3.38 | 5 | P | P |
| 221.1 | theme | grounded | seeded | 3 | 5.24 | 10.16 | -3.35 | 5 | P | P |
| 221.5 | theme | confabulated | unresolved |  | -0.00 | 1.32 | -0.12 | 1 | P | . |
| 221.9 | detail | confabulated | confabulated | 3 | -0.28 | 0.56 | 4.55 | 5 | P | P |
| 225.7 | detail | confabulated | confabulated | 3 | 0.00 | 2.84 | -0.97 | 6 | P | P |
| 225.10 | theme | entangled | confabulated | 3 | -0.00 | 1.97 | -0.03 | 6 | . | P |
| 225.13 | theme | confabulated | confabulated | 3 | -1.47 | 0.12 | 0.44 | 5 | P | P |
| 229.2 | theme | confabulated | unresolved |  | 0.01 | 1.82 | 0.17 | 10 | . | P |
| 229.4 | theme | entangled | unresolved |  | 0.39 | 3.20 | -0.58 | 5 | . | P |
| 229.14 | detail | entangled | seeded | 3 | 0.39 | 3.06 | 0.14 | 6 | P | P |
| 233.9 | theme | confabulated | confabulated | 3 | -0.02 | 2.53 | -0.00 | 7 | P | P |
| 233.11 | theme | confabulated | unresolved |  | 0.07 | 1.91 | -0.64 | 1 | . | . |
| 233.15 | detail | grounded | seeded | 3 | 7.16 | 7.69 | 4.42 | 8 | P | P |
| 237.2 | theme | confabulated | confabulated | 3 | -0.24 | 1.86 | -0.50 | 6 | . | P |
| 237.3 | detail | confabulated | seeded | 3 | 0.35 | 3.02 | -0.00 | 7 | P | P |
| 237.4 | theme | entangled | seeded | 3 | 0.71 | 4.93 | -0.44 | 6 | P | P |
| 241.3 | theme | grounded | seeded | 3 | 3.12 | 14.33 | -1.11 | 6 | P | P |
| 241.6 | theme | confabulated | confabulated | 3 | 0.04 | 2.63 | -0.87 | 5 | P | P |
| 241.11 | detail | confabulated | unresolved |  | -0.13 |  |  |  | . | . |
| 245.1 | theme | confabulated | confabulated | 3 | -0.29 | 1.15 | -0.14 | 5 | P | P |
| 245.4 | theme | entangled | confabulated | 3 | 0.00 | 1.47 | -0.05 | 5 | P | P |
| 245.10 | detail | grounded | seeded | 3 | 1.10 | 3.14 | 0.31 | 8 | P | P |
| 249.3 | theme | confabulated | confabulated | 3 | -0.14 | 8.42 | -1.29 | 7 | P | P |
| 249.5 | theme | confabulated | confabulated | 3 | -0.18 | 0.97 | 0.99 | 6 | P | P |
| 249.6 | detail | confabulated | confabulated | 3 | 0.15 | 4.57 | -1.55 | 11 | P | P |
| 253.0 | theme | entangled | unresolved |  | 0.20 | -1.52 | 2.93 | 1 | P | . |
| 253.2 | theme | grounded | grounded | 2 | 8.21 | 3.98 | 7.65 | 6 | P | P |
| 253.9 | detail | confabulated | confabulated | 3 | 0.79 | 2.59 | 2.96 | 6 | P | P |
| 257.4 | theme | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 257.5 | detail | confabulated | seeded | 3 | 0.00 | 3.81 | -0.37 | 5 | P | P |
| 257.6 | theme | entangled | seeded | 3 | 0.24 | 4.50 | -2.03 | 6 | P | P |
| 261.0 | theme | confabulated | confabulated | 3 | 0.21 | 0.79 | 1.96 | 14 | P | P |
| 261.2 | theme | confabulated | confabulated | 3 | 0.07 | 1.37 | 0.70 | 5 | P | P |
| 261.7 | detail | grounded | seeded | 3 | 1.43 | 2.43 | 1.21 | 6 | P | P |
| 265.0 | theme | confabulated | unresolved |  | -0.00 | 5.40 | -0.66 | 5 | P | . |
| 265.3 | theme | grounded | seeded | 3 | 1.88 | 4.03 | -0.12 | 6 | P | P |
| 265.11 | detail | confabulated | confabulated | 3 | -0.09 | 0.91 | 1.56 | 5 | P | P |
| 269.0 | theme | confabulated | unresolved |  | -0.42 | 0.80 | 0.91 | 5 | P | . |
| 269.4 | theme | confabulated | confabulated | 3 | -0.53 | 0.72 | 0.17 | 6 | P | P |
| 269.5 | detail | entangled | seeded | 3 | 1.05 | 3.53 | 1.81 | 9 | P | P |
| 273.0 | theme | entangled | unresolved |  | 0.55 |  |  |  | . | . |
| 273.4 | theme | confabulated | seeded | 3 | 0.23 | 1.91 | 0.02 | 5 | P | P |
| 273.11 | detail | confabulated | unresolved |  | 0.07 | 1.35 | 0.16 | 5 | . | P |
| 277.1 | theme | confabulated | confabulated | 3 | 0.02 | 5.79 | -2.74 | 5 | P | P |
| 277.3 | theme | grounded | seeded | 3 | 1.28 | 6.39 | -2.04 | 5 | P | P |
| 277.10 | detail | grounded | grounded | 2 | 9.82 | 6.26 | 7.23 | 10 | P | P |
| 281.0 | theme | grounded | seeded | 3 | 5.42 | 12.97 | -4.26 | 5 | P | P |
| 281.3 | theme | grounded | unresolved |  | 2.90 |  |  |  | . | . |
| 281.11 | detail | confabulated | confabulated | 3 | -0.09 | 0.02 | 1.72 | 5 | P | P |
| 285.2 | theme | grounded | unresolved |  | 2.76 | 2.65 | 1.90 | 6 | . | P |
| 285.8 | theme | entangled | seeded | 3 | 0.31 | 4.61 | -0.73 | 6 | P | P |
| 285.15 | detail | confabulated | confabulated | 3 | -0.18 | 1.03 | 0.19 | 6 | . | P |
| 289.0 | theme | grounded | unresolved |  | 1.54 | 2.79 | 3.06 | 5 | P | . |
| 289.4 | theme | grounded | confabulated | 3 | 2.19 | 12.11 | -0.77 | 6 | P | P |
| 289.13 | detail | grounded | unresolved |  | 20.76 |  |  |  | . | . |
| 293.6 | theme | confabulated | confabulated | 3 | 0.09 | 2.69 | -1.18 | 5 | P | P |
| 293.8 | theme | confabulated | unresolved |  | -0.26 | 2.11 | -0.43 | 5 | P | . |
| 293.10 | detail | grounded | seeded | 3 | 14.15 | 15.32 | 5.01 | 6 | P | P |
| 297.1 | theme | grounded | unresolved |  | 2.11 |  |  |  | . | . |
| 297.5 | theme | confabulated | unresolved |  | 1.23 | 6.60 | -0.99 | 5 | P | . |
| 297.13 | detail | grounded | unresolved |  | 53.99 |  |  | 0 | . | . |
| 301.0 | theme | grounded | unresolved |  | 8.45 | 5.92 | 3.73 | 1 | P | . |
| 301.2 | detail | confabulated | seeded | 3 | 0.11 | 4.05 | -0.41 | 8 | P | P |
| 301.4 | theme | grounded | seeded | 3 | 4.48 | 13.25 | -3.80 | 5 | P | P |
| 305.2 | theme | confabulated | unresolved |  | -0.07 | 3.58 | -1.11 | 5 | . | . |
| 305.7 | theme | entangled | seeded | 3 | 2.33 | 5.08 | 0.01 | 18 | P | P |
| 305.16 | detail | confabulated | confabulated | 3 | -2.15 | 0.98 | 0.95 | 5 | P | P |
| 309.4 | theme | confabulated | unresolved |  | -0.17 | 1.86 | -0.21 | 5 | . | . |
| 309.8 | theme | confabulated | confabulated | 3 | -1.36 | 1.27 | -0.82 | 5 | P | P |
| 309.15 | detail | confabulated | confabulated | 3 | -1.07 | 0.08 | 6.00 | 7 | P | P |
| 313.0 | theme | grounded | unresolved |  | 9.32 | 9.51 | 1.41 | 5 | . | P |
| 313.4 | theme | confabulated | confabulated | 3 | -0.17 | 0.47 | 1.40 | 5 | P | P |
| 313.15 | detail | entangled | seeded | 3 | 2.05 | 5.22 | -0.16 | 6 | P | P |
| 317.4 | theme | entangled | confabulated | 3 | -0.20 | 1.24 | -0.15 | 6 | . | P |
| 317.6 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 317.17 | detail | confabulated | confabulated | 3 | -0.82 | -0.27 | 0.94 | 5 | P | P |
| 321.6 | theme | confabulated | confabulated | 3 | -0.67 | 0.26 | 1.06 | 6 | . | P |
| 321.8 | detail | confabulated | confabulated | 3 | -0.24 | -0.23 | 3.34 | 6 | P | P |
| 321.10 | theme | confabulated | confabulated | 3 | -1.30 | -1.64 | 2.24 | 7 | P | P |
| 325.2 | theme | grounded | confabulated | 3 | 3.53 | 6.65 | 0.80 | 5 | P | P |
| 325.6 | theme | entangled | seeded | 3 | 0.50 | 2.99 | 0.66 | 7 | P | P |
| 325.7 | detail | confabulated | confabulated | 3 | 0.00 | 2.78 | -0.70 | 5 | P | P |
| 329.0 | theme | grounded | grounded | 2 | 1.57 | -0.02 | 3.26 | 8 | P | P |
| 329.8 | theme | confabulated | unresolved |  | 0.01 | 1.71 | -0.18 | 7 | . | P |
| 329.12 | detail | confabulated | confabulated | 3 | -0.08 | 1.18 | 0.27 | 7 | P | P |
| 333.2 | theme | grounded | seeded | 3 | 2.32 | 4.00 | 2.04 | 6 | P | P |
| 333.3 | theme | grounded | unresolved |  | 1.48 | 5.87 | 0.49 | 5 | . | . |
| 333.9 | detail | entangled | seeded | 3 | 0.79 | 2.69 | -0.05 | 7 | P | P |
| 337.0 | theme | grounded | unresolved |  | 9.11 | 18.14 | -1.14 | 5 | . | P |
| 337.7 | theme | entangled | unresolved |  | 0.01 |  |  |  | . | . |
| 337.8 | detail | entangled | confabulated | 3 | 0.00 | 2.19 | 0.00 | 7 | P | P |
| 341.1 | theme | entangled | seeded | 3 | 0.67 | 4.45 | -1.83 | 7 | P | P |
| 341.6 | theme | grounded | unresolved |  | 3.44 | 6.82 | 0.40 | 5 | P | . |
| 341.14 | detail | grounded | seeded | 3 | 4.46 | 10.35 | -0.53 | 5 | P | P |
| 345.4 | theme | grounded | seeded | 3 | 6.57 | 15.71 | -1.14 | 6 | P | P |
| 345.6 | theme | grounded | unresolved |  | 2.79 | 2.85 | 1.65 | 6 | . | P |
| 345.12 | detail | entangled | seeded | 3 | 0.93 | 1.80 | 0.59 | 5 | P | P |
| 349.0 | theme | confabulated | confabulated | 3 | -0.05 | 1.16 | 1.39 | 6 | P | P |
| 349.5 | theme | confabulated | unresolved |  | -0.13 | 7.75 | -5.42 | 2 | P | . |
| 349.9 | detail | confabulated | confabulated | 3 | -1.41 | 1.08 | 2.46 | 7 | P | P |
| 353.1 | theme | entangled | confabulated | 3 | -0.05 | 1.23 | 0.02 | 5 | P | P |
| 353.4 | theme | confabulated | confabulated | 3 | -0.00 | 1.33 | -0.12 | 5 | P | P |
| 353.9 | detail | entangled | confabulated | 3 | 0.52 | 3.75 | -0.43 | 12 | P | P |
| 357.0 | theme | grounded | unresolved |  | 17.62 |  |  | 0 | . | . |
| 357.1 | theme | entangled | unresolved |  | 7.10 | 4.34 | 3.96 | 1 | P | . |
| 357.2 | detail | confabulated | seeded | 3 | 0.78 | 4.29 | -1.02 | 5 | P | P |
| 361.2 | theme | confabulated | unresolved |  | -2.09 |  |  | 0 | . | . |
| 361.10 | theme | confabulated | confabulated | 3 | -1.21 | -2.96 | 7.11 | 6 | P | P |
| 365.7 | theme | grounded | unresolved |  | 5.94 | 7.99 | 3.57 | 6 | . | P |
| 365.9 | theme | confabulated | confabulated | 3 | -0.03 | 1.18 | 0.08 | 6 | P | P |
| 365.14 | detail | entangled | seeded | 3 | 0.89 | 11.93 | -3.13 | 6 | P | P |
| 369.0 | theme | confabulated | unresolved |  | -0.24 | 1.12 | 1.41 | 4 | P | . |
| 369.4 | theme | entangled | unresolved |  | 0.00 | 1.39 | -0.00 | 5 | . | P |
| 369.12 | detail | entangled | unresolved |  | 0.76 | 5.33 | 0.71 | 7 | . | P |
| 373.1 | theme | grounded | seeded | 3 | 9.08 | 14.79 | -1.76 | 5 | P | P |
| 373.2 | theme | entangled | seeded | 3 | 0.03 | 2.60 | -0.60 | 6 | P | P |
| 373.5 | detail | confabulated | confabulated | 3 | -0.01 | 1.50 | 0.24 | 5 | P | P |
| 377.2 | theme | confabulated | confabulated | 3 | -0.33 | 1.29 | -0.20 | 6 | P | P |
| 377.13 | theme | entangled | seeded | 3 | 0.70 | 3.21 | -0.43 | 7 | P | P |
| 377.14 | detail | entangled | seeded | 3 | 0.33 | 5.91 | -2.61 | 5 | P | P |
| 381.4 | theme | grounded | unresolved |  | 5.45 |  |  |  | . | . |
| 381.6 | theme | confabulated | confabulated | 3 | -0.00 | 1.42 | -0.01 | 5 | P | P |
| 381.7 | detail | grounded | seeded | 3 | 27.33 | 31.82 | -0.36 | 5 | P | P |
| 385.4 | theme | confabulated | seeded | 3 | 0.61 | 2.92 | -0.21 | 5 | P | P |
| 385.8 | theme | confabulated | confabulated | 3 | -0.01 | 1.33 | -0.06 | 11 | P | P |
| 385.16 | detail | confabulated | unresolved |  |  |  |  |  | . | . |
| 389.5 | theme | confabulated | unresolved |  | 0.00 | 1.32 | -0.12 | 1 | P | . |
| 389.9 | theme | entangled | unresolved |  | 0.55 | 4.21 | -0.98 | 6 | . | P |
| 389.13 | detail | confabulated | seeded | 3 | 0.08 | 3.53 | -1.66 | 5 | P | P |
| 393.4 | theme | confabulated | confabulated | 3 | 0.01 | 3.82 | -0.52 | 8 | P | P |
| 393.11 | theme | confabulated | confabulated | 3 | -0.30 | 0.42 | 1.03 | 6 | P | P |
| 393.12 | detail | confabulated | unresolved |  | -0.61 | -1.43 | 3.53 | 5 | P | . |
| 397.1 | theme | confabulated | confabulated | 3 | -0.08 | 3.12 | -0.90 | 5 | P | P |
| 397.3 | detail | confabulated | unresolved |  | -0.69 |  |  |  | . | . |
| 397.5 | theme | confabulated | unresolved |  | 0.00 | 1.24 | 0.12 | 5 | . | P |
