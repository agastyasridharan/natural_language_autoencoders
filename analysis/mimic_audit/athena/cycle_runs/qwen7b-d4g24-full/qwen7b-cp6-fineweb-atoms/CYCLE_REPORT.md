# Cycle-consistency verdict report

run `qwen7b-cp6-fineweb-atoms`  claims 1196  k 5

## Verdict distribution

| verdict | n | % |
|---|---|---|
| grounded | 51 | 4.3 |
| entangled | 28 | 2.3 |
| seeded | 394 | 32.9 |
| confabulated | 341 | 28.5 |
| unresolved | 382 | 31.9 |

unresolved by reason: validity 157, power 152, no text-passing decoys 71, claim span unlocated (fc) 2

## Gate flow (spec Step 7)

- 1196 claims -> 382 unresolved (pre-checks), 814 reached the gates
- Gate 1 (specificity): 79 pass -> Gate 2, 735 fail -> Gate 3
- Gate 2 (ownership): 51 grounded, 28 entangled
- Gate 3 (shared signal): 394 seeded, 341 confabulated

## Atom -> cycle verdict shift

rows = published atom verdict, columns = cycle verdict

| atom \ cycle | grounded | entangled | seeded | confabulated | unresolved |
|---|---|---|---|---|---|
| confabulated | 0 | 6 | 40 | 290 | 116 |
| entangled | 0 | 10 | 151 | 40 | 159 |
| grounded | 51 | 12 | 203 | 11 | 107 |

## Pre-checks + grounding

- power passed 922/1196, validity passed 901/1196
- grounding estimate (M_BB - med D*, Gate-1 passers only (grounded+entangled; M_BB > 0), n=79): median 4.7128, p10 2.0273, p90 8.3225

## kappa_0 re-fit (Gate 1 null on THIS run)

- calibration kappa_0 = 0.9 (applied)
- re-fit on 647 survivor D* families = 0.9
- decoy-vs-decoy exceedance rate at the calibration kappa_0 = 0.0486 (target <= 0.05, whole run incl. in-sample)
- **out-of-sample** (375 activations disjoint from the 25 calibration ones, 604 families): re-fit kappa_0 = 0.9, FP rate at calibration kappa_0 = 0.049 -> TRANSFERS (<= 0.05)
- _This re-fit is IN-SAMPLE when the run activations are the calibration split (e.g. the S4 small-N run reuses the 25 calibration activations): the LODO families and grid are then identical to calib_cycle.py's, so the FP rate is a consistency check, NOT out-of-sample validation. Genuine out-of-sample validation of kappa_0 requires a run on activations disjoint from calibration (S5's 375 new activations). kappa_0_refit agrees with the applied calibration kappa_0._

## Per-claim

| idx.ci | type | atom | cycle | gate | M_BB | gate1_bar | ground_est | surv | pw | vd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | detail | grounded | seeded | 3 | 12.32 | 17.20 | 2.97 | 10 | P | P |
| 0.4 | theme | entangled | unresolved |  | 3.80 | 5.93 | 0.58 | 5 | . | . |
| 0.6 | theme | confabulated | unresolved |  | 0.01 | 1.82 | -0.30 | 9 | . | P |
| 1.1 | theme | confabulated | confabulated | 3 | -0.24 | 2.82 | -0.38 | 5 | . | P |
| 1.5 | detail | grounded | unresolved |  | 2.35 |  |  |  | . | . |
| 1.8 | theme | entangled | unresolved |  | 0.09 | 1.28 | 0.14 | 5 | P | . |
| 2.5 | theme | entangled | seeded | 3 | 0.39 | 2.42 | -0.79 | 5 | P | P |
| 2.8 | theme | confabulated | confabulated | 3 | -0.75 | 0.12 | 0.60 | 6 | . | P |
| 2.11 | detail | entangled | confabulated | 3 | 0.25 | 1.05 | 0.83 | 5 | P | P |
| 3.12 | theme | confabulated | unresolved |  | -0.11 | 1.03 | -0.03 | 4 | P | . |
| 3.14 | detail | grounded | seeded | 3 | 1.38 | 1.68 | 1.29 | 5 | P | P |
| 3.16 | theme | confabulated | unresolved |  | -1.51 |  |  |  | . | . |
| 4.0 | theme | grounded | seeded | 3 | 1.55 | 4.72 | -1.16 | 5 | P | P |
| 4.2 | theme | entangled | unresolved |  | 0.80 | 4.18 | -0.92 | 5 | P | . |
| 4.7 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 5.6 | theme | confabulated | unresolved |  | -0.31 |  |  |  | . | . |
| 5.9 | theme | confabulated | confabulated | 3 | -0.07 | 0.99 | -0.06 | 8 | . | P |
| 5.10 | detail | confabulated | seeded | 3 | 0.15 | 1.07 | 0.05 | 5 | P | P |
| 6.1 | theme | confabulated | confabulated | 3 | -0.86 | 1.15 | 0.83 | 5 | . | P |
| 6.3 | detail | grounded | seeded | 3 | 7.91 | 16.70 | -1.51 | 11 | P | P |
| 6.9 | theme | entangled | unresolved |  | 0.04 | 3.56 | -0.17 | 5 | . | . |
| 7.3 | theme | confabulated | confabulated | 3 | -0.62 | -0.62 | 2.93 | 5 | P | P |
| 7.5 | detail | entangled | seeded | 3 | 0.61 | 2.14 | 0.00 | 5 | P | P |
| 7.6 | theme | grounded | seeded | 3 | 3.18 | 10.19 | -0.28 | 6 | P | P |
| 8.3 | theme | confabulated | confabulated | 3 | -0.17 | 0.54 | 0.37 | 5 | P | P |
| 8.5 | detail | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 8.10 | theme | confabulated | confabulated | 3 | -0.00 | 1.01 | 0.19 | 7 | . | P |
| 9.0 | theme | grounded | seeded | 3 | 3.46 | 4.61 | 0.85 | 7 | P | P |
| 9.2 | theme | entangled | seeded | 3 | 0.75 | 3.49 | -0.08 | 6 | P | P |
| 9.11 | detail | entangled | seeded | 3 | 0.92 | 2.26 | 0.41 | 5 | P | P |
| 10.0 | theme | entangled | seeded | 3 | 0.63 | 1.57 | 1.92 | 6 | P | P |
| 10.5 | detail | entangled | unresolved |  | 0.00 | 1.75 | -0.44 | 4 | P | . |
| 10.8 | theme | confabulated | unresolved |  | -0.51 | -0.11 | 1.04 | 3 | P | . |
| 11.2 | detail | entangled | seeded | 3 | 2.02 | 3.18 | 1.23 | 7 | P | P |
| 11.3 | theme | entangled | seeded | 3 | 0.42 | 0.46 | 1.03 | 6 | P | P |
| 11.6 | theme | confabulated | unresolved |  | -0.01 | 1.23 | -0.12 | 3 | P | . |
| 12.0 | theme | entangled | seeded | 3 | 2.97 | 10.81 | 3.03 | 9 | P | P |
| 12.4 | theme | entangled | unresolved |  | 0.22 | 1.67 | 0.33 | 8 | . | P |
| 12.8 | detail | confabulated | confabulated | 3 | -0.11 | 2.82 | -1.64 | 5 | P | P |
| 13.2 | theme | grounded | seeded | 3 | 3.00 | 5.88 | -0.48 | 5 | P | P |
| 13.6 | theme | grounded | seeded | 3 | 1.42 | 4.47 | 0.07 | 5 | P | P |
| 13.15 | detail | confabulated | confabulated | 3 | -0.07 | 2.56 | -1.06 | 24 | P | P |
| 14.0 | theme | grounded | unresolved |  | 11.53 | 9.65 | 4.69 | 6 | . | P |
| 14.5 | theme | entangled | entangled | 2 | 0.47 | -0.83 | 3.70 | 6 | P | P |
| 14.11 | detail | grounded | unresolved |  | 5.82 |  |  |  | . | . |
| 15.8 | theme | confabulated | confabulated | 3 | -0.06 | 0.61 | 0.66 | 6 | P | P |
| 15.9 | detail | confabulated | confabulated | 3 | -0.07 | 0.47 | 0.54 | 5 | P | P |
| 15.11 | theme | confabulated | seeded | 3 | 0.04 | 1.12 | -0.02 | 5 | P | P |
| 16.5 | theme | entangled | seeded | 3 | 0.25 | 1.62 | -0.16 | 7 | P | P |
| 16.7 | detail | entangled | confabulated | 3 | 0.08 | 1.08 | 0.19 | 11 | P | P |
| 16.11 | theme | confabulated | seeded | 3 | 0.11 | 2.13 | -0.20 | 8 | P | P |
| 17.2 | theme | grounded | seeded | 3 | 3.31 | 7.56 | -0.31 | 7 | P | P |
| 17.8 | theme | confabulated | confabulated | 3 | -0.08 | 0.78 | 0.23 | 6 | P | P |
| 17.9 | detail | grounded | grounded | 2 | 7.25 | 3.47 | 6.04 | 6 | P | P |
| 18.0 | theme | grounded | unresolved |  | 3.47 | 5.09 | 0.54 | 5 | . | . |
| 18.4 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 18.5 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 19.0 | theme | entangled | confabulated | 3 | 0.11 | 2.47 | -0.57 | 6 | P | P |
| 19.2 | theme | entangled | unresolved |  | 0.47 | 2.07 | 0.10 | 8 | . | P |
| 19.5 | detail | confabulated | seeded | 3 | 0.08 | 1.30 | -0.03 | 7 | P | P |
| 20.2 | theme | grounded | unresolved |  | 3.04 | 5.00 | 0.24 | 6 | P | . |
| 20.9 | theme | confabulated | confabulated | 3 | -0.08 | 1.14 | -0.17 | 6 | . | P |
| 20.10 | detail | confabulated | unresolved |  | -0.29 | -0.53 | 2.29 | 5 | P | . |
| 21.0 | theme | confabulated | confabulated | 3 | -0.30 | 2.13 | -0.49 | 6 | P | P |
| 21.4 | theme | confabulated | confabulated | 3 | -0.36 | -0.44 | 1.96 | 6 | P | P |
| 21.7 | detail | grounded | grounded | 2 | 10.95 | 10.66 | 2.77 | 6 | P | P |
| 22.1 | theme | entangled | confabulated | 3 | 0.54 | 4.50 | -2.35 | 5 | P | P |
| 22.8 | theme | entangled | confabulated | 3 | 0.80 | 3.94 | -1.32 | 5 | P | P |
| 22.9 | detail | confabulated | seeded | 3 | 0.01 | 1.74 | -0.63 | 8 | P | P |
| 23.6 | detail | entangled | unresolved |  | 0.00 | 7.06 | -5.60 | 4 | P | . |
| 23.7 | theme | confabulated | unresolved |  | 0.01 | 1.28 | -0.14 | 4 | P | . |
| 23.9 | theme | entangled | seeded | 3 | 0.25 | 1.83 | -0.39 | 6 | P | P |
| 24.3 | theme | grounded | grounded | 2 | 6.50 | 5.20 | 5.08 | 7 | P | P |
| 24.10 | theme | confabulated | seeded | 3 | 0.08 | 1.21 | -0.00 | 8 | P | P |
| 24.13 | detail | confabulated | confabulated | 3 | -0.60 | 0.19 | 1.87 | 5 | P | P |
| 25.1 | theme | grounded | seeded | 3 | 1.88 | 3.65 | 0.77 | 5 | P | P |
| 25.10 | theme | confabulated | confabulated | 3 | -0.03 | 2.25 | -0.76 | 11 | P | P |
| 25.11 | detail | confabulated | confabulated | 3 | -0.15 | 1.23 | 0.18 | 6 | P | P |
| 26.0 | theme | grounded | seeded | 3 | 1.53 | 2.58 | 0.18 | 6 | P | P |
| 26.4 | theme | confabulated | confabulated | 3 | -0.07 | 0.81 | 0.40 | 5 | . | P |
| 26.11 | detail | entangled | seeded | 3 | 0.27 | 1.46 | 0.14 | 5 | P | P |
| 27.1 | theme | confabulated | confabulated | 3 | -0.19 | 0.77 | 0.26 | 5 | P | P |
| 27.8 | theme | grounded | seeded | 3 | 1.36 | 8.80 | -3.16 | 8 | P | P |
| 27.10 | detail | confabulated | confabulated | 3 | -0.50 | 0.33 | 0.66 | 5 | P | P |
| 28.0 | theme | grounded | seeded | 3 | 4.75 | 7.25 | 1.40 | 10 | P | P |
| 28.4 | theme | confabulated | confabulated | 3 | -0.77 | 2.04 | 0.83 | 7 | P | P |
| 28.9 | detail | confabulated | confabulated | 3 | -0.56 | 0.88 | -0.22 | 8 | P | P |
| 29.2 | theme | grounded | unresolved |  | 8.14 | 9.06 | 3.66 | 6 | P | . |
| 29.4 | theme | confabulated | unresolved |  | 0.00 | 1.15 | -0.25 | 1 | P | . |
| 29.5 | detail | entangled | unresolved |  | 0.00 |  |  |  | . | . |
| 30.4 | theme | grounded | seeded | 3 | 2.31 | 4.85 | -1.25 | 7 | P | P |
| 30.8 | detail | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 30.13 | theme | grounded | unresolved |  | 2.26 |  |  |  | . | . |
| 31.2 | theme | grounded | seeded | 3 | 10.57 | 11.47 | 2.30 | 17 | P | P |
| 31.4 | theme | entangled | seeded | 3 | 0.09 | 1.84 | -0.13 | 16 | P | P |
| 31.5 | detail | entangled | unresolved |  | 0.19 |  |  |  | . | . |
| 32.3 | theme | grounded | seeded | 3 | 1.47 | 3.21 | -0.39 | 8 | P | P |
| 32.4 | detail | confabulated | confabulated | 3 | -0.13 | 1.38 | -0.24 | 5 | P | P |
| 32.15 | theme | confabulated | confabulated | 3 | 0.08 | 0.86 | 0.29 | 6 | P | P |
| 33.1 | theme | grounded | unresolved |  | 16.48 | 19.55 | 0.47 | 13 | . | P |
| 33.3 | detail | grounded | seeded | 3 | 9.50 | 9.86 | 2.14 | 5 | P | P |
| 33.9 | theme | entangled | unresolved |  | 0.10 | 1.61 | -0.26 | 6 | . | P |
| 34.0 | theme | grounded | grounded | 2 | 12.62 | 7.59 | 12.19 | 12 | P | P |
| 34.1 | theme | grounded | seeded | 3 | 4.53 | 12.95 | -1.70 | 5 | P | P |
| 34.8 | detail | entangled | seeded | 3 | 0.74 | 7.74 | -4.65 | 5 | P | P |
| 35.1 | theme | grounded | seeded | 3 | 3.24 | 4.83 | 0.43 | 7 | P | P |
| 35.5 | theme | confabulated | confabulated | 3 | -0.13 | 1.15 | -0.20 | 5 | P | P |
| 35.6 | detail | grounded | grounded | 2 | 20.76 | 17.55 | 10.95 | 10 | P | P |
| 36.0 | theme | grounded | seeded | 3 | 2.28 | 5.38 | -0.95 | 7 | P | P |
| 36.5 | theme | entangled | confabulated | 3 | 0.09 | 1.03 | 0.10 | 5 | P | P |
| 36.6 | detail | entangled | entangled | 2 | 9.41 | 8.53 | 7.15 | 5 | P | P |
| 37.1 | theme | grounded | unresolved |  | 5.17 | 8.68 | 4.13 | 6 | P | . |
| 37.9 | theme | entangled | seeded | 3 | 1.36 | 3.24 | -0.16 | 5 | P | P |
| 37.13 | detail | entangled | unresolved |  | 0.53 | 0.92 | 0.88 | 8 | . | P |
| 38.0 | theme | confabulated | confabulated | 3 | -0.70 | 0.90 | -0.13 | 6 | P | P |
| 38.1 | theme | entangled | confabulated | 3 | 1.09 | 6.27 | -1.06 | 8 | P | P |
| 38.10 | detail | entangled | unresolved |  | 3.57 |  |  | 0 | . | . |
| 39.5 | theme | confabulated | confabulated | 3 | -0.05 | 0.91 | -0.00 | 7 | . | P |
| 39.6 | theme | confabulated | seeded | 3 | 0.03 | 1.08 | -0.03 | 8 | P | P |
| 39.8 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 40.4 | theme | confabulated | confabulated | 3 | -1.02 | -2.62 | 4.37 | 5 | P | P |
| 40.6 | theme | confabulated | confabulated | 3 | -0.02 | 1.15 | 0.05 | 5 | P | P |
| 40.10 | detail | confabulated | confabulated | 3 | -0.03 | 1.60 | -0.23 | 10 | . | P |
| 41.0 | theme | grounded | seeded | 3 | 9.56 | 11.45 | 5.80 | 7 | P | P |
| 41.1 | theme | confabulated | unresolved |  | -0.33 | 0.77 | 1.30 | 4 | . | . |
| 41.3 | detail | grounded | seeded | 3 | 7.09 | 11.42 | -0.08 | 6 | P | P |
| 42.0 | theme | grounded | unresolved |  | 9.83 | 11.23 | 1.00 | 14 | . | P |
| 42.7 | detail | grounded | confabulated | 3 | 1.58 | 3.73 | 2.45 | 12 | P | P |
| 42.8 | theme | confabulated | confabulated | 3 | -1.21 | 0.44 | -0.40 | 5 | . | P |
| 43.1 | theme | confabulated | unresolved |  | 0.01 | 0.97 | 0.20 | 6 | . | P |
| 43.6 | theme | grounded | unresolved |  | 2.20 | 4.20 | -0.13 | 6 | . | P |
| 43.14 | detail | entangled | seeded | 3 | 0.24 | 1.83 | -0.09 | 5 | P | P |
| 44.1 | theme | confabulated | confabulated | 3 | -0.29 | 0.86 | 0.16 | 6 | P | P |
| 44.3 | detail | confabulated | confabulated | 3 | -0.32 | 0.75 | 0.02 | 7 | P | P |
| 44.7 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 45.2 | theme | grounded | unresolved |  | 1.21 | 4.46 | -1.09 | 7 | . | P |
| 45.6 | theme | grounded | seeded | 3 | 2.91 | 6.03 | -1.91 | 6 | P | P |
| 45.15 | detail | confabulated | unresolved |  | -0.27 | 1.19 | -0.04 | 5 | P | . |
| 46.0 | theme | grounded | seeded | 3 | 14.82 | 16.61 | 3.65 | 10 | P | P |
| 46.9 | theme | confabulated | confabulated | 3 | -0.03 | 1.24 | 0.00 | 8 | . | P |
| 46.12 | detail | grounded | seeded | 3 | 1.03 | 3.02 | -0.90 | 5 | P | P |
| 47.8 | theme | confabulated | unresolved |  | 0.04 |  |  |  | . | . |
| 47.10 | theme | grounded | seeded | 3 | 1.02 | 3.40 | -0.83 | 8 | P | P |
| 47.12 | detail | confabulated | confabulated | 3 | -0.40 | 0.91 | 0.30 | 5 | P | P |
| 48.3 | theme | confabulated | confabulated | 3 | -1.04 | 1.30 | -0.19 | 5 | P | P |
| 48.7 | theme | entangled | unresolved |  | 0.15 | 1.33 | -0.05 | 6 | . | P |
| 48.12 | detail | entangled | confabulated | 3 | 0.12 | 3.74 | -0.08 | 9 | P | P |
| 49.0 | theme | grounded | unresolved |  | 2.67 | 3.74 | 1.01 | 5 | . | P |
| 49.6 | theme | confabulated | confabulated | 3 | -0.10 | 1.19 | 0.19 | 5 | P | P |
| 49.16 | detail | confabulated | confabulated | 3 | -0.09 | 1.30 | -0.05 | 6 | P | P |
| 50.5 | theme | confabulated | confabulated | 3 | 0.12 | 0.53 | 2.50 | 9 | P | P |
| 50.8 | detail | grounded | seeded | 3 | 1.28 | 3.39 | -0.37 | 7 | P | P |
| 50.9 | theme | confabulated | unresolved |  | -0.01 | 0.79 | 0.10 | 1 | . | . |
| 51.2 | detail | entangled | seeded | 3 | 0.65 | 2.05 | 0.49 | 5 | P | P |
| 51.3 | theme | grounded | unresolved |  | 1.33 | 1.36 | 0.87 | 1 | P | . |
| 51.5 | theme | grounded | seeded | 3 | 2.03 | 5.48 | -0.59 | 8 | P | P |
| 52.1 | theme | entangled | unresolved |  | 0.88 | 4.31 | -1.56 | 5 | P | . |
| 52.2 | theme | grounded | seeded | 3 | 3.62 | 6.88 | -1.50 | 5 | P | P |
| 52.3 | detail | grounded | seeded | 3 | 5.70 | 9.70 | -1.02 | 8 | P | P |
| 53.0 | theme | confabulated | confabulated | 3 | -0.09 | 0.67 | 0.48 | 5 | P | P |
| 53.6 | detail | entangled | unresolved |  | 0.00 | 11.01 | -7.22 | 2 | P | . |
| 53.11 | theme | grounded | unresolved |  | 17.99 | 12.86 | 9.85 | 7 | . | P |
| 54.1 | theme | grounded | seeded | 3 | 5.07 | 15.30 | -6.11 | 5 | P | P |
| 54.5 | theme | grounded | seeded | 3 | 9.12 | 11.39 | -0.25 | 5 | P | P |
| 54.13 | detail | grounded | grounded | 2 | 7.57 | 5.12 | 6.89 | 11 | P | P |
| 55.4 | detail | confabulated | confabulated | 3 | -0.21 | 0.76 | 0.22 | 6 | P | P |
| 55.6 | theme | entangled | seeded | 3 | 0.03 | 2.71 | -0.61 | 13 | P | P |
| 55.10 | theme | entangled | seeded | 3 | 0.19 | 4.21 | -1.75 | 7 | P | P |
| 56.3 | theme | confabulated | confabulated | 3 | 0.04 | 1.45 | 1.75 | 5 | P | P |
| 56.4 | theme | entangled | entangled | 2 | 1.22 | 0.40 | 2.22 | 6 | P | P |
| 56.5 | detail | confabulated | unresolved |  | 0.00 | 1.09 | 0.31 | 3 | P | . |
| 57.0 | theme | grounded | seeded | 3 | 10.24 | 12.20 | 1.04 | 5 | P | P |
| 57.1 | theme | confabulated | unresolved |  | -0.10 | 0.07 | 0.73 | 1 | . | . |
| 57.2 | detail | grounded | seeded | 3 | 7.41 | 9.04 | 1.27 | 9 | P | P |
| 58.5 | theme | grounded | seeded | 3 | 4.27 | 11.48 | -1.82 | 10 | P | P |
| 58.9 | detail | grounded | seeded | 3 | 5.53 | 8.14 | 2.39 | 6 | P | P |
| 58.10 | theme | confabulated | confabulated | 3 | -0.11 | 0.73 | 0.48 | 5 | P | P |
| 59.0 | theme | grounded | seeded | 3 | 1.66 | 5.08 | 0.96 | 6 | P | P |
| 59.7 | theme | confabulated | confabulated | 3 | -0.23 | 1.03 | 0.05 | 14 | P | P |
| 59.8 | detail | confabulated | confabulated | 3 | -0.74 | 1.02 | -0.39 | 5 | . | P |
| 60.7 | theme | confabulated | seeded | 3 | 0.05 | 1.11 | 0.02 | 7 | P | P |
| 60.8 | detail | grounded | grounded | 2 | 5.36 | 3.34 | 4.71 | 5 | P | P |
| 60.9 | theme | confabulated | confabulated | 3 | 0.00 | 0.11 | 1.35 | 5 | P | P |
| 61.0 | theme | grounded | unresolved |  | 1.71 | 15.65 | -0.58 | 5 | P | . |
| 61.6 | theme | confabulated | confabulated | 3 | -0.30 | 0.28 | 0.56 | 5 | P | P |
| 61.15 | detail | confabulated | confabulated | 3 | -0.36 | 0.05 | 2.02 | 8 | P | P |
| 62.0 | theme | grounded | unresolved |  | 7.72 | 19.62 | -7.77 | 4 | P | . |
| 62.1 | theme | entangled | unresolved |  | 0.17 | 1.42 | -0.08 | 8 | . | P |
| 62.9 | detail | entangled | entangled | 2 | 0.86 | -0.03 | 2.58 | 5 | P | P |
| 63.0 | theme | grounded | seeded | 3 | 19.98 | 30.72 | -0.43 | 22 | P | P |
| 63.4 | theme | confabulated | unresolved |  | 0.00 |  |  |  | . | . |
| 63.11 | detail | confabulated | confabulated | 3 | -0.03 | 1.39 | 0.05 | 6 | P | P |
| 64.0 | theme | entangled | seeded | 3 | 2.51 | 10.67 | -4.14 | 7 | P | P |
| 64.2 | theme | confabulated | unresolved |  | -0.07 | 0.20 | 0.62 | 1 | . | . |
| 64.9 | detail | entangled | unresolved |  | 0.16 | 2.96 | -1.90 | 1 | P | . |
| 65.2 | detail | confabulated | confabulated | 3 | -0.20 | 0.26 | 0.96 | 5 | P | P |
| 65.4 | theme | confabulated | confabulated | 3 | -0.24 | 0.75 | 0.11 | 6 | P | P |
| 65.5 | theme | grounded | seeded | 3 | 3.48 | 8.07 | -1.90 | 5 | P | P |
| 66.0 | theme | confabulated | unresolved |  | 0.04 | 0.95 | 0.20 | 14 | . | P |
| 66.4 | theme | grounded | seeded | 3 | 6.10 | 6.99 | 1.05 | 9 | P | P |
| 66.11 | detail | grounded | unresolved |  | 5.41 | 5.27 | 4.43 | 5 | P | . |
| 67.6 | theme | confabulated | unresolved |  | 0.00 | 1.03 | -0.09 | 4 | P | . |
| 67.7 | detail | entangled | confabulated | 3 | 0.00 | 2.26 | 0.16 | 6 | P | P |
| 67.10 | theme | entangled | seeded | 3 | 0.38 | 1.82 | 1.26 | 8 | P | P |
| 68.5 | theme | confabulated | confabulated | 3 | -0.70 | 2.78 | -1.40 | 10 | P | P |
| 68.9 | detail | grounded | grounded | 2 | 37.23 | 27.33 | 23.52 | 8 | P | P |
| 68.11 | theme | confabulated | seeded | 3 | 0.04 | 1.36 | -0.17 | 5 | P | P |
| 69.9 | theme | grounded | seeded | 3 | 4.52 | 9.57 | -1.03 | 15 | P | P |
| 69.12 | detail | grounded | seeded | 3 | 2.43 | 4.92 | 0.27 | 7 | P | P |
| 69.14 | theme | entangled | seeded | 3 | 2.06 | 10.25 | -4.29 | 9 | P | P |
| 70.5 | theme | confabulated | confabulated | 3 | -1.10 | 0.02 | 1.56 | 5 | P | P |
| 70.6 | theme | entangled | unresolved |  | 0.00 | 0.78 | 0.12 | 1 | P | . |
| 70.12 | detail | confabulated | unresolved |  | -0.69 | -4.20 | 4.74 | 3 | P | . |
| 71.1 | theme | grounded | seeded | 3 | 6.60 | 16.40 | -1.91 | 8 | P | P |
| 71.6 | theme | confabulated | unresolved |  | 0.00 | 1.16 | -0.09 | 10 | . | P |
| 71.9 | detail | confabulated | confabulated | 3 | -1.62 | -1.44 | 1.92 | 5 | P | P |
| 72.0 | theme | grounded | seeded | 3 | 6.90 | 14.89 | -3.94 | 7 | P | P |
| 72.6 | theme | entangled | seeded | 3 | 0.49 | 1.97 | -0.30 | 6 | P | P |
| 72.7 | detail | grounded | unresolved |  | 32.83 | 33.63 | 10.66 | 10 | P | . |
| 73.2 | theme | confabulated | unresolved |  | -0.45 |  |  |  | . | . |
| 73.9 | detail | confabulated | confabulated | 3 | -0.63 | 0.08 | 0.65 | 5 | P | P |
| 73.11 | theme | confabulated | confabulated | 3 | -0.33 | 0.75 | 1.22 | 7 | P | P |
| 74.6 | theme | confabulated | confabulated | 3 | -0.43 | 1.09 | 0.48 | 7 | P | P |
| 74.14 | detail | entangled | unresolved |  | 1.33 | 3.61 | -0.44 | 5 | . | P |
| 74.15 | theme | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 75.5 | theme | grounded | unresolved |  | 5.66 | 11.91 | 3.78 | 5 | P | . |
| 75.7 | theme | confabulated | unresolved |  | 0.10 | 1.25 | -0.21 | 3 | . | . |
| 75.9 | detail | entangled | unresolved |  | 0.13 |  |  |  | . | . |
| 76.1 | theme | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 76.5 | detail | entangled | unresolved |  | 0.00 |  |  |  | . | . |
| 76.6 | theme | entangled | unresolved |  | 0.04 |  |  |  | . | . |
| 77.0 | theme | grounded | seeded | 3 | 3.20 | 9.36 | -0.83 | 9 | P | P |
| 77.3 | theme | entangled | seeded | 3 | 2.57 | 5.75 | 0.32 | 6 | P | P |
| 77.9 | detail | entangled | unresolved |  | -0.01 |  |  | 0 | . | . |
| 78.21 | detail | confabulated | seeded | 3 | 0.07 | 1.26 | -0.01 | 12 | P | P |
| 79.1 | theme | grounded | confabulated | 3 | 1.08 | 7.63 | -3.93 | 6 | P | P |
| 79.12 | theme | confabulated | seeded | 3 | 0.05 | 1.13 | 0.02 | 7 | P | P |
| 79.13 | detail | entangled | seeded | 3 | 0.87 | 2.82 | -0.24 | 8 | P | P |
| 80.1 | theme | grounded | seeded | 3 | 2.41 | 6.15 | -1.01 | 7 | P | P |
| 80.2 | theme | confabulated | confabulated | 3 | -0.22 | 1.31 | -0.36 | 6 | P | P |
| 80.5 | detail | confabulated | seeded | 3 | 0.03 | 0.88 | 0.12 | 6 | P | P |
| 81.3 | theme | confabulated | confabulated | 3 | 0.13 | 1.16 | 1.17 | 19 | P | P |
| 81.8 | detail | grounded | seeded | 3 | 14.10 | 19.68 | 0.28 | 6 | P | P |
| 81.9 | theme | confabulated | seeded | 3 | 0.02 | 1.12 | 0.01 | 6 | P | P |
| 82.7 | theme | confabulated | unresolved |  | -0.00 | 1.03 | -0.13 | 1 | P | . |
| 82.13 | theme | entangled | seeded | 3 | 0.18 | 1.56 | 0.27 | 15 | P | P |
| 82.15 | detail | confabulated | confabulated | 3 | 0.10 | 1.89 | -0.16 | 6 | P | P |
| 83.1 | theme | confabulated | entangled | 2 | 0.08 | -0.05 | 2.40 | 6 | P | P |
| 83.2 | detail | grounded | seeded | 3 | 2.11 | 4.53 | -1.17 | 5 | P | P |
| 83.5 | theme | entangled | unresolved |  | 1.31 | 4.58 | 0.02 | 5 | P | . |
| 84.0 | theme | entangled | confabulated | 3 | 0.12 | 2.39 | 1.67 | 9 | P | P |
| 84.6 | theme | grounded | unresolved |  | 4.18 | 12.28 | -0.29 | 7 | . | P |
| 84.15 | detail | grounded | seeded | 3 | 1.86 | 7.26 | -1.24 | 6 | P | P |
| 85.3 | theme | confabulated | confabulated | 3 | -3.59 | -1.68 | 0.97 | 7 | P | P |
| 85.4 | theme | confabulated | confabulated | 3 | -0.57 | 0.33 | 1.18 | 6 | P | P |
| 85.14 | detail | entangled | unresolved |  | 0.45 | 1.27 | 0.57 | 8 | . | P |
| 86.11 | detail | grounded | unresolved |  | 5.46 |  |  | 0 | . | . |
| 86.12 | theme | grounded | unresolved |  | 6.15 | 10.52 | -0.58 | 5 | . | . |
| 86.17 | theme | confabulated | confabulated | 3 | -0.47 | 1.37 | 0.88 | 9 | P | P |
| 87.3 | theme | confabulated | confabulated | 3 | -1.94 | -0.77 | 0.65 | 5 | P | P |
| 87.6 | theme | confabulated | confabulated | 3 | -0.11 | 1.02 | 0.10 | 5 | . | P |
| 87.9 | detail | grounded | seeded | 3 | 6.63 | 11.87 | -3.23 | 6 | P | P |
| 88.1 | theme | confabulated | confabulated | 3 | -0.02 | 0.15 | 1.30 | 5 | P | P |
| 88.7 | theme | entangled | unresolved |  | 0.66 | 3.82 | 0.09 | 21 | . | P |
| 88.13 | detail | grounded | unresolved |  | 3.73 | 11.45 | -0.75 | 5 | P | . |
| 89.0 | theme | grounded | grounded | 2 | 7.16 | 6.03 | 6.11 | 8 | P | P |
| 89.1 | theme | confabulated | confabulated | 3 | -0.72 | 0.23 | 0.29 | 7 | . | P |
| 89.17 | detail | entangled | unresolved |  | 0.10 | 1.42 | -0.03 | 5 | P | . |
| 90.0 | theme | confabulated | confabulated | 3 | -0.06 | 3.64 | -1.35 | 7 | P | P |
| 90.9 | theme | entangled | unresolved |  | 0.13 | 1.51 | -0.25 | 6 | . | P |
| 90.11 | detail | entangled | seeded | 3 | 0.13 | 4.50 | -1.70 | 5 | P | P |
| 91.1 | detail | entangled | entangled | 2 | 3.87 | 3.72 | 1.89 | 7 | P | P |
| 91.2 | theme | confabulated | unresolved |  | -0.23 | 0.33 | 0.62 | 5 | P | . |
| 91.3 | theme | confabulated | confabulated | 3 | -0.62 | -0.65 | 2.65 | 6 | P | P |
| 92.1 | theme | entangled | unresolved |  | 1.57 | 4.38 | 0.35 | 5 | P | . |
| 92.4 | theme | entangled | unresolved |  | 0.88 |  |  |  | . | . |
| 92.15 | detail | entangled | unresolved |  | 0.71 | 2.06 | 0.07 | 5 | . | P |
| 93.3 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 93.5 | theme | confabulated | confabulated | 3 | -0.29 | 2.01 | -0.81 | 8 | P | P |
| 93.12 | detail | grounded | seeded | 3 | 1.59 | 4.29 | -0.15 | 6 | P | P |
| 94.1 | theme | entangled | seeded | 3 | 0.12 | 2.82 | -1.28 | 5 | P | P |
| 94.9 | detail | entangled | seeded | 3 | 0.28 | 2.54 | -0.02 | 7 | P | P |
| 94.13 | theme | entangled | seeded | 3 | 0.45 | 2.96 | -0.51 | 6 | P | P |
| 95.4 | detail | grounded | unresolved |  | 2.05 | 3.15 | 0.53 | 5 | . | P |
| 95.5 | theme | confabulated | confabulated | 3 | 0.02 | 1.94 | 0.35 | 5 | P | P |
| 95.11 | theme | entangled | confabulated | 3 | 0.53 | 0.87 | 0.72 | 6 | P | P |
| 96.1 | theme | confabulated | confabulated | 3 | -0.59 | 2.02 | -1.31 | 5 | . | P |
| 96.6 | theme | grounded | unresolved |  | 15.53 | 16.00 | 4.72 | 5 | P | . |
| 96.12 | detail | entangled | seeded | 3 | 0.99 | 5.50 | -1.35 | 6 | P | P |
| 97.0 | theme | grounded | grounded | 2 | 11.88 | 7.66 | 8.38 | 5 | P | P |
| 97.7 | theme | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 97.11 | detail | entangled | seeded | 3 | 2.42 | 6.27 | -1.65 | 6 | P | P |
| 98.5 | theme | confabulated | confabulated | 3 | -0.69 | -4.15 | 7.47 | 5 | P | P |
| 98.13 | theme | entangled | seeded | 3 | 0.09 | 1.16 | 0.05 | 8 | P | P |
| 98.14 | detail | confabulated | confabulated | 3 | -0.18 | 0.88 | 1.36 | 7 | P | P |
| 99.5 | theme | grounded | entangled | 2 | 10.28 | 5.47 | 8.31 | 7 | P | P |
| 99.11 | theme | confabulated | unresolved |  | -0.40 | 0.44 | 2.63 | 5 | P | . |
| 99.12 | detail | confabulated | confabulated | 3 | -1.57 | 0.07 | 0.34 | 5 | P | P |
| 100.0 | theme | grounded | seeded | 3 | 8.82 | 18.44 | -2.83 | 6 | P | P |
| 100.7 | theme | confabulated | confabulated | 3 | -0.07 | 0.99 | -0.07 | 5 | P | P |
| 100.8 | detail | confabulated | confabulated | 3 | -0.35 | 0.93 | -0.23 | 5 | P | P |
| 101.0 | theme | confabulated | seeded | 3 | 0.01 | 1.28 | -0.18 | 5 | P | P |
| 101.9 | theme | confabulated | confabulated | 3 | -0.00 | 1.09 | -0.15 | 5 | P | P |
| 101.12 | detail | entangled | unresolved |  | 0.19 | 0.68 | 0.74 | 11 | . | P |
| 102.7 | detail | entangled | unresolved |  | 0.02 |  |  | 0 | . | . |
| 102.10 | theme | confabulated | confabulated | 3 | -0.05 | 0.87 | 0.19 | 5 | P | P |
| 102.13 | theme | entangled | seeded | 3 | 2.35 | 10.29 | -1.53 | 5 | P | P |
| 103.0 | theme | grounded | seeded | 3 | 7.37 | 12.40 | 0.37 | 7 | P | P |
| 103.5 | theme | confabulated | unresolved |  | -0.00 | 0.90 | 0.00 | 1 | P | . |
| 103.13 | detail | grounded | grounded | 2 | 4.26 | 4.02 | 1.88 | 5 | P | P |
| 104.5 | detail | confabulated | unresolved |  | -0.18 | 0.94 | -0.22 | 1 | P | . |
| 104.8 | theme | entangled | seeded | 3 | 0.14 | 2.62 | -0.97 | 6 | P | P |
| 104.14 | theme | confabulated | confabulated | 3 | -1.68 | -0.55 | 0.34 | 5 | . | P |
| 105.2 | detail | entangled | seeded | 3 | 2.59 | 6.14 | -0.67 | 7 | P | P |
| 105.3 | theme | grounded | seeded | 3 | 10.26 | 13.51 | 1.65 | 9 | P | P |
| 105.8 | theme | grounded | seeded | 3 | 2.12 | 5.77 | -1.37 | 7 | P | P |
| 106.0 | theme | confabulated | confabulated | 3 | -0.40 | 1.96 | 0.21 | 8 | P | P |
| 106.5 | detail | confabulated | confabulated | 3 | -1.40 | 0.48 | -0.49 | 6 | P | P |
| 106.10 | theme | grounded | grounded | 2 | 3.78 | 3.49 | 3.23 | 6 | P | P |
| 107.1 | theme | confabulated | confabulated | 3 | -0.22 | 0.65 | 0.95 | 6 | P | P |
| 107.9 | detail | grounded | unresolved |  | 18.65 |  |  |  | . | . |
| 107.10 | theme | confabulated | confabulated | 3 | -0.41 | 0.35 | 0.95 | 9 | P | P |
| 108.2 | theme | entangled | unresolved |  | 0.38 | 4.70 | -0.92 | 6 | . | P |
| 108.6 | theme | grounded | seeded | 3 | 6.63 | 9.14 | 1.37 | 6 | P | P |
| 108.12 | detail | grounded | unresolved |  | 14.77 |  |  |  | . | . |
| 109.0 | theme | grounded | unresolved |  | 19.50 | 19.53 | 3.62 | 17 | . | P |
| 109.4 | theme | entangled | unresolved |  | 0.26 | 2.02 | -0.14 | 9 | . | P |
| 109.6 | detail | grounded | seeded | 3 | 4.98 | 9.18 | -0.90 | 9 | P | P |
| 110.1 | detail | entangled | seeded | 3 | 0.70 | 2.12 | 0.10 | 6 | P | P |
| 110.5 | theme | entangled | unresolved |  | 0.13 | 1.40 | -0.27 | 3 | P | . |
| 110.13 | theme | entangled | unresolved |  | 0.13 |  |  |  | . | . |
| 111.1 | theme | grounded | seeded | 3 | 4.22 | 7.77 | -1.56 | 9 | P | P |
| 111.3 | theme | confabulated | confabulated | 3 | -0.12 | -0.35 | 4.03 | 18 | P | P |
| 111.9 | detail | confabulated | seeded | 3 | 0.11 | 1.55 | -0.14 | 8 | P | P |
| 112.4 | theme | grounded | grounded | 2 | 8.33 | 4.18 | 6.99 | 8 | P | P |
| 112.10 | theme | confabulated | confabulated | 3 | -0.03 | 0.81 | 0.31 | 6 | P | P |
| 112.11 | detail | entangled | seeded | 3 | 0.21 | 1.10 | 0.14 | 5 | P | P |
| 113.4 | detail | grounded | seeded | 3 | 5.37 | 7.67 | 1.54 | 11 | P | P |
| 113.6 | theme | entangled | seeded | 3 | 2.44 | 13.17 | 1.56 | 7 | P | P |
| 113.9 | theme | confabulated | confabulated | 3 | -0.42 | 1.16 | -0.22 | 5 | . | P |
| 114.3 | theme | confabulated | confabulated | 3 | -0.71 | 0.58 | -0.12 | 6 | . | P |
| 114.6 | theme | confabulated | confabulated | 3 | -0.06 | 0.79 | 0.33 | 6 | P | P |
| 114.14 | detail | grounded | seeded | 3 | 2.92 | 3.21 | 1.81 | 6 | P | P |
| 115.2 | theme | entangled | seeded | 3 | 1.01 | 1.65 | 1.07 | 11 | P | P |
| 115.5 | theme | grounded | seeded | 3 | 9.34 | 9.77 | 4.41 | 5 | P | P |
| 115.8 | detail | grounded | unresolved |  | 2.22 | 6.08 | -1.12 | 4 | P | . |
| 116.1 | theme | grounded | seeded | 3 | 6.16 | 6.42 | 1.38 | 6 | P | P |
| 116.14 | detail | confabulated | confabulated | 3 | -0.20 | -0.75 | 4.08 | 5 | P | P |
| 116.17 | theme | entangled | unresolved |  | 0.07 | 3.10 | -0.61 | 6 | . | P |
| 117.4 | detail | confabulated | confabulated | 3 | -1.68 | 1.05 | 0.98 | 9 | P | P |
| 117.5 | theme | grounded | seeded | 3 | 1.71 | 2.59 | 0.35 | 8 | P | P |
| 117.6 | theme | confabulated | confabulated | 3 | -0.32 | 0.57 | 0.39 | 6 | P | P |
| 118.0 | theme | grounded | seeded | 3 | 5.22 | 9.75 | -1.35 | 5 | P | P |
| 118.8 | detail | confabulated | confabulated | 3 | -0.02 | 0.89 | 0.98 | 6 | P | P |
| 118.12 | theme | confabulated | confabulated | 3 | 0.01 | 3.32 | -0.46 | 16 | P | P |
| 119.6 | theme | confabulated | confabulated | 3 | -0.88 | 0.64 | 0.16 | 6 | P | P |
| 119.7 | detail | confabulated | unresolved |  | -0.05 |  |  |  | . | . |
| 119.14 | theme | entangled | unresolved |  | 1.72 | 4.02 | -0.31 | 5 | P | . |
| 120.6 | theme | grounded | seeded | 3 | 1.41 | 9.08 | -4.12 | 12 | P | P |
| 120.8 | theme | grounded | seeded | 3 | 1.42 | 5.21 | -0.03 | 7 | P | P |
| 120.12 | detail | grounded | seeded | 3 | 5.20 | 7.50 | 0.71 | 5 | P | P |
| 121.5 | theme | grounded | seeded | 3 | 8.21 | 8.42 | 1.99 | 5 | P | P |
| 121.8 | theme | grounded | unresolved |  | 4.28 | 3.61 | 3.20 | 5 | . | . |
| 121.13 | detail | confabulated | confabulated | 3 | -0.09 | 0.55 | 1.08 | 6 | P | P |
| 122.1 | theme | grounded | seeded | 3 | 3.25 | 6.83 | -0.47 | 5 | P | P |
| 122.7 | theme | entangled | unresolved |  | 0.03 | 1.13 | 0.15 | 5 | . | P |
| 122.9 | detail | confabulated | confabulated | 3 | -1.89 | 1.27 | -1.53 | 5 | . | P |
| 123.7 | detail | entangled | unresolved |  | 0.08 |  |  | 0 | . | . |
| 123.10 | theme | confabulated | confabulated | 3 | -0.02 | 1.30 | 0.14 | 7 | . | P |
| 123.14 | theme | confabulated | confabulated | 3 | -0.05 | -0.27 | 2.20 | 5 | P | P |
| 124.2 | theme | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 124.3 | detail | grounded | unresolved |  | 12.16 | 15.09 | 0.69 | 8 | P | . |
| 124.10 | theme | entangled | unresolved |  | 0.31 | 3.01 | -0.12 | 5 | . | P |
| 125.6 | theme | confabulated | confabulated | 3 | -2.19 | -1.24 | 0.82 | 13 | P | P |
| 125.7 | theme | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 125.10 | detail | confabulated | confabulated | 3 | -2.06 | -3.21 | 2.74 | 5 | P | P |
| 126.0 | theme | grounded | unresolved |  | 12.90 | 15.59 | -0.53 | 6 | P | . |
| 126.4 | theme | grounded | grounded | 2 | 1.24 | 1.16 | 3.43 | 14 | P | P |
| 126.12 | detail | confabulated | confabulated | 3 | -0.29 | 0.41 | 0.70 | 6 | P | P |
| 127.4 | theme | confabulated | confabulated | 3 | -0.30 | 0.68 | 0.87 | 8 | P | P |
| 127.10 | theme | grounded | unresolved |  | 1.07 | 2.59 | -0.12 | 6 | . | P |
| 127.13 | detail | confabulated | confabulated | 3 | -0.00 | 1.86 | -0.12 | 6 | . | P |
| 128.1 | theme | grounded | unresolved |  | 9.08 | 9.15 | 2.17 | 2 | P | . |
| 128.5 | theme | confabulated | unresolved |  | 0.19 | 6.10 | -3.01 | 4 | P | . |
| 128.6 | detail | entangled | unresolved |  | 1.68 | 1.99 | 0.77 | 4 | P | . |
| 129.4 | theme | confabulated | confabulated | 3 | -0.95 | 0.55 | -0.02 | 8 | . | P |
| 129.5 | theme | entangled | seeded | 3 | 0.49 | 2.05 | 0.28 | 5 | P | P |
| 129.6 | detail | grounded | seeded | 3 | 18.27 | 22.75 | 0.90 | 8 | P | P |
| 130.0 | theme | grounded | grounded | 2 | 6.55 | 4.88 | 4.80 | 6 | P | P |
| 130.3 | detail | grounded | seeded | 3 | 2.08 | 10.31 | -2.24 | 5 | P | P |
| 130.4 | theme | confabulated | unresolved |  | -0.75 | -3.43 | 3.58 | 1 | P | . |
| 131.1 | theme | confabulated | seeded | 3 | 0.09 | 6.33 | -1.74 | 18 | P | P |
| 131.8 | theme | confabulated | unresolved |  | 0.11 |  |  |  | . | . |
| 131.11 | detail | grounded | seeded | 3 | 5.63 | 7.66 | -0.97 | 5 | P | P |
| 132.0 | theme | grounded | seeded | 3 | 1.90 | 11.47 | -3.84 | 6 | P | P |
| 132.3 | theme | entangled | seeded | 3 | 0.31 | 1.57 | 0.11 | 6 | P | P |
| 132.8 | detail | entangled | unresolved |  | 0.06 | 0.04 | 0.91 | 1 | P | . |
| 133.4 | theme | confabulated | confabulated | 3 | -0.10 | 0.38 | 0.98 | 7 | P | P |
| 133.7 | theme | confabulated | confabulated | 3 | -0.29 | 0.74 | 1.09 | 6 | P | P |
| 133.12 | detail | grounded | seeded | 3 | 1.01 | 3.40 | -0.02 | 5 | P | P |
| 134.1 | theme | grounded | unresolved |  | 7.19 | 9.87 | 0.93 | 5 | P | . |
| 134.2 | detail | grounded | unresolved |  | 14.03 | 12.84 | 7.66 | 3 | P | . |
| 134.3 | theme | confabulated | unresolved |  | 0.21 |  |  |  | . | . |
| 135.5 | theme | entangled | seeded | 3 | 0.31 | 2.26 | 0.51 | 7 | P | P |
| 135.9 | theme | confabulated | confabulated | 3 | -0.17 | 1.14 | 0.05 | 11 | P | P |
| 135.14 | detail | entangled | seeded | 3 | 0.15 | 1.37 | 0.03 | 5 | P | P |
| 136.0 | theme | grounded | seeded | 3 | 14.30 | 28.06 | -4.90 | 6 | P | P |
| 136.1 | theme | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 136.11 | detail | confabulated | confabulated | 3 | -0.34 | -1.97 | 3.44 | 5 | P | P |
| 137.0 | theme | grounded | seeded | 3 | 5.93 | 12.38 | 1.02 | 14 | P | P |
| 137.7 | detail | entangled | unresolved |  | 0.70 |  |  |  | . | . |
| 137.12 | theme | confabulated | confabulated | 3 | 0.00 | 1.24 | -0.16 | 7 | P | P |
| 138.0 | theme | confabulated | confabulated | 3 | -0.05 | 0.93 | 0.15 | 6 | P | P |
| 138.3 | theme | confabulated | confabulated | 3 | -0.22 | 1.30 | 0.21 | 8 | P | P |
| 138.5 | detail | confabulated | confabulated | 3 | -0.09 | 0.49 | 0.67 | 5 | P | P |
| 139.4 | theme | entangled | seeded | 3 | 0.80 | 2.07 | 0.97 | 7 | P | P |
| 139.6 | theme | confabulated | confabulated | 3 | -0.10 | 0.40 | 0.84 | 18 | P | P |
| 139.7 | detail | grounded | grounded | 2 | 11.83 | 9.32 | 5.49 | 6 | P | P |
| 140.5 | theme | confabulated | unresolved |  | 0.00 | 0.90 | -0.00 | 1 | . | . |
| 140.7 | detail | confabulated | unresolved |  | -0.15 | -6.89 | 7.64 | 1 | P | . |
| 140.10 | theme | confabulated | confabulated | 3 | -0.00 | 1.02 | -0.02 | 6 | P | P |
| 141.1 | theme | grounded | seeded | 3 | 2.44 | 7.19 | -0.88 | 6 | P | P |
| 141.11 | theme | confabulated | confabulated | 3 | -0.10 | 1.49 | -0.16 | 9 | P | P |
| 141.13 | detail | entangled | unresolved |  | 0.39 | 1.87 | -0.01 | 5 | P | . |
| 142.2 | theme | grounded | unresolved |  | 16.45 | 15.76 | 3.91 | 7 | . | P |
| 142.7 | theme | entangled | seeded | 3 | 0.18 | 2.05 | -0.20 | 6 | P | P |
| 142.15 | detail | entangled | seeded | 3 | 1.01 | 1.75 | 0.51 | 5 | P | P |
| 143.1 | theme | entangled | unresolved |  | 1.85 | 2.79 | 0.57 | 5 | . | . |
| 143.5 | theme | grounded | seeded | 3 | 4.36 | 11.85 | -2.80 | 5 | P | P |
| 143.6 | detail | entangled | seeded | 3 | 0.10 | 2.42 | -0.96 | 7 | P | P |
| 144.4 | theme | confabulated | unresolved |  | -0.34 | 0.56 | 0.99 | 5 | P | . |
| 144.5 | theme | grounded | seeded | 3 | 3.47 | 7.35 | 0.80 | 9 | P | P |
| 144.12 | detail | confabulated | seeded | 3 | 0.03 | 5.52 | 2.94 | 20 | P | P |
| 145.1 | theme | entangled | confabulated | 3 | 0.77 | 1.89 | 0.33 | 5 | P | P |
| 145.2 | theme | grounded | seeded | 3 | 5.01 | 9.05 | -1.11 | 5 | P | P |
| 145.8 | detail | entangled | unresolved |  | 11.93 |  |  | 0 | . | . |
| 146.1 | theme | grounded | seeded | 3 | 4.83 | 9.12 | -0.12 | 9 | P | P |
| 146.2 | theme | entangled | seeded | 3 | 0.83 | 3.46 | 0.02 | 6 | P | P |
| 146.3 | detail | grounded | seeded | 3 | 4.75 | 8.28 | -1.49 | 6 | P | P |
| 147.1 | theme | grounded | unresolved |  | 1.78 | 3.83 | -0.54 | 15 | . | P |
| 147.5 | theme | grounded | seeded | 3 | 1.45 | 1.61 | 0.85 | 5 | P | P |
| 147.11 | detail | entangled | seeded | 3 | 0.00 | 3.00 | -0.62 | 7 | P | P |
| 148.2 | detail | grounded | unresolved |  | 8.75 |  |  |  | . | . |
| 148.5 | theme | confabulated | seeded | 3 | 0.06 | 1.93 | 0.13 | 7 | P | P |
| 148.9 | theme | confabulated | confabulated | 3 | -0.37 | 0.80 | 0.09 | 7 | P | P |
| 149.0 | theme | grounded | seeded | 3 | 4.14 | 4.95 | 1.78 | 5 | P | P |
| 149.5 | theme | grounded | grounded | 2 | 3.87 | 2.91 | 4.07 | 10 | P | P |
| 149.7 | detail | confabulated | confabulated | 3 | -0.01 | 0.82 | 0.25 | 5 | P | P |
| 150.1 | theme | entangled | confabulated | 3 | 0.81 | 5.50 | -2.14 | 6 | P | P |
| 150.4 | detail | confabulated | confabulated | 3 | -0.01 | 1.10 | -0.09 | 6 | P | P |
| 150.18 | theme | entangled | unresolved |  | 0.37 | 2.49 | -0.51 | 9 | . | P |
| 151.0 | theme | grounded | unresolved |  | 16.22 | 8.83 | 10.97 | 2 | P | . |
| 151.3 | theme | entangled | unresolved |  | 0.95 | 2.92 | 0.24 | 5 | P | . |
| 151.9 | detail | grounded | seeded | 3 | 2.95 | 10.30 | -2.76 | 5 | P | P |
| 152.1 | theme | grounded | seeded | 3 | 1.79 | 6.39 | -2.19 | 7 | P | P |
| 152.4 | theme | grounded | seeded | 3 | 1.72 | 7.71 | -2.13 | 8 | P | P |
| 152.10 | detail | entangled | seeded | 3 | 0.27 | 0.46 | 0.90 | 5 | P | P |
| 153.2 | detail | entangled | unresolved |  | 0.95 | 3.12 | 1.08 | 4 | P | . |
| 153.4 | theme | confabulated | unresolved |  | -0.08 | -2.23 | 4.03 | 5 | P | . |
| 153.5 | theme | grounded | unresolved |  | 5.79 | 14.43 | 2.23 | 5 | P | . |
| 154.1 | theme | grounded | seeded | 3 | 1.11 | 3.59 | -1.27 | 6 | P | P |
| 154.14 | theme | confabulated | seeded | 3 | 0.06 | 1.11 | 0.04 | 10 | P | P |
| 154.17 | detail | grounded | seeded | 3 | 2.43 | 7.90 | -4.15 | 5 | P | P |
| 155.6 | detail | entangled | unresolved |  | 0.00 |  |  |  | . | . |
| 155.9 | theme | entangled | seeded | 3 | 0.56 | 2.09 | 0.17 | 5 | P | P |
| 155.11 | theme | entangled | seeded | 3 | 0.07 | 1.02 | 0.06 | 5 | P | P |
| 156.2 | theme | grounded | seeded | 3 | 2.92 | 11.13 | -4.28 | 5 | P | P |
| 156.4 | theme | confabulated | confabulated | 3 | -0.32 | 1.78 | 0.24 | 10 | P | P |
| 156.7 | detail | entangled | unresolved |  | 0.00 | 1.40 | -0.50 | 1 | P | . |
| 157.0 | theme | entangled | seeded | 3 | 0.68 | 4.70 | -1.51 | 5 | P | P |
| 157.1 | detail | entangled | seeded | 3 | 0.24 | 1.05 | 0.64 | 6 | P | P |
| 157.2 | theme | entangled | confabulated | 3 | 0.33 | 1.96 | 0.73 | 5 | P | P |
| 158.4 | theme | grounded | unresolved |  | 4.85 | 9.91 | 3.47 | 5 | . | . |
| 158.10 | theme | confabulated | confabulated | 3 | -0.01 | 1.12 | 0.03 | 5 | . | P |
| 158.13 | detail | entangled | confabulated | 3 | -0.02 | 1.30 | -0.39 | 5 | P | P |
| 159.6 | theme | grounded | seeded | 3 | 1.32 | 5.31 | -0.82 | 6 | P | P |
| 159.12 | detail | grounded | unresolved |  | 4.56 |  |  |  | . | . |
| 159.15 | theme | confabulated | confabulated | 3 | -0.31 | 0.43 | 0.47 | 6 | P | P |
| 160.6 | theme | grounded | grounded | 2 | 1.75 | 0.36 | 3.53 | 10 | P | P |
| 160.7 | detail | confabulated | unresolved |  | 0.00 |  |  |  | . | . |
| 160.8 | theme | confabulated | unresolved |  | 0.00 | 1.00 | 0.08 | 3 | . | . |
| 161.2 | detail | entangled | unresolved |  | 0.83 | 6.85 | -3.77 | 5 | P | . |
| 161.4 | theme | grounded | confabulated | 3 | 1.18 | 3.55 | -0.55 | 5 | P | P |
| 161.5 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 162.0 | theme | grounded | seeded | 3 | 3.21 | 12.57 | -3.44 | 8 | P | P |
| 162.6 | theme | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 162.15 | detail | confabulated | confabulated | 3 | -0.03 | 1.21 | 1.50 | 8 | P | P |
| 163.4 | theme | confabulated | confabulated | 3 | -0.29 | 1.18 | -0.32 | 5 | P | P |
| 163.9 | theme | confabulated | confabulated | 3 | 0.07 | 1.37 | -0.18 | 5 | P | P |
| 163.12 | detail | entangled | seeded | 3 | 0.03 | 1.69 | -0.26 | 6 | P | P |
| 164.1 | theme | entangled | seeded | 3 | 0.38 | 2.37 | -0.11 | 7 | P | P |
| 164.9 | theme | confabulated | unresolved |  | 0.01 |  |  |  | . | . |
| 164.12 | detail | confabulated | confabulated | 3 | -0.67 | 0.43 | 0.07 | 5 | . | P |
| 165.0 | theme | grounded | unresolved |  | 10.15 | 1.09 | 9.96 | 1 | P | . |
| 165.1 | theme | grounded | seeded | 3 | 1.90 | 4.81 | -0.62 | 7 | P | P |
| 165.3 | detail | entangled | seeded | 3 | 2.05 | 5.57 | -0.39 | 7 | P | P |
| 166.0 | theme | grounded | entangled | 2 | 2.31 | 2.09 | 1.21 | 5 | P | P |
| 166.7 | detail | confabulated | confabulated | 3 | -0.22 | 2.72 | -1.34 | 6 | P | P |
| 166.9 | theme | confabulated | unresolved |  | 0.00 | 1.41 | 0.04 | 7 | . | P |
| 167.2 | detail | entangled | confabulated | 3 | 0.10 | 7.18 | -2.67 | 8 | P | P |
| 167.6 | theme | confabulated | confabulated | 3 | -0.80 | 0.23 | 0.57 | 10 | P | P |
| 167.14 | theme | grounded | seeded | 3 | 4.24 | 9.09 | -0.94 | 6 | P | P |
| 168.0 | theme | confabulated | entangled | 2 | 0.03 | -0.26 | 2.51 | 6 | P | P |
| 168.2 | detail | grounded | unresolved |  | 4.53 | 10.13 | -1.54 | 5 | P | . |
| 168.5 | theme | grounded | grounded | 2 | 8.96 | 7.52 | 4.55 | 6 | P | P |
| 169.0 | theme | entangled | seeded | 3 | 0.31 | 2.59 | 0.20 | 6 | P | P |
| 169.4 | detail | grounded | seeded | 3 | 2.18 | 2.85 | 0.63 | 7 | P | P |
| 169.8 | theme | entangled | unresolved |  | 0.03 | 0.92 | 0.29 | 5 | P | . |
| 170.8 | theme | entangled | unresolved |  | 0.16 | 1.42 | -0.03 | 5 | . | P |
| 170.13 | detail | grounded | seeded | 3 | 1.64 | 4.93 | -0.71 | 6 | P | P |
| 170.14 | theme | grounded | unresolved |  | 2.13 | 5.31 | -1.14 | 6 | . | P |
| 171.11 | theme | entangled | unresolved |  | 0.03 |  |  |  | . | . |
| 171.13 | detail | confabulated | confabulated | 3 | -1.23 | 0.58 | -0.18 | 7 | P | P |
| 171.15 | theme | entangled | seeded | 3 | 0.52 | 1.78 | 0.11 | 7 | P | P |
| 172.2 | theme | grounded | unresolved |  | 4.02 | 4.62 | 1.32 | 6 | . | P |
| 172.4 | theme | grounded | unresolved |  | 9.00 |  |  |  | . | . |
| 172.10 | detail | confabulated | confabulated | 3 | -0.05 | 1.02 | 0.53 | 5 | P | P |
| 173.5 | theme | confabulated | confabulated | 3 | -1.15 | 0.07 | 1.21 | 9 | P | P |
| 173.6 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 173.9 | detail | grounded | unresolved |  | 4.10 |  |  |  | . | . |
| 174.7 | theme | entangled | seeded | 3 | 0.07 | 1.25 | -0.00 | 19 | P | P |
| 174.8 | detail | confabulated | unresolved |  | 0.00 |  |  |  | . | . |
| 174.10 | theme | confabulated | seeded | 3 | 0.03 | 1.07 | 0.37 | 5 | P | P |
| 175.1 | theme | confabulated | confabulated | 3 | -0.75 | 0.11 | 0.70 | 5 | P | P |
| 175.8 | theme | entangled | confabulated | 3 | 0.33 | 1.61 | 0.28 | 17 | P | P |
| 175.13 | detail | entangled | unresolved |  | 0.41 | 2.30 | -0.39 | 11 | . | P |
| 176.6 | theme | grounded | seeded | 3 | 1.15 | 4.86 | -1.43 | 8 | P | P |
| 176.10 | theme | entangled | unresolved |  | 0.05 |  |  |  | . | . |
| 176.15 | detail | grounded | entangled | 2 | 3.74 | 2.39 | 2.95 | 6 | P | P |
| 177.0 | theme | grounded | seeded | 3 | 3.61 | 8.43 | -3.15 | 5 | P | P |
| 177.5 | detail | grounded | unresolved |  | 1.31 |  |  | 0 | . | . |
| 177.6 | theme | confabulated | confabulated | 3 | 0.05 | 1.70 | -0.31 | 17 | P | P |
| 178.4 | theme | grounded | seeded | 3 | 3.27 | 6.38 | -0.59 | 8 | P | P |
| 178.10 | theme | confabulated | unresolved |  | -0.07 |  |  |  | . | . |
| 178.11 | detail | confabulated | confabulated | 3 | -0.22 | -0.56 | 1.46 | 5 | P | P |
| 179.4 | theme | entangled | entangled | 2 | 0.41 | -0.84 | 3.01 | 5 | P | P |
| 179.7 | detail | entangled | unresolved |  |  |  |  |  | . | . |
| 179.10 | theme | entangled | seeded | 3 | 0.26 | 1.41 | 0.05 | 6 | P | P |
| 180.2 | theme | entangled | unresolved |  | 2.27 | 7.20 | -0.84 | 5 | P | . |
| 180.10 | theme | confabulated | seeded | 3 | 0.03 | 2.64 | 0.12 | 7 | P | P |
| 180.11 | detail | confabulated | confabulated | 3 | -0.45 | 1.21 | -0.32 | 11 | P | P |
| 181.1 | theme | grounded | seeded | 3 | 0.95 | 2.91 | -0.92 | 5 | P | P |
| 181.2 | detail | confabulated | confabulated | 3 | -0.02 | 2.61 | 0.07 | 6 | P | P |
| 181.4 | theme | grounded | confabulated | 3 | 2.72 | 10.49 | -5.78 | 7 | P | P |
| 182.0 | theme | grounded | unresolved |  | 6.15 | 14.86 | -1.89 | 5 | P | . |
| 182.3 | theme | grounded | grounded | 2 | 9.03 | 7.20 | 7.65 | 12 | P | P |
| 182.8 | detail | grounded | unresolved |  | 5.10 | 2.80 | 5.04 | 4 | P | . |
| 183.0 | theme | grounded | seeded | 3 | 3.93 | 8.98 | -2.90 | 6 | P | P |
| 183.10 | detail | grounded | unresolved |  | 4.88 | 7.03 | 4.16 | 5 | P | . |
| 183.15 | theme | confabulated | confabulated | 3 | -0.11 | 1.70 | 2.31 | 18 | P | P |
| 184.1 | theme | grounded | grounded | 2 | 8.29 | 7.33 | 4.25 | 8 | P | P |
| 184.7 | theme | confabulated | unresolved |  | 0.07 | 1.15 | -0.11 | 6 | . | P |
| 184.11 | detail | grounded | confabulated | 3 | 1.37 | 2.83 | -0.09 | 5 | P | P |
| 185.0 | theme | grounded | grounded | 2 | 8.43 | 8.36 | 8.04 | 7 | P | P |
| 185.2 | theme | grounded | unresolved |  | 7.97 |  |  |  | . | . |
| 185.11 | detail | confabulated | confabulated | 3 | -0.18 | 0.99 | 0.80 | 7 | P | P |
| 186.4 | theme | entangled | seeded | 3 | 2.73 | 5.83 | -1.66 | 8 | P | P |
| 186.6 | detail | entangled | unresolved |  | 0.05 |  |  |  | . | . |
| 186.7 | theme | confabulated | confabulated | 3 | -0.14 | 1.80 | -0.75 | 5 | P | P |
| 187.1 | theme | entangled | unresolved |  | 0.26 | 2.47 | -0.38 | 5 | . | P |
| 187.4 | theme | confabulated | confabulated | 3 | -1.26 | -0.18 | 1.13 | 5 | P | P |
| 187.5 | detail | grounded | seeded | 3 | 12.80 | 17.20 | -0.32 | 8 | P | P |
| 188.2 | detail | grounded | seeded | 3 | 1.50 | 4.58 | -1.75 | 5 | P | P |
| 188.13 | theme | confabulated | confabulated | 3 | -0.02 | 0.96 | 0.00 | 5 | P | P |
| 188.21 | theme | entangled | seeded | 3 | 0.25 | 1.46 | 0.32 | 6 | P | P |
| 189.2 | theme | entangled | confabulated | 3 | 0.39 | 2.18 | 0.18 | 5 | P | P |
| 189.9 | theme | entangled | confabulated | 3 | 0.04 | 1.23 | 0.24 | 8 | P | P |
| 189.11 | detail | confabulated | seeded | 3 | 0.00 | 0.74 | 0.79 | 13 | P | P |
| 190.4 | theme | entangled | seeded | 3 | 2.62 | 4.62 | 2.96 | 7 | P | P |
| 190.6 | theme | confabulated | unresolved |  | -0.00 | 0.94 | -0.04 | 3 | . | . |
| 190.11 | detail | confabulated | confabulated | 3 | -0.71 | 0.41 | 0.53 | 6 | . | P |
| 191.2 | theme | grounded | seeded | 3 | 2.13 | 2.78 | 0.61 | 7 | P | P |
| 191.12 | theme | entangled | seeded | 3 | 0.41 | 3.25 | -0.40 | 6 | P | P |
| 191.14 | detail | entangled | seeded | 3 | 0.09 | 1.03 | 0.18 | 7 | P | P |
| 192.4 | theme | grounded | unresolved |  | 4.65 | 3.79 | 1.77 | 1 | P | . |
| 192.5 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 192.10 | detail | grounded | seeded | 3 | 1.84 | 4.24 | -0.96 | 6 | P | P |
| 193.2 | theme | grounded | confabulated | 3 | 1.27 | 2.87 | 2.13 | 8 | P | P |
| 193.7 | detail | confabulated | entangled | 2 | 0.07 | -1.87 | 3.58 | 10 | P | P |
| 193.8 | theme | confabulated | unresolved |  | -0.17 | 1.08 | -0.27 | 5 | P | . |
| 194.2 | theme | confabulated | unresolved |  | -1.37 | 4.10 | -3.61 | 4 | P | . |
| 194.10 | theme | confabulated | confabulated | 3 | -0.66 | 0.50 | 1.93 | 8 | P | P |
| 194.12 | detail | confabulated | confabulated | 3 | -0.09 | 4.34 | -1.27 | 8 | P | P |
| 195.1 | theme | entangled | unresolved |  | 0.38 | 1.21 | 0.21 | 11 | . | P |
| 195.6 | theme | grounded | seeded | 3 | 6.95 | 8.70 | 1.27 | 7 | P | P |
| 195.13 | detail | confabulated | confabulated | 3 | -0.77 | -0.32 | 0.93 | 5 | P | P |
| 196.4 | theme | confabulated | unresolved |  | -0.00 | 1.15 | -0.25 | 1 | P | . |
| 196.8 | theme | confabulated | confabulated | 3 | -0.03 | 1.36 | 1.59 | 22 | P | P |
| 196.12 | detail | entangled | seeded | 3 | 0.42 | 1.48 | 0.52 | 7 | P | P |
| 197.1 | theme | entangled | unresolved |  | 0.16 | 2.12 | -0.62 | 7 | . | P |
| 197.5 | theme | grounded | unresolved |  | 0.95 | 5.97 | -2.42 | 6 | . | P |
| 197.15 | detail | grounded | seeded | 3 | 4.96 | 8.80 | -0.17 | 8 | P | P |
| 198.3 | theme | grounded | unresolved |  | 1.92 | 4.62 | -0.60 | 5 | . | P |
| 198.4 | theme | grounded | seeded | 3 | 2.03 | 2.37 | 1.30 | 17 | P | P |
| 198.5 | detail | entangled | unresolved |  | 0.36 |  |  | 0 | . | . |
| 199.3 | detail | entangled | seeded | 3 | 1.02 | 9.04 | -4.46 | 8 | P | P |
| 199.8 | theme | confabulated | unresolved |  | 0.00 | 0.69 | 0.21 | 1 | P | . |
| 199.14 | theme | confabulated | confabulated | 3 | -1.13 | -0.22 | 0.98 | 5 | P | P |
| 200.0 | theme | grounded | grounded | 2 | 18.90 | 12.93 | 13.75 | 15 | P | P |
| 200.11 | detail | confabulated | seeded | 3 | 0.02 | 1.21 | -0.15 | 8 | P | P |
| 200.14 | theme | entangled | seeded | 3 | 0.82 | 2.49 | -0.26 | 8 | P | P |
| 201.2 | theme | grounded | unresolved |  | 3.42 | 7.49 | -0.66 | 12 | . | P |
| 201.4 | theme | confabulated | confabulated | 3 | -0.83 | -0.19 | 1.58 | 6 | P | P |
| 201.12 | detail | confabulated | confabulated | 3 | -0.06 | -1.56 | 6.49 | 17 | P | P |
| 202.3 | theme | grounded | seeded | 3 | 7.60 | 12.19 | -1.78 | 8 | P | P |
| 202.9 | detail | grounded | unresolved |  | 10.91 | 5.12 | 6.69 | 1 | P | . |
| 202.10 | theme | confabulated | confabulated | 3 | -0.01 | 1.08 | -0.03 | 6 | . | P |
| 203.1 | theme | confabulated | confabulated | 3 | -0.06 | 1.36 | 0.23 | 5 | P | P |
| 203.5 | theme | entangled | seeded | 3 | 0.47 | 2.35 | -0.39 | 7 | P | P |
| 203.11 | detail | entangled | unresolved |  | 0.28 | 1.91 | -0.05 | 5 | . | P |
| 204.2 | theme | entangled | unresolved |  | 0.01 |  |  | 0 | . | . |
| 204.7 | theme | confabulated | unresolved |  | 0.00 | 0.78 | 0.13 | 1 | P | . |
| 204.15 | detail | entangled | confabulated | 3 | 0.24 | 2.07 | -0.64 | 7 | P | P |
| 205.1 | theme | confabulated | unresolved |  | 0.07 | 2.36 | -0.59 | 2 | P | . |
| 205.3 | theme | confabulated | seeded | 3 | 0.15 | 2.27 | -0.20 | 7 | P | P |
| 205.7 | detail | grounded | unresolved |  | 14.27 | 21.37 | 0.11 | 12 | P | . |
| 206.1 | theme | grounded | unresolved |  | 2.83 | 8.89 | -2.36 | 5 | P | . |
| 206.4 | theme | confabulated | unresolved |  | 0.13 | 2.20 | -0.68 | 5 | . | P |
| 206.16 | detail | confabulated | unresolved |  | -0.52 | -0.16 | 0.76 | 5 | P | . |
| 207.6 | theme | grounded | seeded | 3 | 2.51 | 5.71 | 0.30 | 6 | P | P |
| 207.9 | theme | entangled | confabulated | 3 | 0.33 | 3.47 | -1.49 | 5 | P | P |
| 207.12 | detail | entangled | seeded | 3 | 1.82 | 8.47 | -3.85 | 5 | P | P |
| 208.3 | theme | grounded | entangled | 2 | 3.80 | 2.49 | 2.70 | 6 | P | P |
| 208.8 | theme | entangled | unresolved |  | 0.04 | 1.48 | 0.06 | 5 | . | . |
| 208.9 | detail | confabulated | confabulated | 3 | -0.95 | 0.68 | -0.22 | 8 | P | P |
| 209.3 | detail | confabulated | confabulated | 3 | -0.27 | 2.70 | 1.00 | 23 | P | P |
| 209.5 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 209.15 | theme | entangled | seeded | 3 | 0.11 | 2.04 | -0.78 | 5 | P | P |
| 210.5 | theme | entangled | seeded | 3 | 0.46 | 3.15 | -0.52 | 5 | P | P |
| 210.10 | detail | confabulated | confabulated | 3 | -0.09 | 2.29 | 0.06 | 12 | P | P |
| 210.11 | theme | confabulated | unresolved |  | 0.01 | 1.27 | 0.02 | 5 | . | P |
| 211.3 | theme | entangled | confabulated | 3 | 0.35 | 2.70 | -0.79 | 6 | P | P |
| 211.4 | theme | confabulated | confabulated | 3 | -1.04 | -4.16 | 5.41 | 6 | P | P |
| 211.8 | detail | confabulated | confabulated | 3 | -1.04 | 0.60 | -0.44 | 8 | P | P |
| 212.0 | theme | grounded | seeded | 3 | 9.14 | 10.05 | 2.06 | 5 | P | P |
| 212.5 | theme | grounded | unresolved |  | 12.36 |  |  |  | . | . |
| 212.8 | detail | grounded | seeded | 3 | 6.07 | 21.63 | -11.11 | 9 | P | P |
| 213.1 | detail | grounded | grounded | 2 | 9.79 | 8.16 | 5.38 | 6 | P | P |
| 213.4 | theme | confabulated | unresolved |  | -1.47 | -2.92 | 2.99 | 2 | P | . |
| 213.6 | theme | confabulated | confabulated | 3 | -0.00 | 0.80 | 0.24 | 9 | P | P |
| 214.2 | theme | entangled | unresolved |  | 0.01 |  |  | 0 | . | . |
| 214.5 | theme | entangled | seeded | 3 | 0.81 | 3.06 | 0.24 | 7 | P | P |
| 214.7 | detail | entangled | unresolved |  | 0.29 |  |  |  | . | . |
| 215.0 | theme | grounded | seeded | 3 | 3.10 | 3.38 | 1.99 | 7 | P | P |
| 215.9 | theme | confabulated | seeded | 3 | 0.10 | 1.07 | 0.20 | 8 | P | P |
| 215.10 | detail | confabulated | seeded | 3 | 0.21 | 1.51 | 0.12 | 10 | P | P |
| 216.4 | theme | confabulated | unresolved |  | -0.86 | -0.71 | 1.76 | 5 | P | . |
| 216.11 | detail | confabulated | unresolved |  | -0.10 | 0.96 | -0.09 | 2 | P | . |
| 216.12 | theme | grounded | seeded | 3 | 1.02 | 1.74 | 0.24 | 6 | P | P |
| 217.0 | theme | entangled | seeded | 3 | 0.50 | 3.54 | -1.01 | 7 | P | P |
| 217.10 | theme | confabulated | confabulated | 3 | -0.04 | 0.73 | 0.25 | 5 | P | P |
| 217.12 | detail | entangled | seeded | 3 | 0.55 | 1.22 | 0.25 | 5 | P | P |
| 218.1 | theme | confabulated | confabulated | 3 | -0.10 | 0.58 | 0.53 | 5 | . | P |
| 218.13 | theme | confabulated | confabulated | 3 | -0.03 | 1.18 | -0.14 | 6 | P | P |
| 218.15 | detail | confabulated | seeded | 3 | 0.01 | 0.49 | 0.73 | 6 | P | P |
| 219.0 | theme | grounded | seeded | 3 | 6.51 | 8.31 | 1.58 | 5 | P | P |
| 219.3 | detail | grounded | unresolved |  | 1.58 | 6.85 | -1.45 | 5 | P | . |
| 219.8 | theme | confabulated | confabulated | 3 | -0.05 | 0.99 | 0.01 | 6 | . | P |
| 220.0 | theme | grounded | seeded | 3 | 8.14 | 22.69 | -2.94 | 6 | P | P |
| 220.2 | detail | confabulated | confabulated | 3 | -0.44 | 1.85 | -0.02 | 5 | . | P |
| 220.6 | theme | entangled | unresolved |  | 0.00 | 1.02 | -0.06 | 2 | P | . |
| 221.1 | theme | grounded | seeded | 3 | 4.61 | 4.91 | 1.93 | 6 | P | P |
| 221.4 | detail | entangled | unresolved |  | 0.33 | 2.12 | -0.20 | 5 | P | . |
| 221.5 | theme | confabulated | confabulated | 3 | -1.25 | -0.90 | 1.91 | 6 | P | P |
| 222.0 | theme | entangled | unresolved |  | 0.98 | 3.95 | 0.56 | 5 | P | . |
| 222.5 | theme | entangled | unresolved |  | 0.51 | 2.03 | 0.39 | 5 | . | P |
| 222.8 | detail | grounded | grounded | 2 | 22.84 | 22.40 | 6.10 | 10 | P | P |
| 223.8 | theme | confabulated | unresolved |  | -0.05 | 1.16 | -0.05 | 4 | P | . |
| 223.12 | theme | confabulated | confabulated | 3 | -0.01 | 1.06 | -0.02 | 5 | . | P |
| 223.14 | detail | confabulated | seeded | 3 | 0.03 | 0.99 | 2.75 | 6 | P | P |
| 224.1 | theme | confabulated | confabulated | 3 | -0.08 | 1.12 | 1.22 | 6 | . | P |
| 224.5 | theme | grounded | seeded | 3 | 3.54 | 4.88 | 1.67 | 7 | P | P |
| 224.7 | detail | entangled | unresolved |  | 1.65 | 3.92 | 0.24 | 5 | P | . |
| 225.1 | theme | entangled | seeded | 3 | 1.25 | 5.27 | -1.59 | 6 | P | P |
| 225.8 | theme | entangled | seeded | 3 | 0.05 | 3.95 | -0.19 | 6 | P | P |
| 225.11 | detail | confabulated | confabulated | 3 | -0.20 | 1.04 | 0.10 | 8 | P | P |
| 226.0 | theme | grounded | seeded | 3 | 20.53 | 25.11 | -2.72 | 5 | P | P |
| 226.8 | theme | confabulated | confabulated | 3 | -0.03 | 1.09 | 0.04 | 20 | . | P |
| 226.13 | detail | confabulated | confabulated | 3 | -1.22 | -1.88 | 2.75 | 6 | P | P |
| 227.3 | theme | confabulated | confabulated | 3 | -0.03 | 0.78 | 0.18 | 7 | P | P |
| 227.8 | theme | confabulated | confabulated | 3 | -0.17 | 1.11 | -0.19 | 9 | P | P |
| 227.14 | detail | entangled | seeded | 3 | 0.53 | 1.69 | 0.00 | 9 | P | P |
| 228.5 | theme | confabulated | unresolved |  | -0.44 | -4.21 | 5.49 | 5 | P | . |
| 228.6 | theme | confabulated | confabulated | 3 | -0.17 | 1.04 | -0.02 | 6 | P | P |
| 228.12 | detail | entangled | confabulated | 3 | 0.18 | 1.70 | -0.32 | 7 | P | P |
| 229.2 | theme | entangled | unresolved |  | 0.42 | 3.88 | -0.48 | 4 | P | . |
| 229.6 | detail | entangled | unresolved |  | 0.00 |  |  |  | . | . |
| 229.7 | theme | entangled | unresolved |  | 0.34 |  |  |  | . | . |
| 230.3 | theme | confabulated | confabulated | 3 | -0.69 | 1.95 | 0.32 | 6 | P | P |
| 230.9 | detail | entangled | unresolved |  | 0.27 | 1.06 | 0.36 | 7 | . | P |
| 230.12 | theme | confabulated | confabulated | 3 | -0.19 | -0.21 | 1.13 | 5 | . | P |
| 231.5 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 231.7 | theme | confabulated | confabulated | 3 | -0.02 | 1.02 | -0.14 | 5 | . | P |
| 231.14 | detail | grounded | seeded | 3 | 1.27 | 3.96 | -0.99 | 6 | P | P |
| 232.4 | theme | confabulated | confabulated | 3 | -0.36 | 0.02 | 1.04 | 5 | P | P |
| 232.10 | detail | grounded | unresolved |  | 4.34 |  |  |  | . | . |
| 232.11 | theme | grounded | seeded | 3 | 1.05 | 2.68 | 0.20 | 5 | P | P |
| 233.7 | theme | grounded | entangled | 2 | 1.63 | -2.78 | 6.26 | 5 | P | P |
| 233.9 | theme | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 233.14 | detail | entangled | unresolved |  | 0.53 | 3.85 | -0.93 | 5 | P | . |
| 234.0 | theme | grounded | seeded | 3 | 1.28 | 2.50 | 0.56 | 5 | P | P |
| 234.9 | detail | grounded | unresolved |  | 7.23 | 15.88 | -5.16 | 12 | . | P |
| 234.10 | theme | entangled | seeded | 3 | 0.53 | 5.81 | -2.50 | 10 | P | P |
| 235.0 | theme | grounded | seeded | 3 | 7.25 | 8.88 | 4.31 | 8 | P | P |
| 235.1 | theme | entangled | seeded | 3 | 0.80 | 2.05 | 0.36 | 5 | P | P |
| 235.5 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 236.5 | theme | confabulated | unresolved |  | -0.00 | 0.90 | 0.00 | 1 | P | . |
| 236.7 | theme | entangled | confabulated | 3 | 0.57 | 1.22 | 0.47 | 6 | P | P |
| 236.13 | detail | entangled | entangled | 2 | 5.36 | 5.24 | 3.52 | 9 | P | P |
| 237.2 | theme | confabulated | confabulated | 3 | -1.11 | 1.09 | 0.69 | 8 | P | P |
| 237.8 | theme | confabulated | confabulated | 3 | -0.07 | 0.95 | 0.15 | 5 | P | P |
| 237.13 | detail | grounded | seeded | 3 | 5.38 | 7.91 | -0.86 | 7 | P | P |
| 238.4 | theme | confabulated | confabulated | 3 | -0.47 | -0.21 | 3.85 | 7 | P | P |
| 238.9 | theme | entangled | unresolved |  | 0.14 | 2.31 | 0.08 | 6 | . | P |
| 238.14 | detail | entangled | unresolved |  | 0.41 | 1.12 | 0.52 | 5 | P | . |
| 239.3 | theme | entangled | seeded | 3 | 0.14 | 1.70 | -0.20 | 6 | P | P |
| 239.6 | theme | confabulated | confabulated | 3 | -0.40 | 0.29 | 1.05 | 6 | P | P |
| 239.14 | detail | confabulated | confabulated | 3 | -1.25 | -0.31 | 1.48 | 5 | P | P |
| 240.1 | theme | grounded | seeded | 3 | 4.79 | 6.53 | 0.38 | 6 | P | P |
| 240.4 | detail | grounded | grounded | 2 | 12.19 | 11.81 | 3.93 | 6 | P | P |
| 240.5 | theme | entangled | entangled | 2 | 1.23 | 0.42 | 3.97 | 12 | P | P |
| 241.1 | theme | grounded | seeded | 3 | 6.59 | 8.38 | 2.33 | 6 | P | P |
| 241.2 | theme | grounded | seeded | 3 | 6.48 | 18.92 | -6.33 | 7 | P | P |
| 241.12 | detail | grounded | seeded | 3 | 8.36 | 9.08 | 4.63 | 9 | P | P |
| 242.3 | theme | confabulated | seeded | 3 | 0.04 | 1.31 | 0.06 | 5 | P | P |
| 242.5 | theme | grounded | unresolved |  | 6.20 | 6.34 | 3.42 | 5 | P | . |
| 242.6 | detail | entangled | seeded | 3 | 0.94 | 2.95 | -0.41 | 5 | P | P |
| 243.0 | theme | grounded | grounded | 2 | 3.69 | -0.31 | 5.57 | 6 | P | P |
| 243.5 | theme | confabulated | confabulated | 3 | -0.01 | 1.39 | 0.18 | 22 | P | P |
| 243.14 | detail | grounded | unresolved |  | 1.67 | 6.40 | -2.37 | 5 | . | P |
| 244.9 | detail | confabulated | confabulated | 3 | 0.03 | 0.80 | 0.79 | 21 | P | P |
| 244.10 | theme | entangled | seeded | 3 | 0.05 | 1.36 | 0.06 | 6 | P | P |
| 244.14 | theme | confabulated | confabulated | 3 | -0.43 | 1.90 | -0.67 | 9 | P | P |
| 245.1 | theme | confabulated | confabulated | 3 | -0.60 | 0.91 | -0.19 | 5 | . | P |
| 245.6 | theme | entangled | seeded | 3 | 0.75 | 3.18 | -0.52 | 6 | P | P |
| 245.9 | detail | entangled | unresolved |  | 0.05 | 2.24 | -0.99 | 3 | P | . |
| 246.0 | theme | confabulated | seeded | 3 | 0.08 | 5.05 | -1.24 | 16 | P | P |
| 246.7 | theme | entangled | unresolved |  | 0.06 | 1.09 | -0.12 | 1 | P | . |
| 246.11 | detail | entangled | unresolved |  | 2.19 | 4.58 | 0.85 | 6 | P | . |
| 247.0 | theme | grounded | seeded | 3 | 6.66 | 10.18 | 0.61 | 9 | P | P |
| 247.2 | detail | grounded | seeded | 3 | 5.67 | 14.18 | -4.37 | 5 | P | P |
| 247.6 | theme | entangled | unresolved |  | 0.21 | 1.44 | 0.06 | 7 | . | P |
| 248.4 | theme | entangled | seeded | 3 | 2.02 | 7.26 | -1.08 | 6 | P | P |
| 248.7 | theme | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 248.10 | detail | grounded | grounded | 2 | 7.21 | 3.97 | 5.73 | 6 | P | P |
| 249.3 | theme | grounded | unresolved |  | 18.01 | 23.09 | 5.93 | 5 | . | P |
| 249.7 | theme | entangled | unresolved |  | 0.09 |  |  |  | . | . |
| 249.8 | detail | grounded | unresolved |  | 4.21 | 7.80 | 0.41 | 7 | . | P |
| 250.3 | detail | grounded | unresolved |  | 16.56 | 9.32 | 8.14 | 1 | P | . |
| 250.4 | theme | grounded | seeded | 3 | 5.25 | 11.10 | -1.82 | 7 | P | P |
| 250.5 | theme | entangled | unresolved |  | 0.14 |  |  |  | . | . |
| 251.3 | theme | grounded | seeded | 3 | 4.97 | 6.23 | 3.38 | 9 | P | P |
| 251.5 | theme | confabulated | unresolved |  | -0.02 | 1.17 | -0.12 | 4 | P | . |
| 251.10 | detail | entangled | confabulated | 3 | 0.57 | 4.76 | 0.29 | 6 | P | P |
| 252.0 | theme | entangled | seeded | 3 | 0.34 | 5.36 | -2.26 | 6 | P | P |
| 252.6 | detail | entangled | unresolved |  | 0.00 |  |  |  | . | . |
| 252.11 | theme | entangled | unresolved |  | 0.25 |  |  |  | . | . |
| 253.4 | theme | grounded | grounded | 2 | 4.10 | 2.41 | 3.16 | 8 | P | P |
| 253.11 | theme | confabulated | confabulated | 3 | -0.14 | 0.61 | 0.56 | 6 | P | P |
| 253.13 | detail | grounded | seeded | 3 | 1.31 | 5.80 | -2.83 | 6 | P | P |
| 254.1 | theme | confabulated | confabulated | 3 | -0.07 | 1.73 | 1.87 | 14 | P | P |
| 254.9 | detail | grounded | unresolved |  | 3.18 | 8.01 | 1.31 | 5 | P | . |
| 254.11 | theme | grounded | unresolved |  | 1.41 | 5.42 | 0.50 | 5 | P | . |
| 255.2 | theme | grounded | seeded | 3 | 2.32 | 5.50 | -0.12 | 7 | P | P |
| 255.6 | theme | grounded | seeded | 3 | 1.10 | 6.15 | -3.47 | 7 | P | P |
| 255.8 | detail | entangled | unresolved |  | 0.00 | 2.14 | -1.04 | 2 | P | . |
| 256.5 | theme | grounded | seeded | 3 | 3.12 | 4.90 | 0.09 | 20 | P | P |
| 256.6 | theme | confabulated | confabulated | 3 | -0.49 | 1.04 | -0.33 | 5 | P | P |
| 256.11 | detail | entangled | seeded | 3 | 0.00 | 1.73 | -0.25 | 11 | P | P |
| 257.2 | detail | confabulated | confabulated | 3 | -0.22 | 0.88 | 0.87 | 7 | . | P |
| 257.5 | theme | grounded | seeded | 3 | 1.83 | 6.04 | -1.93 | 5 | P | P |
| 257.7 | theme | entangled | seeded | 3 | 0.13 | 1.40 | 0.25 | 13 | P | P |
| 258.0 | theme | grounded | unresolved |  | 9.29 | 8.58 | 2.81 | 5 | P | . |
| 258.1 | detail | grounded | seeded | 3 | 13.93 | 34.86 | -14.74 | 15 | P | P |
| 258.2 | theme | grounded | grounded | 2 | 6.53 | 5.25 | 3.87 | 8 | P | P |
| 259.1 | theme | confabulated | confabulated | 3 | -0.36 | 0.82 | 0.14 | 8 | . | P |
| 259.2 | theme | confabulated | confabulated | 3 | -0.13 | 1.24 | -0.01 | 6 | . | P |
| 259.4 | detail | entangled | seeded | 3 | 0.34 | 2.01 | -0.33 | 5 | P | P |
| 260.5 | theme | grounded | seeded | 3 | 7.07 | 11.63 | -0.83 | 7 | P | P |
| 260.10 | detail | grounded | entangled | 2 | 3.14 | 2.90 | 4.57 | 13 | P | P |
| 260.13 | theme | entangled | seeded | 3 | 0.23 | 2.38 | -0.56 | 6 | P | P |
| 261.1 | theme | grounded | unresolved |  | 2.15 | 4.59 | -0.51 | 8 | . | P |
| 261.6 | theme | entangled | seeded | 3 | 0.07 | 3.66 | -2.12 | 13 | P | P |
| 261.10 | detail | entangled | unresolved |  | 0.54 | 2.67 | -0.26 | 5 | . | P |
| 262.1 | theme | entangled | confabulated | 3 | 0.00 | 1.41 | 0.06 | 8 | P | P |
| 262.6 | theme | confabulated | confabulated | 3 | -0.09 | 1.29 | 0.03 | 17 | P | P |
| 262.12 | detail | confabulated | confabulated | 3 | -0.42 | 0.74 | -0.13 | 5 | P | P |
| 263.2 | theme | grounded | unresolved |  | 8.07 | 14.19 | -1.39 | 6 | . | P |
| 263.3 | detail | entangled | unresolved |  | 1.19 | 5.22 | 0.17 | 7 | P | . |
| 263.4 | theme | confabulated | confabulated | 3 | -0.15 | 0.91 | 0.01 | 5 | P | P |
| 264.8 | theme | confabulated | unresolved |  | -0.00 | 1.27 | -0.37 | 1 | P | . |
| 264.17 | detail | confabulated | confabulated | 3 | -0.34 | 0.93 | -0.31 | 6 | P | P |
| 264.18 | theme | entangled | confabulated | 3 | 0.81 | 3.68 | -0.31 | 7 | P | P |
| 265.0 | theme | grounded | unresolved |  | 27.28 | 8.86 | 19.32 | 1 | P | . |
| 265.3 | theme | confabulated | unresolved |  | -0.19 |  |  |  | . | . |
| 265.9 | detail | entangled | seeded | 3 | 0.23 | 1.12 | 0.88 | 9 | P | P |
| 266.5 | theme | grounded | seeded | 3 | 2.64 | 6.09 | 1.08 | 6 | P | P |
| 266.7 | theme | confabulated | confabulated | 3 | -0.03 | 0.88 | 0.12 | 6 | . | P |
| 266.8 | detail | grounded | entangled | 2 | 1.59 | 0.67 | 3.33 | 7 | P | P |
| 267.3 | theme | confabulated | confabulated | 3 | -0.11 | 3.85 | -0.99 | 8 | P | P |
| 267.4 | theme | grounded | seeded | 3 | 1.85 | 9.47 | -5.48 | 7 | P | P |
| 267.7 | detail | confabulated | confabulated | 3 | -0.02 | 0.87 | 0.05 | 5 | . | P |
| 268.0 | theme | grounded | unresolved |  | 1.48 |  |  |  | . | . |
| 268.4 | theme | grounded | seeded | 3 | 4.50 | 4.51 | 2.61 | 8 | P | P |
| 268.8 | detail | confabulated | confabulated | 3 | -0.00 | -2.36 | 6.73 | 9 | P | P |
| 269.3 | theme | entangled | seeded | 3 | 1.51 | 4.67 | -1.30 | 6 | P | P |
| 269.4 | theme | confabulated | confabulated | 3 | -0.42 | 0.65 | 1.28 | 5 | P | P |
| 269.10 | detail | entangled | seeded | 3 | 0.55 | 2.95 | -0.71 | 6 | P | P |
| 270.1 | theme | grounded | seeded | 3 | 3.12 | 3.59 | 1.13 | 5 | P | P |
| 270.4 | theme | entangled | seeded | 3 | 1.24 | 3.65 | -0.46 | 6 | P | P |
| 270.10 | detail | grounded | seeded | 3 | 1.29 | 3.85 | -0.13 | 7 | P | P |
| 271.0 | theme | confabulated | unresolved |  | -0.24 | 3.17 | -1.09 | 5 | P | . |
| 271.6 | theme | confabulated | confabulated | 3 | -0.13 | 0.77 | 0.44 | 5 | P | P |
| 271.9 | detail | entangled | seeded | 3 | 2.09 | 3.05 | 0.47 | 5 | P | P |
| 272.0 | theme | grounded | seeded | 3 | 3.70 | 5.41 | 2.05 | 5 | P | P |
| 272.10 | theme | entangled | unresolved |  | 0.00 | 1.08 | -0.03 | 6 | . | P |
| 272.11 | detail | entangled | seeded | 3 | 0.38 | 1.33 | 0.25 | 9 | P | P |
| 273.0 | theme | grounded | seeded | 3 | 1.51 | 10.88 | -3.02 | 6 | P | P |
| 273.5 | theme | entangled | seeded | 3 | 0.97 | 3.15 | -0.49 | 6 | P | P |
| 273.7 | detail | entangled | seeded | 3 | 0.00 | 4.10 | -2.26 | 5 | P | P |
| 274.8 | theme | entangled | unresolved |  | 0.04 | 0.70 | 0.42 | 3 | P | . |
| 274.9 | detail | confabulated | confabulated | 3 | 0.26 | 0.58 | 1.07 | 6 | P | P |
| 274.10 | theme | entangled | unresolved |  | 0.57 |  |  |  | . | . |
| 275.2 | theme | grounded | seeded | 3 | 1.70 | 5.08 | 0.05 | 6 | P | P |
| 275.4 | theme | confabulated | confabulated | 3 | -0.09 | 5.49 | -0.72 | 19 | P | P |
| 275.12 | detail | confabulated | confabulated | 3 | -0.06 | 2.34 | -0.27 | 6 | P | P |
| 276.0 | theme | grounded | seeded | 3 | 2.66 | 6.15 | -1.32 | 9 | P | P |
| 276.4 | theme | confabulated | confabulated | 3 | -0.52 | -0.49 | 3.52 | 8 | . | P |
| 276.11 | detail | grounded | seeded | 3 | 2.70 | 5.05 | 1.06 | 8 | P | P |
| 277.2 | theme | grounded | unresolved |  | 5.67 | 6.18 | 1.42 | 5 | . | P |
| 277.5 | theme | grounded | unresolved |  | 2.67 | 8.33 | -2.92 | 5 | . | . |
| 277.10 | detail | confabulated | confabulated | 3 | -0.12 | 0.78 | 0.26 | 5 | P | P |
| 278.0 | theme | grounded | seeded | 3 | 4.22 | 12.03 | -2.80 | 9 | P | P |
| 278.2 | detail | confabulated | confabulated | 3 | -0.30 | 0.86 | -0.08 | 5 | P | P |
| 278.6 | theme | entangled | unresolved |  | 0.63 | 2.96 | 0.49 | 9 | . | P |
| 279.5 | theme | grounded | seeded | 3 | 2.71 | 7.32 | 0.72 | 11 | P | P |
| 279.12 | theme | entangled | unresolved |  | 0.07 | 0.83 | 0.13 | 1 | P | . |
| 279.13 | detail | grounded | seeded | 3 | 3.05 | 6.42 | -0.31 | 9 | P | P |
| 280.4 | theme | confabulated | confabulated | 3 | -0.72 | -0.01 | 0.93 | 6 | P | P |
| 280.10 | theme | entangled | seeded | 3 | 0.31 | 0.34 | 1.54 | 7 | P | P |
| 280.14 | detail | entangled | confabulated | 3 | 0.13 | 2.71 | -0.76 | 5 | P | P |
| 281.2 | theme | confabulated | confabulated | 3 | 0.12 | 2.20 | -0.64 | 5 | P | P |
| 281.8 | detail | grounded | grounded | 2 | 3.52 | 3.15 | 5.16 | 7 | P | P |
| 281.10 | theme | confabulated | confabulated | 3 | 0.10 | 1.11 | -0.04 | 5 | P | P |
| 282.2 | theme | entangled | unresolved |  | 0.31 | 3.94 | 0.94 | 8 | . | P |
| 282.10 | detail | entangled | unresolved |  | 0.30 |  |  |  | . | . |
| 282.13 | theme | confabulated | unresolved |  | -0.03 | 1.16 | -0.29 | 1 | P | . |
| 283.2 | theme | confabulated | confabulated | 3 | -0.63 | -1.03 | 1.52 | 5 | P | P |
| 283.8 | theme | confabulated | unresolved |  | -0.01 | 0.80 | 0.15 | 2 | P | . |
| 283.16 | detail | entangled | seeded | 3 | 0.53 | 1.97 | 0.23 | 8 | P | P |
| 284.0 | theme | grounded | seeded | 3 | 3.89 | 7.60 | -0.32 | 8 | P | P |
| 284.11 | theme | grounded | grounded | 2 | 5.63 | 4.77 | 2.57 | 5 | P | P |
| 284.12 | detail | entangled | confabulated | 3 | 0.20 | 4.69 | -0.63 | 13 | P | P |
| 285.2 | theme | confabulated | confabulated | 3 | -0.19 | 4.83 | 1.43 | 7 | P | P |
| 285.9 | theme | confabulated | confabulated | 3 | -0.52 | 1.10 | -0.24 | 6 | P | P |
| 285.14 | detail | confabulated | confabulated | 3 | -0.50 | 0.85 | -0.31 | 5 | P | P |
| 286.1 | theme | confabulated | confabulated | 3 | -0.40 | 7.89 | -5.38 | 6 | P | P |
| 286.4 | theme | grounded | grounded | 2 | 20.39 | 15.99 | 6.68 | 6 | P | P |
| 286.6 | detail | confabulated | unresolved |  | -0.01 |  |  |  | . | . |
| 287.8 | theme | entangled | unresolved |  | -0.04 |  |  |  | . | . |
| 287.12 | theme | entangled | unresolved |  | 0.01 | 1.47 | -0.07 | 3 | P | . |
| 287.14 | detail | grounded | unresolved |  | 2.38 | 4.02 | 1.57 | 5 | P | . |
| 288.0 | theme | confabulated | confabulated | 3 | 0.05 | 1.90 | -0.16 | 7 | P | P |
| 288.7 | detail | entangled | seeded | 3 | 0.19 | 5.60 | -0.31 | 6 | P | P |
| 288.8 | theme | confabulated | confabulated | 3 | -0.92 | 0.36 | 0.69 | 7 | . | P |
| 289.0 | theme | entangled | seeded | 3 | 0.59 | 2.52 | -0.22 | 5 | P | P |
| 289.7 | theme | entangled | seeded | 3 | 0.55 | 1.31 | 0.49 | 6 | P | P |
| 289.12 | detail | grounded | seeded | 3 | 2.35 | 3.12 | 1.10 | 9 | P | P |
| 290.2 | theme | entangled | seeded | 3 | 0.75 | 5.14 | -2.29 | 5 | P | P |
| 290.4 | detail | confabulated | confabulated | 3 | -0.39 | 0.48 | 1.23 | 7 | P | P |
| 290.7 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 291.2 | theme | confabulated | confabulated | 3 | -0.78 | 0.08 | 0.42 | 6 | . | P |
| 291.5 | theme | confabulated | entangled | 2 | 0.07 | -0.33 | 1.59 | 5 | P | P |
| 291.9 | detail | grounded | unresolved |  | 1.13 |  |  |  | . | . |
| 292.2 | detail | entangled | unresolved |  | 0.99 | 4.15 | -0.74 | 5 | P | . |
| 292.3 | theme | grounded | seeded | 3 | 1.47 | 3.93 | -0.68 | 5 | P | P |
| 292.13 | theme | confabulated | confabulated | 3 | -0.09 | 0.60 | 1.20 | 5 | P | P |
| 293.2 | detail | confabulated | confabulated | 3 | -0.99 | -1.30 | 3.40 | 5 | P | P |
| 293.9 | theme | entangled | seeded | 3 | 0.30 | 2.31 | -0.73 | 6 | P | P |
| 293.11 | theme | confabulated | confabulated | 3 | -1.87 | -3.33 | 4.76 | 5 | P | P |
| 294.4 | theme | entangled | seeded | 3 | 0.81 | 3.02 | 1.07 | 7 | P | P |
| 294.5 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 294.12 | theme | confabulated | confabulated | 3 | -1.14 | 0.79 | -0.44 | 6 | P | P |
| 295.0 | theme | grounded | grounded | 2 | 9.07 | 3.83 | 8.72 | 7 | P | P |
| 295.1 | theme | grounded | seeded | 3 | 1.38 | 3.50 | -0.44 | 6 | P | P |
| 295.10 | detail | entangled | seeded | 3 | 0.25 | 1.93 | -0.17 | 6 | P | P |
| 296.0 | theme | grounded | grounded | 2 | 10.87 | 7.20 | 7.62 | 7 | P | P |
| 296.5 | detail | confabulated | confabulated | 3 | -0.00 | 4.00 | 0.82 | 23 | P | P |
| 296.6 | theme | confabulated | seeded | 3 | 0.06 | 1.10 | 0.08 | 17 | P | P |
| 297.0 | theme | grounded | unresolved |  | 2.29 | 8.04 | -3.40 | 7 | P | . |
| 297.11 | theme | confabulated | seeded | 3 | 0.11 | 1.23 | -0.06 | 6 | P | P |
| 297.13 | detail | confabulated | confabulated | 3 | -0.75 | -1.92 | 3.26 | 6 | P | P |
| 298.0 | theme | confabulated | confabulated | 3 | -0.06 | 7.76 | -2.65 | 6 | P | P |
| 298.1 | theme | entangled | seeded | 3 | 0.96 | 4.94 | 2.69 | 9 | P | P |
| 298.8 | detail | grounded | unresolved |  | 7.98 | 3.52 | 8.01 | 5 | P | . |
| 299.0 | theme | grounded | seeded | 3 | 2.20 | 6.36 | -1.01 | 5 | P | P |
| 299.5 | theme | confabulated | unresolved |  | -0.14 | 1.30 | 0.03 | 3 | P | . |
| 299.10 | detail | grounded | seeded | 3 | 2.96 | 5.80 | -0.59 | 5 | P | P |
| 300.0 | theme | grounded | grounded | 2 | 10.63 | 7.41 | 6.19 | 8 | P | P |
| 300.11 | theme | entangled | seeded | 3 | 0.06 | 1.07 | 0.05 | 5 | P | P |
| 300.13 | detail | grounded | grounded | 2 | 9.41 | 6.34 | 6.89 | 7 | P | P |
| 301.4 | theme | grounded | seeded | 3 | 3.38 | 4.67 | 1.01 | 7 | P | P |
| 301.5 | theme | entangled | seeded | 3 | 1.81 | 3.75 | -0.70 | 5 | P | P |
| 301.6 | detail | entangled | unresolved |  | 0.00 | -5.10 | 6.00 | 1 | P | . |
| 302.4 | theme | grounded | unresolved |  | 1.58 | 6.84 | -1.93 | 5 | P | . |
| 302.5 | theme | entangled | unresolved |  | 0.30 | 3.51 | -0.24 | 7 | . | P |
| 302.8 | detail | confabulated | confabulated | 3 | -1.30 | -3.75 | 7.72 | 7 | P | P |
| 303.1 | theme | grounded | unresolved |  | 0.90 | 5.60 | -1.54 | 7 | . | P |
| 303.5 | theme | confabulated | confabulated | 3 | -0.61 | -1.06 | 2.06 | 6 | P | P |
| 303.16 | detail | confabulated | unresolved |  | 0.08 | 6.78 | -0.62 | 19 | . | P |
| 304.0 | theme | grounded | seeded | 3 | 3.44 | 8.17 | -1.20 | 5 | P | P |
| 304.10 | detail | confabulated | confabulated | 3 | -0.18 | 1.02 | -0.18 | 7 | P | P |
| 304.13 | theme | confabulated | confabulated | 3 | -0.02 | 1.55 | -0.27 | 6 | . | P |
| 305.3 | detail | entangled | unresolved |  | 0.30 | 1.15 | 1.02 | 5 | P | . |
| 305.8 | theme | entangled | unresolved |  | 0.65 | 2.22 | 0.24 | 6 | . | P |
| 305.13 | theme | grounded | grounded | 2 | 3.05 | 2.13 | 2.05 | 5 | P | P |
| 306.0 | theme | entangled | confabulated | 3 | 0.64 | 1.02 | 1.37 | 8 | P | P |
| 306.1 | theme | confabulated | confabulated | 3 | -0.03 | 0.60 | 1.25 | 5 | . | P |
| 306.9 | detail | entangled | unresolved |  | 0.50 |  |  |  | . | . |
| 307.0 | theme | grounded | seeded | 3 | 13.46 | 22.58 | -3.05 | 12 | P | P |
| 307.7 | theme | confabulated | unresolved |  | -0.04 |  |  |  | . | . |
| 307.8 | detail | confabulated | confabulated | 3 | -0.00 | 0.93 | 0.13 | 6 | P | P |
| 308.4 | theme | grounded | seeded | 3 | 4.23 | 7.20 | -1.39 | 5 | P | P |
| 308.5 | detail | entangled | seeded | 3 | 0.39 | 2.37 | -0.18 | 8 | P | P |
| 308.12 | theme | confabulated | unresolved |  | 0.00 | 0.94 | 0.01 | 6 | . | P |
| 309.5 | theme | entangled | seeded | 3 | 0.33 | 2.51 | 0.07 | 6 | P | P |
| 309.7 | detail | grounded | seeded | 3 | 1.07 | 1.28 | 1.33 | 13 | P | P |
| 309.8 | theme | confabulated | confabulated | 3 | -0.05 | 1.10 | 0.04 | 5 | . | P |
| 310.0 | theme | grounded | seeded | 3 | 10.62 | 25.76 | -11.42 | 6 | P | P |
| 310.2 | detail | entangled | unresolved |  | 2.22 |  |  | 0 | . | . |
| 310.3 | detail | entangled | seeded | 3 | 2.99 | 20.54 | -8.17 | 10 | P | P |
| 311.1 | theme | grounded | unresolved |  | 5.18 | 12.78 | -0.66 | 12 | . | P |
| 311.7 | theme | confabulated | confabulated | 3 | -0.36 | 0.48 | 0.82 | 7 | P | P |
| 311.8 | detail | entangled | seeded | 3 | 0.76 | 4.63 | -2.05 | 8 | P | P |
| 312.0 | theme | grounded | unresolved |  | 1.70 | 9.32 | -6.73 | 1 | P | . |
| 312.1 | theme | grounded | unresolved |  | 3.28 |  |  |  | . | . |
| 312.8 | detail | confabulated | confabulated | 3 | -0.00 | 1.42 | -0.24 | 6 | P | P |
| 313.0 | theme | grounded | seeded | 3 | 3.89 | 13.51 | -4.57 | 6 | P | P |
| 313.7 | theme | confabulated | confabulated | 3 | -0.05 | 0.85 | 0.05 | 9 | P | P |
| 313.8 | detail | grounded | grounded | 2 | 8.24 | 2.97 | 7.38 | 5 | P | P |
| 314.3 | theme | entangled | unresolved |  | 0.86 | 1.08 | 1.20 | 2 | P | . |
| 314.9 | theme | entangled | unresolved |  | 0.00 | 1.19 | 0.03 | 12 | . | P |
| 314.13 | detail | confabulated | confabulated | 3 | -0.00 | 0.69 | 0.54 | 5 | P | P |
| 315.0 | theme | confabulated | confabulated | 3 | 0.03 | 1.73 | 0.02 | 7 | P | P |
| 315.1 | theme | grounded | seeded | 3 | 1.41 | 5.13 | -1.45 | 9 | P | P |
| 315.6 | detail | entangled | unresolved |  | 0.01 |  |  |  | . | . |
| 316.1 | theme | grounded | unresolved |  | 2.18 | 6.08 | -0.09 | 5 | P | . |
| 316.3 | detail | entangled | seeded | 3 | 2.23 | 3.94 | -0.08 | 5 | P | P |
| 316.9 | theme | entangled | unresolved |  | 0.81 | 2.91 | -0.53 | 5 | . | P |
| 317.2 | theme | entangled | unresolved |  | 2.45 | 7.50 | -0.49 | 5 | P | . |
| 317.3 | theme | grounded | grounded | 2 | 9.43 | 7.48 | 7.66 | 6 | P | P |
| 317.5 | detail | grounded | unresolved |  | 2.48 |  |  | 0 | . | . |
| 318.2 | theme | confabulated | confabulated | 3 | -0.09 | 0.54 | 2.78 | 13 | P | P |
| 318.5 | theme | entangled | unresolved |  | 0.56 |  |  |  | . | . |
| 318.10 | detail | grounded | seeded | 3 | 5.14 | 5.81 | 4.02 | 14 | P | P |
| 319.2 | detail | grounded | unresolved |  | 17.47 | 21.15 | 0.34 | 5 | P | . |
| 319.3 | theme | grounded | entangled | 2 | 3.34 | 2.10 | 5.98 | 13 | P | P |
| 319.4 | theme | grounded | seeded | 3 | 20.28 | 23.30 | 4.48 | 6 | P | P |
| 320.2 | theme | grounded | unresolved |  | 1.04 | 1.02 | 1.20 | 4 | P | . |
| 320.4 | detail | grounded | unresolved |  | 14.10 | 7.04 | 10.39 | 2 | P | . |
| 320.6 | theme | entangled | seeded | 3 | 0.70 | 3.37 | -0.68 | 6 | P | P |
| 321.7 | theme | entangled | entangled | 2 | 1.19 | 0.99 | 1.47 | 12 | P | P |
| 321.8 | detail | entangled | unresolved |  | 0.20 | 1.13 | 0.90 | 4 | P | . |
| 321.14 | theme | grounded | seeded | 3 | 4.96 | 8.01 | 0.60 | 6 | P | P |
| 322.5 | detail | entangled | unresolved |  | 0.00 |  |  |  | . | . |
| 322.6 | theme | entangled | unresolved |  | 0.03 | 1.17 | 0.08 | 5 | . | . |
| 322.8 | theme | entangled | seeded | 3 | 0.19 | 2.34 | 0.09 | 11 | P | P |
| 323.4 | theme | confabulated | confabulated | 3 | -0.01 | 1.09 | 0.02 | 6 | P | P |
| 323.6 | theme | entangled | confabulated | 3 | 0.42 | 1.88 | -0.02 | 6 | P | P |
| 323.7 | detail | grounded | confabulated | 3 | 1.93 | 3.73 | 1.19 | 10 | P | P |
| 324.6 | theme | confabulated | confabulated | 3 | -0.14 | -0.15 | 3.57 | 11 | P | P |
| 324.9 | theme | confabulated | confabulated | 3 | -0.04 | 0.67 | 0.36 | 5 | P | P |
| 324.10 | detail | grounded | grounded | 2 | 5.17 | 3.68 | 2.91 | 5 | P | P |
| 325.0 | theme | confabulated | confabulated | 3 | -0.01 | 4.33 | -0.21 | 11 | P | P |
| 325.10 | detail | entangled | seeded | 3 | 0.04 | 3.53 | 0.52 | 16 | P | P |
| 325.17 | theme | confabulated | confabulated | 3 | -0.87 | -0.39 | 0.60 | 5 | P | P |
| 326.1 | theme | confabulated | unresolved |  | -0.53 | -0.53 | 1.61 | 5 | P | . |
| 326.4 | theme | entangled | unresolved |  | 0.68 | 3.42 | -0.96 | 6 | . | P |
| 326.15 | detail | entangled | seeded | 3 | 1.49 | 4.24 | -1.08 | 5 | P | P |
| 327.0 | theme | grounded | unresolved |  | 4.81 | 5.12 | 4.10 | 5 | P | . |
| 327.3 | theme | entangled | unresolved |  | 0.72 | 0.77 | 4.44 | 5 | P | . |
| 327.8 | detail | entangled | seeded | 3 | 0.59 | 4.79 | -0.98 | 8 | P | P |
| 328.1 | theme | grounded | seeded | 3 | 1.72 | 4.82 | -1.18 | 7 | P | P |
| 328.7 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 328.10 | theme | confabulated | confabulated | 3 | -0.08 | 0.53 | 0.43 | 5 | P | P |
| 329.2 | theme | grounded | grounded | 2 | 5.01 | 2.02 | 7.86 | 5 | P | P |
| 329.4 | theme | entangled | unresolved |  | 0.22 |  |  |  | . | . |
| 329.7 | detail | confabulated | unresolved |  | 0.02 | 1.00 | -0.02 | 6 | . | P |
| 330.1 | theme | grounded | seeded | 3 | 6.48 | 9.66 | 1.41 | 7 | P | P |
| 330.5 | theme | confabulated | unresolved |  | -0.30 | 0.04 | 0.90 | 2 | P | . |
| 330.11 | detail | grounded | seeded | 3 | 2.44 | 3.49 | 0.42 | 6 | P | P |
| 331.0 | theme | grounded | confabulated | 3 | 3.42 | 4.32 | 2.36 | 9 | P | P |
| 331.6 | theme | confabulated | unresolved |  | 0.00 | 1.01 | 0.00 | 3 | P | . |
| 331.7 | detail | entangled | unresolved |  | 0.00 | 3.52 | -1.89 | 2 | P | . |
| 332.2 | theme | confabulated | unresolved |  | -0.01 | 2.81 | -1.92 | 1 | . | . |
| 332.4 | theme | confabulated | confabulated | 3 | -1.35 | -0.05 | 1.82 | 5 | P | P |
| 332.11 | detail | entangled | seeded | 3 | 2.00 | 8.24 | -3.10 | 10 | P | P |
| 333.0 | theme | entangled | confabulated | 3 | 0.16 | 2.42 | 0.33 | 5 | P | P |
| 333.4 | theme | entangled | seeded | 3 | 1.47 | 3.70 | 0.42 | 10 | P | P |
| 333.16 | detail | grounded | seeded | 3 | 3.29 | 5.94 | -0.39 | 10 | P | P |
| 334.0 | theme | grounded | unresolved |  | 1.05 | 4.82 | -0.95 | 6 | P | . |
| 334.2 | detail | grounded | seeded | 3 | 2.44 | 5.35 | 0.17 | 5 | P | P |
| 334.4 | theme | grounded | seeded | 3 | 3.11 | 11.15 | -1.36 | 6 | P | P |
| 335.1 | theme | entangled | unresolved |  | 0.36 | 4.51 | -1.22 | 5 | . | . |
| 335.4 | theme | confabulated | confabulated | 3 | -1.42 | 0.88 | -1.16 | 7 | P | P |
| 335.12 | detail | grounded | unresolved |  | 1.96 | 4.97 | -1.89 | 5 | . | P |
| 336.3 | theme | confabulated | confabulated | 3 | -0.10 | 0.52 | 0.83 | 5 | P | P |
| 336.5 | theme | grounded | unresolved |  | 10.11 | 6.75 | 4.25 | 1 | P | . |
| 336.9 | detail | grounded | seeded | 3 | 2.37 | 8.48 | -3.41 | 9 | P | P |
| 337.0 | theme | grounded | grounded | 2 | 2.69 | 1.64 | 4.74 | 7 | P | P |
| 337.2 | theme | confabulated | confabulated | 3 | -1.28 | 0.37 | 0.66 | 5 | P | P |
| 337.12 | detail | grounded | confabulated | 3 | 1.70 | 5.31 | -1.31 | 7 | P | P |
| 338.1 | theme | grounded | seeded | 3 | 2.23 | 3.81 | 0.65 | 8 | P | P |
| 338.4 | theme | grounded | entangled | 2 | 3.54 | 3.49 | 1.95 | 5 | P | P |
| 338.6 | detail | entangled | entangled | 2 | 0.95 | 0.70 | 1.25 | 5 | P | P |
| 339.3 | theme | confabulated | confabulated | 3 | -0.54 | -0.46 | 1.59 | 6 | P | P |
| 339.10 | theme | entangled | unresolved |  | 0.57 | 2.60 | 1.12 | 6 | . | P |
| 339.13 | detail | grounded | grounded | 2 | 4.90 | 4.13 | 2.96 | 7 | P | P |
| 340.2 | theme | confabulated | confabulated | 3 | -0.03 | 1.10 | 0.21 | 8 | . | P |
| 340.6 | theme | confabulated | entangled | 2 | 0.14 | -0.41 | 4.57 | 17 | P | P |
| 340.8 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 341.1 | theme | entangled | seeded | 3 | 0.24 | 1.49 | 0.04 | 5 | P | P |
| 341.5 | detail | grounded | unresolved |  | 1.81 | 6.80 | -0.79 | 5 | P | . |
| 341.8 | theme | entangled | unresolved |  | 0.74 | 2.27 | 0.56 | 5 | P | . |
| 342.0 | theme | confabulated | confabulated | 3 | -0.16 | 1.39 | 0.47 | 6 | P | P |
| 342.1 | theme | entangled | seeded | 3 | 1.44 | 3.19 | 0.82 | 9 | P | P |
| 342.3 | detail | grounded | grounded | 2 | 4.59 | 2.91 | 4.10 | 7 | P | P |
| 343.0 | theme | grounded | grounded | 2 | 25.54 | 14.84 | 14.03 | 5 | P | P |
| 343.4 | theme | entangled | seeded | 3 | 0.72 | 2.84 | 0.37 | 6 | P | P |
| 343.11 | detail | entangled | unresolved |  | 0.44 |  |  |  | . | . |
| 344.2 | theme | grounded | seeded | 3 | 1.01 | 4.32 | -1.67 | 6 | P | P |
| 344.3 | detail | entangled | unresolved |  | 0.60 | 0.88 | 2.70 | 7 | P | . |
| 344.10 | theme | confabulated | confabulated | 3 | -0.27 | 0.94 | 0.27 | 5 | P | P |
| 345.3 | theme | grounded | seeded | 3 | 5.57 | 8.79 | 0.42 | 6 | P | P |
| 345.4 | theme | confabulated | confabulated | 3 | -1.64 | -1.16 | 1.73 | 5 | P | P |
| 345.9 | detail | confabulated | confabulated | 3 | -0.00 | 0.82 | 0.45 | 6 | P | P |
| 346.4 | theme | entangled | unresolved |  | 0.03 | 0.94 | 0.13 | 5 | . | P |
| 346.6 | theme | grounded | seeded | 3 | 1.24 | 3.28 | -0.06 | 8 | P | P |
| 346.9 | detail | confabulated | confabulated | 3 | -0.03 | 0.92 | 0.12 | 5 | . | P |
| 347.3 | theme | grounded | grounded | 2 | 5.83 | 2.19 | 5.31 | 7 | P | P |
| 347.5 | theme | confabulated | confabulated | 3 | -0.45 | 0.81 | 0.19 | 5 | P | P |
| 347.6 | detail | grounded | seeded | 3 | 15.34 | 29.30 | -6.07 | 8 | P | P |
| 348.3 | theme | grounded | unresolved |  | 7.30 | 9.46 | 2.01 | 5 | P | . |
| 348.5 | detail | confabulated | unresolved |  | 0.00 |  |  | 0 | . | . |
| 348.8 | theme | confabulated | unresolved |  | -0.20 | 0.93 | 0.12 | 5 | . | . |
| 349.6 | theme | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 349.9 | detail | grounded | seeded | 3 | 8.07 | 11.76 | -0.47 | 7 | P | P |
| 349.10 | theme | entangled | seeded | 3 | 0.43 | 1.85 | -0.15 | 8 | P | P |
| 350.0 | theme | entangled | seeded | 3 | 0.53 | 7.35 | -4.41 | 7 | P | P |
| 350.1 | theme | grounded | seeded | 3 | 4.09 | 5.02 | 0.81 | 6 | P | P |
| 351.2 | theme | grounded | seeded | 3 | 4.92 | 8.69 | 0.47 | 8 | P | P |
| 351.10 | detail | entangled | seeded | 3 | 2.15 | 4.27 | 0.69 | 8 | P | P |
| 351.14 | theme | entangled | seeded | 3 | 0.27 | 4.64 | -1.18 | 6 | P | P |
| 352.2 | theme | grounded | seeded | 3 | 3.49 | 5.38 | 0.12 | 5 | P | P |
| 352.5 | theme | entangled | seeded | 3 | 1.86 | 4.83 | -0.65 | 5 | P | P |
| 352.8 | detail | confabulated | confabulated | 3 | -3.19 | -3.94 | 2.93 | 7 | P | P |
| 353.1 | theme | confabulated | confabulated | 3 | -0.14 | 2.00 | -0.82 | 10 | P | P |
| 353.4 | theme | confabulated | confabulated | 3 | -0.83 | -1.65 | 2.78 | 5 | P | P |
| 353.10 | detail | confabulated | confabulated | 3 | -0.12 | 0.58 | 0.66 | 5 | P | P |
| 354.0 | theme | entangled | unresolved |  | 0.44 | 3.11 | 0.21 | 7 | . | P |
| 354.7 | theme | confabulated | unresolved |  | 0.00 | 1.06 | -0.10 | 5 | . | P |
| 354.13 | detail | confabulated | seeded | 3 | 0.03 | 1.18 | 0.50 | 7 | P | P |
| 355.8 | theme | confabulated | seeded | 3 | 0.01 | 1.56 | -0.10 | 11 | P | P |
| 355.9 | detail | confabulated | confabulated | 3 | -0.29 | 0.50 | 1.02 | 12 | P | P |
| 355.12 | theme | entangled | seeded | 3 | 0.97 | 3.77 | -1.52 | 5 | P | P |
| 356.2 | theme | entangled | seeded | 3 | 0.09 | 2.64 | -0.96 | 5 | P | P |
| 356.9 | theme | grounded | grounded | 2 | 3.38 | 2.87 | 2.51 | 5 | P | P |
| 356.11 | detail | confabulated | confabulated | 3 | -1.24 | 1.92 | -1.42 | 7 | P | P |
| 357.3 | theme | grounded | seeded | 3 | 4.36 | 5.88 | 1.34 | 5 | P | P |
| 357.5 | theme | entangled | unresolved |  | 0.54 | 1.98 | 0.25 | 5 | P | . |
| 357.10 | detail | confabulated | confabulated | 3 | -1.43 | 1.28 | -1.38 | 7 | P | P |
| 358.2 | detail | entangled | seeded | 3 | 4.36 | 11.90 | 0.23 | 9 | P | P |
| 358.4 | theme | grounded | entangled | 2 | 2.58 | 1.67 | 3.13 | 13 | P | P |
| 358.6 | theme | entangled | unresolved |  | 0.29 | 1.31 | 0.11 | 5 | . | P |
| 359.1 | theme | entangled | unresolved |  | 1.27 | 3.21 | 0.95 | 5 | . | P |
| 359.2 | detail | grounded | entangled | 2 | 3.79 | 3.43 | 1.94 | 9 | P | P |
| 359.9 | theme | confabulated | confabulated | 3 | -0.02 | 1.02 | -0.02 | 7 | P | P |
| 360.2 | detail | grounded | seeded | 3 | 9.66 | 10.93 | 5.30 | 6 | P | P |
| 360.3 | theme | confabulated | confabulated | 3 | -0.72 | -0.44 | 1.38 | 5 | P | P |
| 360.5 | theme | grounded | unresolved |  | 1.61 | 1.71 | 1.33 | 5 | P | . |
| 361.0 | theme | grounded | seeded | 3 | 5.94 | 15.81 | -5.80 | 9 | P | P |
| 361.7 | theme | entangled | unresolved |  | 0.04 |  |  |  | . | . |
| 361.12 | detail | entangled | seeded | 3 | 4.59 | 6.84 | -0.18 | 5 | P | P |
| 362.3 | theme | confabulated | confabulated | 3 | -0.48 | 0.97 | 2.76 | 11 | P | P |
| 362.6 | detail | entangled | unresolved |  | 0.00 |  |  | 0 | . | . |
| 362.7 | theme | confabulated | confabulated | 3 | -0.13 | 2.10 | -0.64 | 11 | . | P |
| 363.0 | theme | confabulated | seeded | 3 | 0.02 | 10.51 | -0.79 | 17 | P | P |
| 363.10 | theme | confabulated | seeded | 3 | 0.09 | 0.69 | 0.44 | 6 | P | P |
| 363.15 | detail | entangled | seeded | 3 | 0.62 | 4.13 | -0.89 | 18 | P | P |
| 364.0 | theme | confabulated | confabulated | 3 | -0.47 | 0.01 | 1.23 | 5 | P | P |
| 364.5 | theme | grounded | grounded | 2 | 9.17 | 5.35 | 6.50 | 8 | P | P |
| 364.12 | detail | grounded | unresolved |  | 15.11 | 3.36 | 15.20 | 4 | P | . |
| 365.7 | theme | entangled | unresolved |  | 0.01 | 1.26 | -0.19 | 4 | . | . |
| 365.9 | theme | grounded | grounded | 2 | 3.60 | 1.10 | 6.52 | 7 | P | P |
| 365.11 | detail | grounded | seeded | 3 | 1.56 | 4.99 | 0.94 | 10 | P | P |
| 366.5 | theme | grounded | seeded | 3 | 3.15 | 9.72 | -1.56 | 6 | P | P |
| 366.10 | detail | confabulated | confabulated | 3 | -0.02 | 1.53 | -0.30 | 6 | P | P |
| 366.13 | theme | confabulated | seeded | 3 | 0.01 | 1.19 | 0.03 | 5 | P | P |
| 367.2 | theme | confabulated | confabulated | 3 | -0.87 | 0.02 | 0.31 | 8 | P | P |
| 367.7 | theme | entangled | unresolved |  | 0.01 | 1.30 | 0.09 | 3 | . | . |
| 367.8 | detail | grounded | confabulated | 3 | 1.66 | 1.91 | 2.26 | 6 | P | P |
| 368.4 | theme | grounded | seeded | 3 | 1.37 | 2.51 | 0.22 | 6 | P | P |
| 368.6 | detail | entangled | unresolved |  | 0.07 | 1.74 | -0.20 | 4 | P | . |
| 368.15 | theme | confabulated | confabulated | 3 | -0.10 | 0.85 | 0.19 | 5 | P | P |
| 369.0 | theme | grounded | unresolved |  | 3.67 | 1.57 | 3.48 | 3 | P | . |
| 369.3 | theme | entangled | confabulated | 3 | 0.34 | 2.33 | 0.12 | 6 | P | P |
| 369.6 | detail | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 370.4 | theme | confabulated | confabulated | 3 | -0.12 | 4.33 | -2.14 | 6 | P | P |
| 370.13 | detail | confabulated | unresolved |  | 0.19 | 1.49 | -0.27 | 7 | . | P |
| 370.15 | theme | entangled | seeded | 3 | 0.39 | 3.34 | -0.99 | 7 | P | P |
| 371.7 | theme | confabulated | unresolved |  | -0.00 | 0.77 | 0.13 | 1 | P | . |
| 371.12 | theme | entangled | seeded | 3 | 0.09 | 2.07 | -0.73 | 9 | P | P |
| 371.16 | detail | entangled | unresolved |  | 0.47 | 2.11 | -0.19 | 7 | . | P |
| 372.0 | theme | grounded | confabulated | 3 | 4.11 | 10.47 | -2.11 | 8 | P | P |
| 372.3 | theme | confabulated | confabulated | 3 | -1.12 | -2.85 | 4.80 | 6 | P | P |
| 372.4 | detail | entangled | unresolved |  | 0.25 | -0.26 | 1.41 | 1 | P | . |
| 373.2 | detail | entangled | seeded | 3 | 2.33 | 5.85 | 0.56 | 6 | P | P |
| 373.3 | theme | entangled | unresolved |  | 0.41 | 5.18 | 0.26 | 5 | P | . |
| 373.9 | theme | confabulated | confabulated | 3 | -0.20 | 7.75 | -1.46 | 17 | P | P |
| 374.1 | theme | grounded | seeded | 3 | 1.35 | 3.32 | -0.54 | 7 | P | P |
| 374.8 | theme | confabulated | seeded | 3 | 0.02 | 1.09 | 0.11 | 5 | P | P |
| 374.9 | detail | confabulated | unresolved |  | -0.40 |  |  |  | . | . |
| 375.4 | theme | confabulated | confabulated | 3 | -0.17 | 0.94 | 0.17 | 8 | P | P |
| 375.10 | detail | confabulated | unresolved |  |  |  |  |  | . | . |
| 375.12 | theme | entangled | seeded | 3 | 0.21 | 1.14 | 0.01 | 7 | P | P |
| 376.2 | detail | grounded | seeded | 3 | 7.35 | 22.98 | -9.57 | 9 | P | P |
| 376.4 | theme | grounded | seeded | 3 | 1.39 | 2.80 | 1.32 | 8 | P | P |
| 376.9 | theme | confabulated | confabulated | 3 | 0.05 | 0.55 | 0.70 | 6 | P | P |
| 377.7 | theme | confabulated | unresolved |  | 0.01 | 1.79 | -0.60 | 9 | . | P |
| 377.8 | detail | grounded | entangled | 2 | 9.93 | 4.75 | 8.66 | 7 | P | P |
| 377.9 | theme | entangled | seeded | 3 | 0.21 | 1.73 | -0.39 | 7 | P | P |
| 378.2 | theme | confabulated | confabulated | 3 | -0.12 | 1.00 | -0.00 | 6 | P | P |
| 378.6 | theme | entangled | seeded | 3 | 0.15 | 1.16 | 0.04 | 7 | P | P |
| 378.11 | detail | confabulated | confabulated | 3 | -0.18 | 1.21 | 0.45 | 7 | P | P |
| 379.0 | theme | grounded | seeded | 3 | 18.38 | 18.42 | 3.96 | 21 | P | P |
| 379.1 | theme | entangled | confabulated | 3 | 0.67 | 1.89 | 0.69 | 9 | P | P |
| 379.7 | detail | grounded | unresolved |  | 13.56 | 9.40 | 8.90 | 5 | P | . |
| 380.1 | theme | confabulated | confabulated | 3 | -0.18 | 0.84 | 0.13 | 5 | P | P |
| 380.3 | detail | grounded | seeded | 3 | 2.36 | 7.64 | -2.72 | 6 | P | P |
| 380.12 | theme | confabulated | confabulated | 3 | -0.21 | 0.78 | 0.28 | 5 | P | P |
| 381.6 | theme | entangled | unresolved |  | -0.09 |  |  |  | . | . |
| 381.7 | detail | entangled | unresolved |  | 0.57 | 1.08 | 0.85 | 4 | P | . |
| 381.8 | theme | confabulated | confabulated | 3 | -0.09 | 0.83 | 0.23 | 10 | P | P |
| 382.5 | theme | confabulated | confabulated | 3 | -0.23 | 4.82 | -0.95 | 22 | P | P |
| 382.9 | theme | entangled | unresolved |  | 0.07 | 1.94 | -0.30 | 22 | . | P |
| 382.12 | detail | entangled | confabulated | 3 | -0.04 | 0.77 | 0.12 | 6 | P | P |
| 383.5 | theme | confabulated | seeded | 3 | 0.13 | 1.37 | 0.03 | 5 | P | P |
| 383.6 | theme | confabulated | confabulated | 3 | 0.01 | 2.51 | 0.32 | 9 | P | P |
| 383.12 | detail | entangled | unresolved |  | 0.59 | 2.08 | -0.54 | 2 | P | . |
| 384.4 | theme | confabulated | confabulated | 3 | 0.06 | 2.41 | 0.96 | 9 | P | P |
| 384.7 | theme | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 384.8 | detail | entangled | unresolved |  | 0.61 | -2.10 | 7.19 | 3 | P | . |
| 385.4 | theme | grounded | grounded | 2 | 6.72 | 6.40 | 5.54 | 6 | P | P |
| 385.7 | theme | confabulated | unresolved |  | 0.02 | 0.87 | 0.06 | 1 | . | . |
| 385.14 | detail | confabulated | unresolved |  |  |  |  |  | . | . |
| 386.6 | theme | confabulated | unresolved |  | -0.08 | 1.02 | -0.02 | 4 | . | . |
| 386.8 | theme | entangled | unresolved |  | 0.03 | 1.75 | -0.09 | 9 | . | P |
| 386.11 | detail | confabulated | unresolved |  | -1.00 | -0.78 | 1.71 | 5 | P | . |
| 387.5 | theme | entangled | unresolved |  | 0.35 | -6.03 | 7.29 | 1 | P | . |
| 387.9 | theme | entangled | seeded | 3 | 0.62 | 2.14 | -0.26 | 5 | P | P |
| 387.10 | detail | entangled | seeded | 3 | 0.85 | 3.01 | 0.60 | 7 | P | P |
| 388.0 | theme | entangled | confabulated | 3 | 3.42 | 4.05 | 3.85 | 11 | P | P |
| 388.10 | theme | entangled | confabulated | 3 | 0.20 | 1.94 | 0.21 | 6 | P | P |
| 388.13 | detail | grounded | unresolved |  | 2.28 | 5.11 | -1.93 | 1 | P | . |
| 389.7 | theme | confabulated | unresolved |  | -0.03 | 0.88 | 0.02 | 3 | . | . |
| 389.11 | theme | entangled | confabulated | 3 | 0.10 | 1.18 | -0.03 | 6 | P | P |
| 390.0 | theme | entangled | confabulated | 3 | 0.15 | 2.18 | 0.19 | 6 | P | P |
| 390.6 | detail | confabulated | entangled | 2 | 0.02 | -4.48 | 6.87 | 14 | P | P |
| 390.7 | theme | entangled | seeded | 3 | 0.39 | 1.58 | -0.03 | 5 | P | P |
| 391.5 | theme | confabulated | seeded | 3 | 0.10 | 2.29 | 0.16 | 5 | P | P |
| 391.8 | theme | entangled | seeded | 3 | 0.78 | 2.35 | -0.08 | 11 | P | P |
| 391.13 | detail | confabulated | confabulated | 3 | -0.27 | 1.03 | 0.50 | 5 | P | P |
| 392.4 | theme | confabulated | unresolved |  | -1.04 | 0.14 | 0.62 | 5 | P | . |
| 392.10 | detail | entangled | confabulated | 3 | 0.40 | 2.14 | -0.26 | 5 | P | P |
| 392.11 | theme | confabulated | confabulated | 3 | -0.01 | 0.93 | 0.12 | 7 | P | P |
| 393.1 | theme | confabulated | unresolved |  | -1.25 | -2.10 | 11.02 | 5 | P | . |
| 393.2 | detail | grounded | unresolved |  | 7.62 |  |  |  | . | . |
| 393.3 | theme | grounded | seeded | 3 | 5.12 | 16.66 | 4.28 | 7 | P | P |
| 394.0 | theme | grounded | unresolved |  | 3.56 |  |  |  | . | . |
| 394.1 | detail | grounded | seeded | 3 | 8.92 | 13.50 | 0.98 | 6 | P | P |
| 394.2 | theme | confabulated | confabulated | 3 | -0.34 | 1.20 | 1.65 | 9 | P | P |
| 395.2 | theme | entangled | seeded | 3 | 0.89 | 5.66 | -2.19 | 5 | P | P |
| 395.7 | detail | grounded | seeded | 3 | 11.34 | 17.14 | -2.19 | 7 | P | P |
| 395.8 | theme | confabulated | unresolved |  | -0.11 | 1.15 | 0.44 | 4 | P | . |
| 396.3 | theme | entangled | seeded | 3 | 0.95 | 1.83 | 0.28 | 5 | P | P |
| 396.4 | theme | entangled | seeded | 3 | 0.16 | 1.49 | -0.16 | 5 | P | P |
| 396.11 | detail | entangled | seeded | 3 | 0.84 | 2.35 | -0.16 | 7 | P | P |
| 397.1 | theme | confabulated | unresolved |  | -0.75 | -1.21 | 3.00 | 2 | P | . |
| 397.4 | detail | confabulated | unresolved |  | 0.00 |  |  |  | . | . |
| 397.6 | theme | confabulated | confabulated | 3 | -0.10 | 1.99 | 0.01 | 23 | P | P |
| 398.1 | theme | grounded | seeded | 3 | 3.59 | 8.37 | 0.01 | 7 | P | P |
| 398.2 | theme | grounded | seeded | 3 | 2.87 | 9.65 | -1.38 | 6 | P | P |
| 398.13 | detail | confabulated | confabulated | 3 | -0.32 | -0.24 | 1.78 | 5 | P | P |
| 399.0 | theme | grounded | seeded | 3 | 3.12 | 6.85 | -0.69 | 6 | P | P |
| 399.10 | detail | grounded | seeded | 3 | 4.19 | 5.61 | 2.55 | 7 | P | P |
| 399.13 | theme | confabulated | unresolved |  | -0.03 |  |  |  | . | . |
