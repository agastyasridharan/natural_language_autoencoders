# Cycle-consistency verdict report

run `qwen7b-cp6-fineweb-atoms`  claims 75  k 5

## Verdict distribution

| verdict | n | % |
|---|---|---|
| grounded | 3 | 4.0 |
| entangled | 3 | 4.0 |
| seeded | 26 | 34.7 |
| confabulated | 19 | 25.3 |
| unresolved | 24 | 32.0 |

unresolved by reason: validity 13, power 8, no text-passing decoys 3

## Gate flow (spec Step 7)

- 75 claims -> 24 unresolved (pre-checks), 51 reached the gates
- Gate 1 (specificity): 6 pass -> Gate 2, 45 fail -> Gate 3
- Gate 2 (ownership): 3 grounded, 3 entangled
- Gate 3 (shared signal): 26 seeded, 19 confabulated

## Atom -> cycle verdict shift

rows = published atom verdict, columns = cycle verdict

| atom \ cycle | grounded | entangled | seeded | confabulated | unresolved |
|---|---|---|---|---|---|
| confabulated | 0 | 0 | 3 | 17 | 8 |
| entangled | 0 | 1 | 9 | 2 | 10 |
| grounded | 3 | 2 | 14 | 0 | 6 |

## Pre-checks + grounding

- power passed 60/75, validity passed 54/75
- grounding estimate (M_BB - med D*, Gate-1 passers only (grounded+entangled; M_BB > 0), n=6): median 3.7313, p10 2.8292, p90 5.4814

## kappa_0 re-fit (Gate 1 null on THIS run)

- calibration kappa_0 = 0.9 (applied)
- re-fit on 43 survivor D* families = 0.8
- decoy-vs-decoy exceedance rate at the calibration kappa_0 = 0.0434 (target <= 0.05, whole run incl. in-sample)
- **out-of-sample** (0 activations disjoint from the 25 calibration ones, 0 families): re-fit kappa_0 = None, FP rate at calibration kappa_0 = None -> DOES NOT TRANSFER (> 0.05)
- _This re-fit is IN-SAMPLE when the run activations are the calibration split (e.g. the S4 small-N run reuses the 25 calibration activations): the LODO families and grid are then identical to calib_cycle.py's, so the FP rate is a consistency check, NOT out-of-sample validation. Genuine out-of-sample validation of kappa_0 requires a run on activations disjoint from calibration (S5's 375 new activations). kappa_0_refit agrees with the applied calibration kappa_0._

## Per-claim

| idx.ci | type | atom | cycle | gate | M_BB | gate1_bar | ground_est | surv | pw | vd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | detail | grounded | seeded | 3 | 12.32 | 17.20 | 2.97 | 10 | P | P |
| 0.4 | theme | entangled | unresolved |  | 3.80 | 5.93 | 0.58 | 5 | . | . |
| 0.6 | theme | confabulated | unresolved |  | 0.01 | 1.82 | -0.30 | 9 | . | P |
| 16.5 | theme | entangled | seeded | 3 | 0.25 | 1.62 | -0.16 | 7 | P | P |
| 16.7 | detail | entangled | confabulated | 3 | 0.08 | 1.08 | 0.19 | 11 | P | P |
| 16.11 | theme | confabulated | seeded | 3 | 0.11 | 2.13 | -0.20 | 8 | P | P |
| 32.3 | theme | grounded | seeded | 3 | 1.47 | 3.21 | -0.39 | 8 | P | P |
| 32.4 | detail | confabulated | confabulated | 3 | -0.13 | 1.38 | -0.24 | 5 | P | P |
| 32.15 | theme | confabulated | confabulated | 3 | 0.08 | 0.86 | 0.29 | 6 | P | P |
| 48.3 | theme | confabulated | confabulated | 3 | -1.04 | 1.30 | -0.19 | 5 | P | P |
| 48.7 | theme | entangled | unresolved |  | 0.15 | 1.33 | -0.05 | 6 | . | P |
| 48.12 | detail | entangled | confabulated | 3 | 0.12 | 3.74 | -0.08 | 9 | P | P |
| 64.0 | theme | entangled | seeded | 3 | 2.51 | 10.67 | -4.14 | 7 | P | P |
| 64.2 | theme | confabulated | unresolved |  | -0.07 | 0.20 | 0.62 | 1 | . | . |
| 64.9 | detail | entangled | unresolved |  | 0.16 | 2.96 | -1.90 | 1 | P | . |
| 80.1 | theme | grounded | seeded | 3 | 2.41 | 6.15 | -1.01 | 7 | P | P |
| 80.2 | theme | confabulated | confabulated | 3 | -0.22 | 1.31 | -0.36 | 6 | P | P |
| 80.5 | detail | confabulated | seeded | 3 | 0.03 | 0.88 | 0.12 | 6 | P | P |
| 96.1 | theme | confabulated | confabulated | 3 | -0.59 | 2.02 | -1.31 | 5 | . | P |
| 96.6 | theme | grounded | unresolved |  | 15.53 | 16.00 | 4.72 | 5 | P | . |
| 96.12 | detail | entangled | seeded | 3 | 0.99 | 5.50 | -1.35 | 6 | P | P |
| 112.4 | theme | grounded | grounded | 2 | 8.33 | 4.18 | 6.99 | 8 | P | P |
| 112.10 | theme | confabulated | confabulated | 3 | -0.03 | 0.81 | 0.31 | 6 | P | P |
| 112.11 | detail | entangled | seeded | 3 | 0.21 | 1.10 | 0.14 | 5 | P | P |
| 128.1 | theme | grounded | unresolved |  | 9.08 | 9.15 | 2.17 | 2 | P | . |
| 128.5 | theme | confabulated | unresolved |  | 0.19 | 6.10 | -3.01 | 4 | P | . |
| 128.6 | detail | entangled | unresolved |  | 1.68 | 1.99 | 0.77 | 4 | P | . |
| 144.4 | theme | confabulated | unresolved |  | -0.34 | 0.56 | 0.99 | 5 | P | . |
| 144.5 | theme | grounded | seeded | 3 | 3.47 | 7.35 | 0.80 | 9 | P | P |
| 144.12 | detail | confabulated | seeded | 3 | 0.03 | 5.52 | 2.94 | 20 | P | P |
| 160.6 | theme | grounded | grounded | 2 | 1.75 | 0.36 | 3.53 | 10 | P | P |
| 160.7 | detail | confabulated | unresolved |  | 0.00 |  |  |  | . | . |
| 160.8 | theme | confabulated | unresolved |  | 0.00 | 1.00 | 0.08 | 3 | . | . |
| 176.6 | theme | grounded | seeded | 3 | 1.15 | 4.86 | -1.43 | 8 | P | P |
| 176.10 | theme | entangled | unresolved |  | 0.05 |  |  |  | . | . |
| 176.15 | detail | grounded | entangled | 2 | 3.74 | 2.39 | 2.95 | 6 | P | P |
| 192.4 | theme | grounded | unresolved |  | 4.65 | 3.79 | 1.77 | 1 | P | . |
| 192.5 | theme | confabulated | unresolved |  | -0.00 |  |  | 0 | . | . |
| 192.10 | detail | grounded | seeded | 3 | 1.84 | 4.24 | -0.96 | 6 | P | P |
| 208.3 | theme | grounded | entangled | 2 | 3.80 | 2.49 | 2.70 | 6 | P | P |
| 208.8 | theme | entangled | unresolved |  | 0.04 | 1.48 | 0.06 | 5 | . | . |
| 208.9 | detail | confabulated | confabulated | 3 | -0.95 | 0.68 | -0.22 | 8 | P | P |
| 224.1 | theme | confabulated | confabulated | 3 | -0.08 | 1.12 | 1.22 | 6 | . | P |
| 224.5 | theme | grounded | seeded | 3 | 3.54 | 4.88 | 1.67 | 7 | P | P |
| 224.7 | detail | entangled | unresolved |  | 1.65 | 3.92 | 0.24 | 5 | P | . |
| 240.1 | theme | grounded | seeded | 3 | 4.79 | 6.53 | 0.38 | 6 | P | P |
| 240.4 | detail | grounded | grounded | 2 | 12.19 | 11.81 | 3.93 | 6 | P | P |
| 240.5 | theme | entangled | entangled | 2 | 1.23 | 0.42 | 3.97 | 12 | P | P |
| 256.5 | theme | grounded | seeded | 3 | 3.12 | 4.90 | 0.09 | 20 | P | P |
| 256.6 | theme | confabulated | confabulated | 3 | -0.49 | 1.04 | -0.33 | 5 | P | P |
| 256.11 | detail | entangled | seeded | 3 | 0.00 | 1.73 | -0.25 | 11 | P | P |
| 272.0 | theme | grounded | seeded | 3 | 3.70 | 5.41 | 2.05 | 5 | P | P |
| 272.10 | theme | entangled | unresolved |  | 0.00 | 1.08 | -0.03 | 6 | . | P |
| 272.11 | detail | entangled | seeded | 3 | 0.38 | 1.33 | 0.25 | 9 | P | P |
| 288.0 | theme | confabulated | confabulated | 3 | 0.05 | 1.90 | -0.16 | 7 | P | P |
| 288.7 | detail | entangled | seeded | 3 | 0.19 | 5.60 | -0.31 | 6 | P | P |
| 288.8 | theme | confabulated | confabulated | 3 | -0.92 | 0.36 | 0.69 | 7 | . | P |
| 304.0 | theme | grounded | seeded | 3 | 3.44 | 8.17 | -1.20 | 5 | P | P |
| 304.10 | detail | confabulated | confabulated | 3 | -0.18 | 1.02 | -0.18 | 7 | P | P |
| 304.13 | theme | confabulated | confabulated | 3 | -0.02 | 1.55 | -0.27 | 6 | . | P |
| 320.2 | theme | grounded | unresolved |  | 1.04 | 1.02 | 1.20 | 4 | P | . |
| 320.4 | detail | grounded | unresolved |  | 14.10 | 7.04 | 10.39 | 2 | P | . |
| 320.6 | theme | entangled | seeded | 3 | 0.70 | 3.37 | -0.68 | 6 | P | P |
| 336.3 | theme | confabulated | confabulated | 3 | -0.10 | 0.52 | 0.83 | 5 | P | P |
| 336.5 | theme | grounded | unresolved |  | 10.11 | 6.75 | 4.25 | 1 | P | . |
| 336.9 | detail | grounded | seeded | 3 | 2.37 | 8.48 | -3.41 | 9 | P | P |
| 352.2 | theme | grounded | seeded | 3 | 3.49 | 5.38 | 0.12 | 5 | P | P |
| 352.5 | theme | entangled | seeded | 3 | 1.86 | 4.83 | -0.65 | 5 | P | P |
| 352.8 | detail | confabulated | confabulated | 3 | -3.19 | -3.94 | 2.93 | 7 | P | P |
| 368.4 | theme | grounded | seeded | 3 | 1.37 | 2.51 | 0.22 | 6 | P | P |
| 368.6 | detail | entangled | unresolved |  | 0.07 | 1.74 | -0.20 | 4 | P | . |
| 368.15 | theme | confabulated | confabulated | 3 | -0.10 | 0.85 | 0.19 | 5 | P | P |
| 384.4 | theme | confabulated | confabulated | 3 | 0.06 | 2.41 | 0.96 | 9 | P | P |
| 384.7 | theme | confabulated | unresolved |  | -0.00 |  |  |  | . | . |
| 384.8 | detail | entangled | unresolved |  | 0.61 | -2.10 | 7.19 | 3 | P | . |
