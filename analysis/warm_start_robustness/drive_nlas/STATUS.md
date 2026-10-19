# Supplied-NLA warm-start robustness status

Last updated: 2026-09-10 17:01 America/New_York.

- Scope is frozen in `PROTOCOL.yaml`: the Qwen2.5-7B layer-20
  plausible/control versus implausible/false-but-related contrast,
  final/latest and FVE-nearest comparisons, two samples per activation and
  condition, and clustered inference.
- All four shared Google Drive folders are accessible. Actor and critic
  checkpoint directories and sidecars are present at the iterations recorded
  in `RUN_MANIFEST.yaml`.
- The inspected sidecars agree on the live NLA contract: model, layer,
  dimensionality, injection token and neighbors, injection scale, MSE scale,
  and prompt templates.
- All Drive file IDs were enumerated and saved under `manifests/`. The corrected
  control iteration-780 actor/critic pair finished downloading to Athena: all 22
  selected files (26,152,546,974 bytes) are individually SHA-256 verified in an
  atomic receipt copied back to this repository.
- The three browser-downloaded implausible critic files were staged, hashed on
  both machines, checked against the HF index, scanned tensor-by-tensor for
  non-finite values, and only then atomically installed. The combined
  iteration-780 receipt covers all 22 selected files (26,152,547,002 bytes).
- Both critics pass complete HF validation: 253 indexed backbone tensors and
  10,878,422,016 logical tensor bytes, plus finite bfloat16 3584-by-3584 value
  heads. Both two-rank actor DCPs were converted to full HF; every converted
  tensor is bitwise equal to the loaded DCP state after serialization.
- The anti-Carthage run ends at iteration 100. It will be reported as latest
  available and as FVE-nearest only if the measured gap is within the frozen
  one-percentage-point tolerance.
- No existing FVE-robustness dashboard page has been modified. A separate page
  will be generated for this experiment.

## Resolved initial blocker

- The control iteration-780 actor contains only `model/__0_0.distcp`; its
  required `model/__1_0.distcp` and `model/.metadata` are absent from the shared
  folder. The paired iteration-780 critic has no exported `hf/` weights or value
  head at all.
- Control iteration 600 has both actor shards and actor metadata, but its critic
  export contains only two of the three indexed model shards; specifically,
  `model-00002-of-00003.safetensors` is absent.
- Control iterations 200 and 400 are structurally complete. Anti-Carthage 100,
  pro-Carthage 400/600/780, and false-but-related 200/400/600/780 are also
  structurally complete at the Drive-inventory level. Their tensor contents
  have not yet been downloaded or validated.
- Work stopped before large checkpoint downloads or GPU execution. Replacing
  the intended final control with iteration 400 or another model would change
  the primary experimental comparison and requires user direction.

On 2026-09-10 the user supplied corrected control folder
`1xUX_fD5Stz0LR4fZf9W0gEfIvQJ5Wr1X`. Recursive inspection found complete actor
and critic pairs at iterations 200, 400, 600, and 780. The corrected source is
being transferred to a separate `control_corrected` directory; the incomplete
files above remain preserved but cannot enter evaluation manifests.

The repository-local `OPENAI_API_KEY_2` authenticated successfully without its
value being printed or persisted in any experiment artifact. The frozen
`gpt-5-nano` judge is available.

FineWeb (2,000 raw vectors) and static OOD (1,400 raw vectors) passed the
canonical manifest converter on Athena. Their manifests and source hashes were
copied locally. The DCP converter and full-HF generation loader pass syntax and
all 31 CPU/mock tests.

## Resolved injection-harness blocker

The preliminary generator used EasyNLA's layer-1 Karvonen residual-addition
hook. These supplied actors require the raw vector to be normalized to the
sidecar's `injection_scale: 150`, to replace the marker token's input embedding
via `inject_at_marked_positions`, and to be passed through `inputs_embeds`.

The wrong-hook diagnostic produced FVE of -67.47% for control and -55.80% for
implausible, contradicting the expected positive 68--70% range. Those values
and all associated smoke generations are explicitly quarantined as invalid
diagnostics under `manifests/diagnostics/`; they are not experimental results.
The evaluator and generator now use that native pathway. A GPU smoke test used
10 FineWeb activations and two samples per activation for each condition. All
40 generations passed strict tag parsing; every marker row exactly equaled the
scaled activation; observed injected norms ranged from 150.000046 to
150.000092; and there were no marker leaks or CJK alerts. The complete atomic
records are saved both on Athena and under `artifacts/smoke_native/` locally.

The earlier Drive quota blocker is resolved. Its failed and retried logs remain
preserved under `logs/`.

## Current blocker

The requested OOD analysis requires 16 benchmark families, but the frozen
`static-ood.json` contains only seven. Canonical, admissible saved gold exists
for three additional families (BrowseComp, Destructive Coding Actions, and
SWE-bench Verified), so only 10/16 families are currently usable.

The six unresolved families are: Overly Agentic Actions (no completed canonical
trajectory), Humanity's Last Exam (frozen data access returned HTTP 403),
tau2-bench Retail (saved gold uses the superseded Luna simulator and is marked
inadmissible), Terminal-Bench 2 (no canonical gold; official runner needs
Docker), FinanceAgent (no canonical gold; third-party credentials required),
and CyberGym (explicitly excluded by the user's earlier scope amendment).

Consequently, the preregistered 16-family OOD macro cannot be computed. Work is
stopped before exact FVE evaluation, full claim generation, or paid judge calls
pending a choice between a clearly labeled 10-family partial OOD analysis and
completing/amending the OOD scope.
