"""Build the warm-start robustness run dashboard from durable run artifacts.

The page is intentionally data-only: every displayed measurement comes from
RUN_MANIFEST.yaml or the copied RL metrics JSONL.  Re-run after syncing the
remote metrics file to refresh the active curve.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError(f"metrics steps must be contiguous from zero; got tail {steps[-5:]}")
    return rows


def fnum(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def build(
    metrics: Path,
    manifest_path: Path,
    redundancy_index: Path,
    health_path: Path,
    output: Path,
) -> None:
    rows = read_jsonl(metrics)
    manifest = yaml.safe_load(manifest_path.read_text())
    active = manifest.get("active_training", {})
    planned = manifest.get("planned_training", {}).get("rl", {})
    total_steps = int(planned.get("iterations", 780))
    save_every = int(active.get("save_every_steps", 20))
    latest = rows[-1] if rows else {}
    latest_step = int(latest.get("step", -1))
    completed = latest_step + 1
    latest_ckpt = (completed // save_every) * save_every
    eval_rows = [row for row in rows if "eval/fve_pct" in row]
    latest_eval = eval_rows[-1] if eval_rows else {}
    spend = manifest["teacher"]["total_api_spend_including_qc_and_rejected_smoke_usd"]
    teacher = manifest["teacher"]["full_generation"]
    sft = manifest.get("completed_sft", {})
    recorded_state = str(active.get("status", "")).strip().lower()
    state = "complete" if completed >= total_steps else (recorded_state or "running")
    progress = min(100.0, 100.0 * completed / total_steps)
    redundancy = json.loads(redundancy_index.read_text()) if redundancy_index.exists() else {}
    verified_records = redundancy.get("records", [])
    latest_verified = max(
        verified_records,
        key=lambda row: int(row.get("checkpoint_step", -1)),
        default={},
    )
    verified_step = latest_verified.get("checkpoint_step")
    verified_commit = str(latest_verified.get("commit", ""))
    verified_bytes = int(latest_verified.get("verified_logical_bytes", 0))
    verified_files = latest_verified.get("verified_files")
    health = json.loads(health_path.read_text()) if health_path.exists() else {}
    worker_health = health.get("durability_workers", {})
    health_alerts = health.get("alerts", [])
    ready_steps = health.get("continuation_snapshots", {}).get("ready_steps", [])

    chart_data = {
        "eval": [
            {
                "step": int(row["step"]),
                "fve": float(row["eval/fve_pct"]),
                "extraction": float(row["eval/extraction_rate"]),
            }
            for row in eval_rows
        ],
        "train": [
            {"step": int(row["step"]), "fve": float(row["fve_pct"])}
            for row in rows
        ],
        "target": 68.0,
    }
    eval_table = "".join(
        "<tr>"
        f"<td>{int(row['step'])}</td>"
        f"<td class='num strong'>{row['eval/fve_pct']:.1f}%</td>"
        f"<td class='num'>{row['eval/reward_mean_valid']:.3f}</td>"
        f"<td class='num'>{100 * row['eval/extraction_rate']:.1f}%</td>"
        f"<td class='num'>{row['time/eval_base_fve_s']:.1f}s</td>"
        "</tr>"
        for row in eval_rows
    ) or "<tr><td colspan='5' class='muted'>No held-out evaluation has completed.</td></tr>"

    data_json = json.dumps(chart_data, separators=(",", ":"))
    built = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    eta = html.escape(str(active.get("projected_completion_local", "—")))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NLA warm-start robustness</title>
<style>
:root{{--bg:#f7f3ec;--paper:#fffdf8;--ink:#211d17;--muted:#756b5e;--line:#e1d9cc;
--green:#2f6f55;--green2:#6ca081;--rust:#a25546;--blue:#52749b;--wash:#eee8dc}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 Georgia,'Times New Roman',serif}} .wrap{{max-width:1160px;margin:0 auto;padding:28px}}
nav{{display:flex;gap:18px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:24px}}
nav a{{color:var(--muted);text-decoration:none}} nav a.on{{color:var(--green);font-weight:700}}
h1{{font-size:30px;line-height:1.1;margin:0 0 8px;letter-spacing:-.02em}} h2{{font-size:18px;margin:0 0 12px}}
.dek{{color:var(--muted);max-width:850px;margin:0 0 20px}} .status{{display:inline-block;padding:3px 10px;
border:1px solid #b8d1c1;border-radius:999px;color:var(--green);background:#edf6f0;font:700 12px system-ui;
text-transform:uppercase;letter-spacing:.05em;margin-left:8px;vertical-align:middle}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px}}
.card,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:9px;box-shadow:0 2px 8px #3a2e2010}}
.card{{padding:14px 16px}} .label{{color:var(--muted);font:600 11px system-ui;text-transform:uppercase;letter-spacing:.06em}}
.value{{font:700 26px/1.2 Georgia;margin-top:4px}} .sub{{font-size:12px;color:var(--muted);margin-top:3px}}
.panel{{padding:18px 20px;margin-bottom:16px}} .split{{display:grid;grid-template-columns:1.55fr 1fr;gap:16px}}
.progress{{height:8px;background:var(--wash);border-radius:9px;overflow:hidden;margin:9px 0 5px}}
.progress i{{display:block;height:100%;background:var(--green);width:{progress:.3f}%}}
.chartbox{{height:330px;position:relative}} canvas{{width:100%;height:100%}}
.legend{{display:flex;gap:18px;font:12px system-ui;color:var(--muted);margin-top:8px}}
.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px}} .dash{{width:18px;border-top:2px dashed var(--rust);display:inline-block;margin:0 6px 3px 0}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th{{color:var(--muted);font:600 11px system-ui;
text-transform:uppercase;letter-spacing:.05em;text-align:left;border-bottom:2px solid var(--line);padding:7px}}
td{{border-bottom:1px solid #eee8df;padding:8px 7px}} .num{{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.strong{{color:var(--green);font-weight:800}} .muted{{color:var(--muted)}} .facts{{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px}}
.facts div{{border-bottom:1px solid #eee8df;padding:7px 0}} .facts b{{float:right;font-variant-numeric:tabular-nums}}
.caution{{border-left:3px solid var(--rust);padding-left:12px;color:#62443c}} footer{{font-size:12px;color:var(--muted);padding:8px 0 20px}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr 1fr}}.split{{grid-template-columns:1fr}}}}
@media(max-width:520px){{.wrap{{padding:18px}}.grid{{grid-template-columns:1fr}}.facts{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<nav><a href="../fve_robustness/index.html">AV–AR robustness</a><a class="on" href="dashboard.html">Warm start</a></nav>
<h1>Robustness to the warm start <span class="status">{state}</span></h1>
<p class="dek">Qwen2.5–7B, layer 20. A new implausible-initialized NLA trained from 20,000 Ultra-FineWeb documents. Its exact final evaluation is followed by a matched 20k plausible-initialized control; the released plausible NLA remains external context only.</p>
<div class="grid">
 <div class="card"><div class="label">Latest held-out FVE</div><div class="value">{fnum(latest_eval.get('eval/fve_pct'),1,'%')}</div><div class="sub">step {latest_eval.get('step','—')} · fixed 256-prompt set</div></div>
 <div class="card"><div class="label">RL progress</div><div class="value">{completed} / {total_steps}</div><div class="progress"><i></i></div><div class="sub">latest Athena checkpoint {latest_ckpt or 'none'}</div></div>
 <div class="card"><div class="label">Latest extraction</div><div class="value">{fnum(100*latest_eval.get('eval/extraction_rate') if latest_eval else None,1,'%')}</div><div class="sub">marker failures: {int(latest.get('av/marker_bad_count',0)) if latest else '—'}</div></div>
 <div class="card"><div class="label">Teacher API spend</div><div class="value">${spend:.2f}</div><div class="sub">hard cap $100 · no RL API calls</div></div>
</div>
<section class="panel"><h2>Checkpoint durability</h2><div class="facts">
 <div>Latest off-cluster verified step <b>{verified_step if verified_step is not None else 'pending'}</b></div>
 <div>Payload commit <b>{html.escape(verified_commit[:12] if verified_commit else 'pending')}</b></div>
 <div>Payload files verified <b>{verified_files if verified_files is not None else '—'}</b></div>
 <div>Verified logical size <b>{fnum(verified_bytes / 1_000_000_000 if verified_bytes else None,2,' GB')}</b></div>
 <div>Latest immutable Athena bundle <b>{max(ready_steps) if ready_steps else 'pending'}</b></div>
 <div>Durability workers <b>{'active' if int(worker_health.get('snapshotter_processes',0)) else 'inactive'} snapshot / {'active' if int(worker_health.get('uploader_processes',0)) else 'inactive'} upload</b></div>
 <div>Watchdog alerts <b>{len(health_alerts)}</b></div>
 <div>Last watchdog check <b>{html.escape(str(health.get('checked_at_utc','pending')))}</b></div>
 </div><p class="muted">Every 20-step transaction is copied into an immutable restart bundle on Athena, then uploaded to a private off-cluster archive and checked file-by-file by size and SHA-256. Metrics, logs, optimizer state, actor/reference adapters, critic, value head, configs, and code hashes are included.</p></section>
<div class="split">
 <section class="panel"><h2>Reconstruction learning curve</h2><div class="chartbox"><canvas id="curve" aria-label="Held-out and training FVE by RL step"></canvas></div>
 <div class="legend"><span><i class="dot" style="background:var(--green)"></i>held-out FVE</span><span><i class="dot" style="background:var(--blue)"></i>training shard FVE</span><span><i class="dash"></i>reference report’s ~68% final implausible FVE</span></div></section>
 <section class="panel"><h2>Run facts</h2><div class="facts">
  <div>Usable documents <b>{manifest['conditions']['implausible']['corpus_documents_usable']:,}</b></div>
  <div>Raw activations <b>{manifest['data_generation']['completed_rows']:,}</b></div>
  <div>Retained teacher rows <b>{teacher['retained_rows_total']:,}</b></div>
  <div>Teacher retention <b>{100*teacher['retained_fraction']:.2f}%</b></div>
  <div>AV SFT steps <b>{sft.get('av_optimizer_steps','—')}</b></div>
  <div>AR SFT steps <b>{sft.get('ar_optimizer_steps','—')}</b></div>
  <div>AR teacher-forced FVE <b>{fnum(sft.get('ar_final_heldout_teacher_forced_fve_percent'),2,'%')}</b></div>
  <div>Checkpoint cadence <b>{save_every} steps</b></div>
  <div>Matched plausible control <b>queued after implausible evaluation</b></div>
 </div><p class="caution" style="margin-top:15px">The public plausible checkpoint used a larger, different recipe. Its published score is context, not a matched control for this 20k run.</p>
 <p class="muted">Current implausible-RL ETA: {eta}</p></section>
</div>
<section class="panel"><h2>Fixed held-out evaluations</h2><table><thead><tr><th>RL step</th><th style="text-align:right">FVE</th><th style="text-align:right">valid reward</th><th style="text-align:right">extraction</th><th style="text-align:right">eval time</th></tr></thead><tbody>{eval_table}</tbody></table></section>
<section class="panel"><h2>Interpretation boundary</h2><p>The curve measures reconstruction, not semantic convergence or plausibility. Claim-level agreement and plausibility evaluation remain separate downstream analyses. The run uses a document-hash held-out split, raw stored activations, Karvonen norm-matched residual injection, suffix-anchored AR extraction, and per-step steering coverage checks.</p></section>
<footer>Built {built} from <code>RUN_MANIFEST.yaml</code> and {html.escape(str(metrics.relative_to(HERE) if metrics.is_relative_to(HERE) else metrics))}. Reference protocol: <a href="https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv/how-robust-are-natural-language-autoencoders-to">Zhang &amp; Turner warm-start report</a>.</footer>
</main><script>
const D={data_json};
function draw(){{const c=document.getElementById('curve'),dpr=window.devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*dpr;c.height=r.height*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);const W=r.width,H=r.height,p={{l:48,r:18,t:16,b:35}},iw=W-p.l-p.r,ih=H-p.t-p.b;
 const X=s=>p.l+iw*s/{total_steps},Y=v=>p.t+ih*(1-(v+10)/90);x.clearRect(0,0,W,H);x.font='12px system-ui';x.fillStyle='#756b5e';x.strokeStyle='#e1d9cc';x.lineWidth=1;
 for(let v=0;v<=80;v+=20){{const y=Y(v);x.beginPath();x.moveTo(p.l,y);x.lineTo(W-p.r,y);x.stroke();x.fillText(v+'%',8,y+4)}}for(let s=0;s<={total_steps};s+=130){{const xx=X(s);x.fillText(String(s),xx-8,H-9)}}
 x.setLineDash([6,5]);x.strokeStyle='#a25546';x.beginPath();x.moveTo(p.l,Y(D.target));x.lineTo(W-p.r,Y(D.target));x.stroke();x.setLineDash([]);
 function line(a,col,w,rad){{if(!a.length)return;x.strokeStyle=col;x.lineWidth=w;x.beginPath();a.forEach((q,i)=>{{const xx=X(q.step),yy=Y(q.fve);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}});x.stroke();if(rad){{x.fillStyle=col;a.forEach(q=>{{x.beginPath();x.arc(X(q.step),Y(q.fve),rad,0,Math.PI*2);x.fill()}})}}}}
 line(D.train,'#52749b',1.2,0);line(D.eval,'#2f6f55',2.7,3.2);x.fillStyle='#756b5e';x.fillText('RL step',W/2-20,H-8)}}
draw();addEventListener('resize',draw);
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(page)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"wrote {output} from {len(rows)} RL steps and {len(eval_rows)} held-out evals")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=HERE / "results" / "rl_metrics.jsonl")
    p.add_argument("--manifest", type=Path, default=HERE / "RUN_MANIFEST.yaml")
    p.add_argument(
        "--redundancy-index",
        type=Path,
        default=HERE / "results" / "continuation_verified_index.json",
    )
    p.add_argument(
        "--health",
        type=Path,
        default=HERE / "results" / "continuation_health.json",
    )
    p.add_argument("--output", type=Path, default=HERE / "dashboard.html")
    args = p.parse_args()
    build(
        args.metrics.resolve(),
        args.manifest.resolve(),
        args.redundancy_index.resolve(),
        args.health.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
