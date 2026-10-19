"""Restart RL after a startup failure, preserving completed SFT and launch history."""
import datetime
import fcntl
import hashlib
import json
import subprocess

from pipeline import ROOT, launch, state, wait_jobs


def main():
    with (ROOT / "rl_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # This entry point starts from SFT; it must never discard RL progress.
        assert not list((ROOT / "rl").glob("iter_*")), "Use explicit checkpoint resume for an existing RL run"
        assert not (ROOT / "rl/optim_latest.pt").exists(), "RL optimizer already exists"
        occupied = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ], text=True).strip()
        assert not occupied, f"GPUs still have compute processes; no launch: {occupied}"
        command = json.loads((ROOT / "commands/rl.json").read_text())
        argv = command["argv"]
        for flag in ("--av-ckpt", "--av-adapter", "--ar-ckpt"):
            from pathlib import Path
            assert Path(argv[argv.index(flag) + 1]).is_dir(), flag
        steps = int(argv[argv.index("--num-steps") + 1])
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt = f"rl_restart_{stamp}"
        history = ROOT / "restart_history"
        history.mkdir(exist_ok=True)
        (history / f"{attempt}_previous_status.json").write_bytes((ROOT / "status.json").read_bytes())
        (history / f"{attempt}.json").write_text(json.dumps({
            "command": command,
            "controller_sha256": hashlib.sha256((ROOT / "restart_rl.py").read_bytes()).hexdigest(),
            "reason": "Initial RL model load failed while other GPU processes occupied memory; all GPUs verified idle before retry.",
        }, indent=2))
        state("rl_starting", attempt=attempt, log=f"logs/{attempt}.log", updates=steps, gpus=command["gpus"])
        wait_jobs([launch(attempt, argv, command["gpus"])])
        assert (ROOT / "rl" / f"iter_{steps:06d}" / "adapter_config.json").exists()
        assert int((ROOT / "rl/critic_latest/saved_at_step.txt").read_text()) == steps
        state("complete", updates=steps, planned_explanations=steps * 2048)
        (ROOT / "COMPLETE").touch()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        state("failed", error=str(exc))
        raise
