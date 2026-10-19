"""Verify and preserve a scheduled RL checkpoint, then stop only its owned processes."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def identity(pid):
    try:
        base = Path('/proc') / str(pid)
        stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, ppid=int(stat[1]), start=stat[19], state=stat[0],
                    command=(base / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def still_same(item):
    current = identity(item['pid'])
    return current and current['start'] == item['start'] and current['state'] != 'Z'


def descendants(pid):
    rows = [identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    rows = [x for x in rows if x]
    selected = {pid}
    while True:
        new = selected | {x['pid'] for x in rows if x['ppid'] in selected}
        if new == selected:
            return [x for x in rows if x['pid'] in selected]
        selected = new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step', required=True, type=int)
    p.add_argument('--launcher-pid', required=True, type=int)
    args = p.parse_args()
    with (ROOT / 'pause_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher = identity(args.launcher_pid)
        assert launcher and 'torch.distributed.run' in launcher['command']
        assert 'nla.train_rl_vllm' in launcher['command'] and str(ROOT) in launcher['command']
        old_status = json.loads((ROOT / 'status.json').read_text())
        logfile = ROOT / old_status['log']
        marker = f'(+ optim_latest @ step {args.step}) (+ critic_latest @ step {args.step})'
        request = dict(step=args.step, launcher=launcher, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (ROOT / 'PAUSE_REQUESTED.json').write_text(json.dumps(request, indent=2))
        deadline = time.monotonic() + 7200
        while marker not in logfile.read_text():
            assert still_same(launcher), 'Training exited before requested checkpoint; inspect before stopping anything'
            if time.monotonic() > deadline:
                raise TimeoutError('Requested checkpoint not completed within two hours')
            time.sleep(5)

        # Inspect on CPU only, including optimizer structure, before stopping training.
        import torch
        from safetensors import safe_open
        torch.set_num_threads(2)
        run = ROOT / 'rl'
        actor = run / f'iter_{args.step:06d}'
        critic = run / 'critic_latest'
        assert int((critic / 'saved_at_step.txt').read_text()) == args.step
        assert (actor / 'adapter_config.json').is_file()
        optimizer = run / 'optim_latest.pt'
        opt = torch.load(optimizer, map_location='cpu', weights_only=True, mmap=True)
        assert opt['step'] == args.step
        optimizer_summary = {}
        for key in ('actor_optim', 'critic_optim'):
            saved = opt[key]
            assert saved['state'] and saved['param_groups']
            params = [i for group in saved['param_groups'] for i in group['params']]
            assert set(saved['state']) <= set(params)
            assert all(s.get('step') == args.step for s in saved['state'].values())
            optimizer_summary[key] = dict(parameter_count=len(params), populated_states=len(saved['state']))
        tensor_counts = {}
        for directory in (actor, critic):
            files = list(directory.rglob('*.safetensors'))
            assert files
            for path in files:
                with safe_open(path, framework='pt', device='cpu') as f:
                    keys = list(f.keys())
                    assert keys
                    tensor_counts[str(path.relative_to(run))] = len(keys)
            index = directory / 'model.safetensors.index.json'
            if index.exists():
                for key, name in json.loads(index.read_text())['weight_map'].items():
                    with safe_open(directory / name, framework='pt', device='cpu') as f:
                        assert key in f.keys(), key

        archive = ROOT / 'pause_snapshots' / f'step_{args.step:06d}'
        archive.mkdir(parents=True, exist_ok=False)
        # Hard links preserve immutable saved tensors without duplicating tens of GB.
        # Subsequent trainer saves atomically replace optimizer/critic files.
        for source, name in ((actor, actor.name), (critic, 'critic_latest'), (run / 'best_pair', 'best_pair')):
            if source.exists():
                shutil.copytree(source, archive / name, copy_function=os.link)
        os.link(optimizer, archive / optimizer.name)
        del opt
        for source in (ROOT / 'run_manifest.json', ROOT / 'commands/rl.json', ROOT / 'warmstart_checkpoints.json'):
            shutil.copy2(source, archive / source.name)
        for source in run.glob('*.json*'):
            shutil.copy2(source, archive / source.name)
        for source in run.glob('*.yaml'):
            shutil.copy2(source, archive / source.name)

        owned = descendants(args.launcher_pid)
        assert still_same(launcher)
        print(f'Checkpoint {args.step} verified and archived; stopping launcher {args.launcher_pid}', flush=True)
        os.kill(args.launcher_pid, signal.SIGTERM)
        until = time.monotonic() + 45
        while time.monotonic() < until and any(still_same(x) for x in owned):
            time.sleep(1)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            remaining = [x for x in owned if still_same(x)]
            if not remaining:
                break
            for item in remaining:
                try:
                    os.kill(item['pid'], sig)
                except ProcessLookupError:
                    pass
            time.sleep(5)
        assert not any(still_same(x) for x in owned), 'Some owned processes remain'
        controller = identity(launcher['ppid'])
        until = time.monotonic() + 20
        while controller and still_same(controller) and time.monotonic() < until:
            time.sleep(1)
        assert not controller or not still_same(controller), 'Old controller has not exited'

        gpu_state = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True)
        completed = []
        for path in sorted(run.glob('metrics_rank*.jsonl')):
            completed.append(json.loads(path.read_text().splitlines()[-1])['step'] + 1)
        # Hash the preserved files after GPUs are released; this also reads all bytes.
        checksums = {}
        for path in sorted(archive.rglob('*')):
            if path.is_file():
                with path.open('rb') as f:
                    checksums[str(path.relative_to(archive))] = hashlib.file_digest(f, 'sha256').hexdigest()
        command = json.loads((ROOT / 'commands/rl.json').read_text())
        command['argv'] += ['--resume-from-lora', str(actor), '--start-step', str(args.step)]
        command['notes'] = 'Prepared only; do not launch until the user requests resume. Require matching actor, critic, and both optimizer states at this step. Keep all original scientific settings and total budget.'
        (archive / 'resume_command.json').write_text(json.dumps(command, indent=2))
        record = dict(phase='paused', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      checkpoint_step=args.step, total_updates=2100, snapshot=str(archive),
                      log=old_status['log'], optimizer_summary=optimizer_summary,
                      tensor_counts=tensor_counts, last_completed_per_rank=completed,
                      updates_after_checkpoint=max(completed)-args.step,
                      stopped_processes=owned, gpu_compute_processes_after_stop=gpu_state,
                      stochastic_resume='Weights, optimizer moments and deterministic data cursor preserved. Exact generation RNG state was not saved by this trainer; continuation is not bitwise identical.')
        (archive / 'sha256.json').write_text(json.dumps(checksums, indent=2))
        (archive / 'pause_manifest.json').write_text(json.dumps(record, indent=2))
        tmp = ROOT / 'paused_status.tmp'
        tmp.write_text(json.dumps(record, indent=2)); tmp.replace(ROOT / 'status.json')
        (ROOT / 'PAUSED.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
