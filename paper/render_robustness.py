"""Render the manuscript's robustness table/figures from saved measured summaries.

This publication helper is new; it does not claim to be the original figure source.
It performs no inference, API calls, resampling, or changes to reported estimates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MODELS = {'qwen7b': 'Qwen2.5-7B', 'gemma12b': 'Gemma-3-12B', 'gemma27b': 'Gemma-3-27B'}
ARMS = {
    'exp1': 'Remove all quoted content',
    'exp2': 'Keep only quoted content',
    'exp3a': 'Remove input-redundant quotes (string)',
    'exp3b': 'Remove input-redundant quotes (semantic)',
    'exp4a': 'Remove quotation delimiters',
    'exp4b': 'Paraphrase quoted content',
    'exp5a': 'Block shuffle',
    'exp5b': 'Sentence shuffle',
    'exp5c': 'Quoted-claim shuffle',
    'exp6a': 'Source replacement (including read token)',
    'exp6b': 'Source replacement (excluding read token)',
    'exp6f': 'Shuffle the AV’s own context quote',
    'exp6r': 'Random same-document span',
    'exp6o': 'Other-document replacement',
    'exp6p': 'Paraphrase source replacement',
    'exp6t': 'True-continuation replacement',
}
TABLE_ARMS = ('exp1', 'exp3a', 'exp3b', 'exp4b', 'exp5a', 'exp5b', 'exp5c', 'exp6a')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path(__file__).resolve().parents[1] / 'analysis/fve_robustness/results/dashboard_inference.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    assert set(ARMS) == set(data['arms']), 'Unexpected or missing manipulation arms'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, model in MODELS.items():
        baseline = data['baseline'][key]
        rows.append(dict(model=model, arm='baseline', label='Baseline FVE', estimate=baseline['FVE'], low=baseline['FVE_ci95'][0], high=baseline['FVE_ci95'][1], quantity='absolute_FVE'))
        for arm in TABLE_ARMS:
            item = data['arms'][arm][key]
            rows.append(dict(model=model, arm=arm, label=ARMS[arm], estimate=item['dFVE'], low=item['dFVE_ci95'][0], high=item['dFVE_ci95'][1], quantity='delta_FVE'))
    with (args.output_dir / 'robustness_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    colors = ['#2166ac', '#b35806', '#1b7837']
    for absolute, filename in [(False, 'nla_robustness_delta_fve_all_v2'), (True, 'nla_robustness_absolute_fve_all')]:
        arms = (['baseline'] if absolute else []) + list(ARMS)
        fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharey=True, sharex=True, layout='constrained')
        for ax, (key, model), color in zip(axes, MODELS.items(), colors):
            values, low, high = [], [], []
            for arm in arms:
                item = data['baseline'][key] if arm == 'baseline' else data['arms'][arm][key]
                field = 'FVE' if arm == 'baseline' else ('FVE_arm' if absolute else 'dFVE')
                value = item[field]; ci = item[field + '_ci95']
                assert ci[0] <= value <= ci[1]
                values.append(value); low.append(value-ci[0]); high.append(ci[1]-value)
            ax.errorbar(values, np.arange(len(arms)), xerr=[low, high], fmt='o', markersize=4, capsize=2, color=color)
            ax.axvline(0, color='0.55', linewidth=.7)
            if absolute: ax.axvline(data['baseline'][key]['FVE'], color=color, linestyle='--', linewidth=.8)
            ax.set_title(model); ax.set_xlabel('FVE' if absolute else 'Change in FVE')
            ax.grid(axis='x', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_yticks(np.arange(len(arms)), ['Baseline' if arm == 'baseline' else ARMS[arm] for arm in arms])
        axes[0].invert_yaxis()
        fig.savefig(args.output_dir / (filename + '.pdf'))
        fig.savefig(args.output_dir / (filename + '.png'), dpi=150)
        plt.close(fig)
    (args.output_dir / 'provenance.json').write_text(json.dumps({'source': str(args.summary), 'sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(), 'note': 'Figures and CSV rendered from existing measured estimates and confidence intervals; no new experiment.'}, indent=2)+'\n')
    print(f'Wrote two PDF/PNG figures and table CSV to {args.output_dir}')


if __name__ == '__main__':
    main()
