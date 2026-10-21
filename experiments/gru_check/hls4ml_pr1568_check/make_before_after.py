# Created: 2026-09-27
# Last modified: 2026-09-27

"""Before/after summary of hls4ml PR #1568 from two check_pr1568.py runs (1.3.0 vs PR @ e1721b3).

Writes BEFORE_AFTER.md and before_after_decision_agreement.png/.pdf; every value is read from report.json.

python make_before_after.py <before_run_dir> <after_run_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = 'u140_reset_after_False_10M_run_of_record__default_tables'
SUPPORT = ['u140_reset_after_False_20M_aug_seed0__default_tables',
           'u70_reset_after_False_gkd20m_hard_seed0__default_tables',
           'u140_reset_after_True_unroll_20M_aug_seed0__default_tables']
BLUE, RED, INK, MUTED = '#2a78d6', '#e34948', '#2b2b2a', '#6b6b67'


def load(run):
    rep = json.load(open(os.path.join(run, 'report.json')))
    return rep, rep['results']


def row(label, r):
    return (f"| {label} | {r['max_abs_diff']:.3f} | {r['mean_abs_diff']:.4f} | "
            f"{100 * r['sign_agreement']:.2f} % | {r['decisions_differ']:,} / {r['n']:,} | "
            f"`{r['firmware_reset_after']}` |")


def main(before_dir, after_dir):
    rb, b = load(before_dir)
    ra, a = load(after_dir)
    assert rb['shots'] == ra['shots'], (rb['shots'], ra['shots'])
    assert b[MAIN]['weights'] == a[MAIN]['weights']
    n = a[MAIN]['n']
    hdr = ('| hls4ml | max \\|Δlogit\\| | mean \\|Δlogit\\| | decision agreement | '
           'decisions that differ | generated `reset_after` |\n|---|---|---|---|---|---|')
    md = [
        '<!-- Created: 2026-09-27 | Last updated: 2026-09-27 -->',
        '# hls4ml PR #1568: GRU `reset_after=False`, before vs after',
        '',
        'Trained d=5 GRU decoder (u140, `reset_after=False`, run of record, seed 0; '
        f"checkpoint `{os.path.basename(a[MAIN]['weights'])}`), converted with the Vivado "
        'backend, `io_parallel`, every precision `ap_fixed<32,16>`, GRU activation tables of '
        '65,536 entries (hls4ml default input ranges). Reference: the same Keras model on the '
        f"CPU (no TF32). Inputs: {ra['shots']} ({n:,} shots), identical for both versions.",
        '',
        f"- before: upstream hls4ml `{rb['hls4ml']}`",
        f"- after: hls4ml `{ra['hls4ml']}` (PR #1568, commit `e1721b3`)",
        '',
        hdr,
        row(f"{rb['hls4ml']} (before)", b[MAIN]),
        row('PR #1568 (after)', a[MAIN]),
        '',
        'Decision = sign of the logit (logit > 0 predicts a logical flip). "Generated '
        '`reset_after`" is read from the generated `firmware/parameters.h`; hls4ml 1.3.0 '
        'writes no such setting, and its GRU kernel always computes the `reset_after=True` '
        'form. Keras logit standard deviation on these shots: '
        f"{a[MAIN]['keras_logit_std']:.2f}.",
        '',
        '## Supporting checks (same shots and settings; not part of the main comparison)',
        '',
        '| model | hls4ml | max \\|Δlogit\\| | mean \\|Δlogit\\| | decision agreement | '
        'decisions that differ | generated `reset_after` |\n|---|---|---|---|---|---|---|',
    ]
    for k in SUPPORT:
        label = k.replace('__default_tables', '').replace('_', ' ')
        for tag, src in (('1.3.0', b), ('PR #1568', a)):
            if k in src and 'error' not in src[k]:
                md.append(f'| {label} | {tag} ' + row('', src[k])[3:])
    md += ['', 'Figure: `before_after_decision_agreement.png` / `.pdf`.',
           f"Source runs: `{os.path.basename(before_dir)}` (before), "
           f"`{os.path.basename(after_dir)}` (after)."]
    open(os.path.join(HERE, 'BEFORE_AFTER.md'), 'w').write('\n'.join(md) + '\n')

    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': MUTED, 'xtick.color': INK, 'ytick.color': MUTED,
                         'text.color': INK, 'axes.labelcolor': INK})
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    vals = [100 * b[MAIN]['sign_agreement'], 100 * a[MAIN]['sign_agreement']]
    diffs = [b[MAIN]['decisions_differ'], a[MAIN]['decisions_differ']]
    labels = [f"hls4ml {rb['hls4ml']}", 'hls4ml PR #1568']
    bars = ax.bar(labels, vals, color=[RED, BLUE], width=0.55)
    for bar, v, dcount in zip(bars, vals, diffs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v:.2f} %', ha='center',
                va='bottom', fontsize=13, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2, f'{dcount:,} of {n:,}\ndecisions differ',
                ha='center', va='center', fontsize=11, color='white')
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('decision agreement with Keras (%)')
    ax.set_title('GRU reset_after=False, same trained model\nand same 10,000 evaluation shots',
                 fontsize=12)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'before_after_decision_agreement.png'), dpi=200)
    fig.savefig(os.path.join(HERE, 'before_after_decision_agreement.pdf'))
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
