<!-- Created: 2026-11-01 | Last updated: 2026-11-01 -->
# Experiments index

One folder per experiment, named by topic and start date. Folders reference each other by relative
path (and grace1 runs from the same layout), so they stay where they are; this page is the map.
Shared code is in `src/`, figure scripts in `figures/`, run tables in the top-level `GRU_*.csv` files.
Terms: FORMAL pool = the fixed shot pool split into train [0, 15M), val [15M, 15.2M), eval [15.2M, 17M) and
sealed [17M, 19M) blocks; selection block = [0, 10M) of that pool, used to pick checkpoints.

## Current studies (paper)

| Folder | Question | Start here | Results |
|---|---|---|---|
| `depth_vs_distance/` | Does the GRU depth needed to beat MWPM grow with d? (d5, d7; d9 paused) | `PLAN.md`, `queue.sh`, `run_dvd.sh`, `dvd.py` | `STATUS.md`, `GRU_depth_vs_distance_runs.csv`, `figures/fig_depth_vs_distance.py` |
| `xbasis_p004_2026-10/` | Same teacher recipes in X-basis memory, Z vs X per d | `PLAN.md`, `queue_x.sh`, `run_x.sh` | `GRU_xbasis_runs.csv`, `zx_comparison.md` |
| `mwpm_correlated_2026-10-29/` | Correlated vs uncorrelated MWPM vs the best GRU, d5/7/9 | `README.md`, `run_all.sh` | `README.md` |
| `threshold_sweep_2026-10-27/` | MWPM threshold sweep (sinter) | `src/threshold_sweep.py`, `src/threshold_analysis.py` | `sinter_stats.csv`, `figures/threshold_figure.py` |
| `depth_scaling_d7_2026-10/` | GRU depth L = 1-5 at d7, width 70 | `README.md`, `launch_depth_sweep.sh` | `depth_sweep_report.md`, `GRU_depth_scaling_runs_2026-10-23.csv` |
| `d9_deep_students_2026-10/` | Deep narrow d9 GRUs | `README.md`, `launch.sh` | `GRU_d9_deep_students_runs_2026-10-27.csv` |
| `kd_pilot_p004_2026-10/` | Distillation pilot at p = 0.004, FP32 | `README.md`, `launch.sh` | `GRU_distillation_runs_2026-10-19.csv` |
| `kd_sweep_d5_u140_2026-10-31/` | d5 u140 distillation sweep over alpha x T | `README.md`, `queue.sh` | `report.py` output in the folder |
| `kd_round2_d5_2026-11-01/` | d5 round 2: GRU 70 from the best sweep student | `README.md`, `run_r2.sh` | `report_r2.py` output in the folder |
| `../runs/qat_kd5_u140/` | kd5_u140 quantization (PTQ, QAT), hls4ml C-sim, Vivado synthesis | `NOTES.md`, `SUMMARY.md` | `SUMMARY.md`, `results/` |
| `d7_4x70_hls_2026-10-29/` | hls4ml gate and synthesis for the stacked d7 4 x GRU 70 | `README.md` | `README.md` |
| `selection_2026-10-14/` | Checkpoint selection on the selection block, eval-block scoring | `score_selection.py`, `eval_selected.sh` | `selection_scores.csv`, `eval_selected.csv` |
| `mwpm_fixes_d9_2026-10/` | d9 runs 9-1 to 9-11 (fixes to beat MWPM at d9); also holds the kill rule | `driver.py`, `run_one.py` | `results.tsv` |
| `ceiling_probe_d7_2026-10-05/` | Wider and deeper teachers (f2280, f3280, f3280e240) | `run_arm*.sh` | `train_slice_scores.csv` |

## hls4ml and FPGA

| Folder | Question | Start here |
|---|---|---|
| `u70_hls_baseline_2026-09-29/` | First u70 GRU through hls4ml: inspect, convert, C-sim | `stage1_inspect_eval.py`, `stage3_4_convert_csim.py` |
| `u70_hls_main_2ed80fb9_2026-10-03/` | u70 on hls4ml main: C-sim gate, PTQ width sweep, QGRU check | `README.md` |
| `gru_check/` | reset_after and PR 1568 checks of the hls4ml GRU | `check_pr1568.py` |

## Earlier training runs (Sept to early Oct 2026)

| Folder | What it ran |
|---|---|
| `queue12_2026-09-30/` | 12-run queue R01-R17: width, data size, LR and p at d5/7/9 (`results.tsv`) |
| `scaling100M_2026-09-29/` | 100M-shot runs with the K1/K2 recipe |
| `foundational_2026-09-29/` | F1-F5 at d7, 20M shots: width and LR |
| `cosine_2026-09-27/` | Cosine schedule K1-K3 at d7 (`loss_curves.png`) |
| `gclip_probe_2026-09-27/` | Gradient-norm probe that set global clipnorm 0.64 |
| `overnight_2026-09-27/`, `morning_2026-09-27/` | First d7 u280 / u140 runs and seed repeats |
| `queue_ra1_20M/`, `control_ra0/` | reset_after = True queue at 20M and its reset_after = False control |
| `overfit_test_2026-10-05/`, `train_slice_eval_2026-10-05/` | Small fixed data for 300 epochs; scores on the training slice |
| `summary_2026-09-29/` | Overview plot of the September runs |

## Pools, infrastructure checks, setup

| Folder | What it checked |
|---|---|
| `stage0/`, `stage1/` | Repository setup plan (MWPM threshold for this noise model) |
| `pools_p003_2026-10-14/`, `pools_p005_2026-10-15/`, `queue12_pools_2026-09-30/`, `pools_trainonly/` | Pool generation scripts and logs |
| `a5_equivalence/`, `a6_env/`, `a7_smoke/` | New repo vs old repo outputs, environment check, smoke training run |
| `loader_checks/`, `loader_prealloc_test/`, `compact_pool_check/`, `multi_pool_checks/` | Data loader correctness |
| `memory_growth_test/`, `memory_settings_test/` | Host memory growth (MALLOC_ARENA_MAX=2) and loader memory knobs |
| `resume_exactness/`, `queue_smoke/` | Resume from snapshot bit for bit; queue launcher smoke test |
