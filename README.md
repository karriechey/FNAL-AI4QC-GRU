<!-- Created: 2026-09-25 | Last updated: 2026-11-01 -->
# FNAL-AI4QC-GRU

Code for the GRU surface-code decoder paper.
The GRU pipeline was carried over from `karriechey/FNAL-QCDecoding-FPGA` at commit
`d5ee86f` (tag `pre-paper-2026-09`); `src/` files name their origin in a header line.
Partition ranges live in `configs/partitions.yaml`, data locations in `configs/paths.yaml`.
The manuscript is on Overleaf.

## Where things are

| Path | Contents |
|---|---|
| `src/` | Shared code: circuits and pools (`circuit_generators.py`, `generate_pools.py`, `gen_train_extension.py`, `fresh_shots.py`), models (`StudentModels.py`), training (`train_student.py`, launcher `gh200_run_student_scaling.sh`), scoring (`eval_student_on_tail.py`, `mwpm_on_block.py`, `pair_two_decoders.py`), MWPM threshold (`threshold_sweep.py`, `threshold_analysis.py`), hls4ml export |
| `experiments/` | One folder per experiment; [`experiments/INDEX.md`](experiments/INDEX.md) lists each with its question, entry script and results |
| `runs/qat_kd5_u140/` | Quantization and FPGA synthesis study of the d5 distilled GRU |
| `figures/` | Figure scripts (`figures/output/`, `figures/poster/` hold the images); `poster_style.py` is the shared style |
| `GRU_*.csv` | Run tables, one per study, mirrored in the GRU_all_runs Google Sheet |
| `notebooks/` | `EXPERIMENTS_LOG.ipynb` (running record), `scaling_laws.ipynb` |
| `prereg/`, `protocol/` | Pre-registered decision rules; checkpoint selection rule |
| `configs/` | Pool partitions and fresh-shot seed ranges (`partitions.yaml`), data paths |
| `tests/` | Pipeline checks (equivalence, loader, resume, guards) |
| `scripts/` | Exports for the visualization tools |
| `env/`, `ENV.md`, `requirements.txt` | Container and Python environments |
