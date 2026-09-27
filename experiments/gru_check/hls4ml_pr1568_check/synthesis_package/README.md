<!-- Created: 2026-09-27 | Last updated: 2026-09-27 -->
# GRU `reset_after=True` vs `reset_after=False`: hls4ml projects for C synthesis

Two hls4ml projects of the same GRU decoder, identical except for the Keras GRU
`reset_after` setting, generated with hls4ml pull request
[#1568](https://github.com/fastmachinelearning/hls4ml/pull/1568) ("Vivado: correctly
implement GRU reset_after=False"), which adds the reset-before candidate-state equation to
the Vivado GRU kernel.

## Request

Please run C synthesis on both projects and report, for each:

- latency (clock cycles; min / max, and the initiation interval)
- resources: LUT, FF, DSP, BRAM (and URAM if used)
- the part and clock period used, if you prefer a different default from the one below

The comparison of interest is `reset_after=True` vs `reset_after=False` at the same
precision, part, clock, strategy and reuse factor.

## Contents

```
hls_projects/gru_u140_reset_after_False/   project source (reset_after=False)
hls_projects/gru_u140_reset_after_True/    project source (reset_after=True)
    firmware/                 myproject.cpp/.h, defines.h, parameters.h, weights/, nnet_utils/, ap_types/
    myproject_test.cpp        C testbench
    tb_data/                  100 input vectors and the Keras logits for them
    build_prj.tcl, project.tcl, vivado_synth.tcl, build_lib.sh
    hls4ml_config.yml         the full hls4ml configuration used
models/gru_u140_reset_after_False.keras    trained Keras model
models/gru_u140_reset_after_True.keras     trained Keras model
convert_for_synthesis.py                   the conversion script that generated the projects
generation_summary.json                    versions and settings recorded at generation
reproduce/check_pr1568.py                  standalone numerical check (any installed hls4ml)
reproduce/eval_inputs_10000.npy            10,000 inputs, shape (10000, 6, 24), float32 0/1
reproduce/keras_logits_reset_after_False.npy   Keras FP32 CPU logits of the False model, shape (10000,)
reproduce/keras_logits_reset_after_True.npy    Keras FP32 CPU logits of the True model, shape (10000,)
u70/gru_u70_reset_after_False.keras        the earlier 70-unit model, as a full Keras model
u70/gru_u70_reset_after_False.weights.h5   the same model's weights file exactly as sent before
u70/keras_logits_u70_reset_after_False.npy Keras FP32 CPU logits of the u70 model, shape (10000,)
```

Source only: no compiled libraries or simulation binaries (the projects were written with
`hls_model.write()`, never compiled).

## hls4ml version

| item | value |
|---|---|
| source | PR #1568, head branch `fix/vivado-gru-reset-before` |
| commit | `e1721b302f8a29397b532b82f21dc8d908781a70` (based on upstream `main` `a2abb4d`) |
| reported version | `1.3.0.post1568+ge1721b302f8a29397b532b82f21dc8d908781a70` |
| TensorFlow / Keras | 2.15.0 / tf_keras 2.15.1 |

## Model

| item | value |
|---|---|
| task | surface-code decoder, distance 5, 5 rounds (one output logit per shot; logit > 0 = logical flip) |
| input | `(6, 24)`: 6 timesteps (detector layers) x 24 plaquette positions, 0/1 values |
| layers | `GRU(140, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)` -> `Dense(1)` (linear) |
| `reset_after=False` | 69,441 parameters; GRU bias shape (420,) |
| `reset_after=True` | 69,861 parameters; GRU bias shape (2, 420) |

Both were trained on the same data (20M simulated shots) with the same seed; they are the
checkpoints with the best validation score. The `reset_after=True` model was trained with
Keras `unroll=True`, which does not change its equations or weights and is not used by
hls4ml.

## hls4ml settings (both projects)

| setting | value |
|---|---|
| backend | Vivado |
| io type | `io_parallel` |
| precision | `ap_fixed<32,16>` for every type (`default_precision`; `granularity='name'`) |
| GRU activation tables | 65,536 entries of `ap_fixed<32,16>`; hls4ml's default input ranges |
| strategy / reuse factor | Latency / 1 (hls4ml defaults), GRU recurrent reuse factor 1 |
| part / clock | `xcvu13p-flga2577-2-e`, 5 ns (hls4ml defaults) |
| generated `reset_after` | `false` / `true` in `firmware/parameters.h` of the respective project |

Latency strategy with reuse factor 1 is fully parallel; for a 140-unit GRU this is likely
to be large and slow to synthesize. If you use a different strategy or reuse factor, please
use the same for both projects and report it. The projects can be regenerated with other
settings by editing the `hls4ml` configuration in `convert_for_synthesis.py`.

Numerical check with the same hls4ml build (C simulation, 10,000 evaluation shots, against
Keras on the CPU): decisions differ from Keras on 1 / 10,000 shots for each of these two
models.

## Reproduce the before/after table

`reproduce/check_pr1568.py` converts a `.keras` model with whatever hls4ml is installed (same
settings as above: Vivado, `io_parallel`, `ap_fixed<32,16>`, GRU tables 65,536 entries),
compiles the C simulation, runs the 10,000 inputs and compares with the stored Keras logits.
It needs TensorFlow 2.15 / Keras 2 (the models were saved with tf_keras 2.15) and a C++
compiler.

The inputs are the first 10,000 shots of our held-out evaluation set for this decoder
(distance 5, 5 rounds, p = 0.004): the same shots as the recorded results below. Decision =
logit > 0.

Run it once with stock hls4ml 1.3.0 and once with PR #1568 (commit `e1721b3`):

```bash
cd reproduce
python check_pr1568.py --model ../models/gru_u140_reset_after_False.keras \
    --inputs eval_inputs_10000.npy --expected keras_logits_reset_after_False.npy
python check_pr1568.py --model ../models/gru_u140_reset_after_True.keras \
    --inputs eval_inputs_10000.npy --expected keras_logits_reset_after_True.npy
```

Expected results (recorded with these exact files; Keras logit standard deviation 7.47 for
the False model and 7.82 for the True model):

| hls4ml | model | max \|Δlogit\| | mean \|Δlogit\| | decision agreement | decisions that differ | generated `reset_after` |
|---|---|---|---|---|---|---|
| 1.3.0 | `reset_after=False` | 23.753 | 8.3937 | 61.65 % | 3,835 / 10,000 | not set |
| PR #1568 | `reset_after=False` | 0.113 | 0.0106 | 99.99 % | 1 / 10,000 | `false` |
| 1.3.0 | `reset_after=True` | 0.148 | 0.0452 | 99.99 % | 1 / 10,000 | not set |
| PR #1568 | `reset_after=True` | 0.148 | 0.0452 | 99.99 % | 1 / 10,000 | `true` |

The earlier 70-unit model (`u70/`) is the same kind of decoder: d=5, input `(6, 24)`,
`GRU(70, reset_after=False)` -> `Dense(1)`, 20,021 parameters, trained on 20M shots with hard
labels, seed 0. `gru_u70_reset_after_False.weights.h5` is byte-identical to the weights file
sent before (SHA-256 `95d1d3e3...`); the `.keras` file is the same model with its
architecture included. Check it the same way:

```bash
python check_pr1568.py --model ../u70/gru_u70_reset_after_False.keras \
    --inputs eval_inputs_10000.npy --expected ../u70/keras_logits_u70_reset_after_False.npy
```

| hls4ml | model | max \|Δlogit\| | mean \|Δlogit\| | decision agreement | decisions that differ | generated `reset_after` |
|---|---|---|---|---|---|---|
| 1.3.0 | u70 `reset_after=False` | 45.633 | 17.5870 | 20.58 % | 7,942 / 10,000 | not set |
| PR #1568 | u70 `reset_after=False` | 0.127 | 0.0185 | 99.99 % | 1 / 10,000 | `false` |

(Keras logit standard deviation 7.78.) No hls4ml project is included for the u70; the
synthesis request concerns the 140-unit pair.

hls4ml 1.3.0 writes no `reset_after` setting; its GRU kernel always computes the
`reset_after=True` equations, so the `reset_after=False` model converts and compiles but
computes a different function. A different compiler or platform may change the last digits
of the logit errors and, rarely, a decision close to the threshold.

## How the projects were generated

In the `qdec:tf215-hlsfix` container (TensorFlow 2.15.0 image with hls4ml installed from the
commit above), from this folder:

```bash
python3 convert_for_synthesis.py --out .
```

The script builds each model, loads its trained weights, saves the `.keras` file, converts
it with the settings above and calls `hls_model.write()`. The absolute output paths that
hls4ml records in `hls4ml_config.yml` were replaced by relative ones afterwards; they are not
used by synthesis.

## Running synthesis

From a project folder, e.g. `hls_projects/gru_u140_reset_after_False/`:

```bash
vitis_hls -f build_prj.tcl "reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 vsynth=0"
```

(`vivado_hls` accepts the same command.) The synthesis report is written to
`myproject_prj/solution1/syn/report/myproject_csynth.rpt`.
