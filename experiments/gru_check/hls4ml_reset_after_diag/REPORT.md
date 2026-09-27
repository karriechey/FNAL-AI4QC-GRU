<!-- Created: 2026-09-26 | Last updated: 2026-09-26 -->
# hls4ml and the Keras GRU `reset_after` setting
With hls4ml 1.3.0, `reset_after=False` Keras GRUs are not faithfully
implemented. Current upstream (`main` at `a2abb4d`, 2026-09-25) fixes the bias parsing but
still does not implement the reset-before recurrence equation in the Vivado GRU backend.
Therefore `reset_after=False` can convert and compile while silently producing the wrong
function. `reset_after=True` is reproduced correctly.

For 1.3.0 this is shown by both the source and a numerical test. For upstream `main` it
rests on the source (unchanged GRU templates; nothing reads `apply_reset_gate`), the text of
PR #1534 itself, and an exact numpy emulation; a build of `main` was not run.

Evidence files in this folder:

- `diag_hls4ml_gru_reset_after.py` — the diagnostic
- `run_20260926T235437Z/report.json`, `run_20260926T235437Z/stdout.log` — the final run

## 1. Environment

All conversions ran inside the `qdec:tf215` image (the environment of the C2 result of
record; see `ENV.md`).

| item | value |
|---|---|
| hls4ml | 1.3.0, PyPI wheel (not a git checkout), `/usr/local/lib/python3.10/dist-packages/hls4ml` |
| same version in `~/tfenv` | file-for-file identical source (`diff -rq`) |
| Python | 3.10.12 |
| TensorFlow | 2.15.0+nv24.3 |
| Keras | `tf.keras` = `tf_keras` 2.15.1 (`TF_USE_LEGACY_KERAS=1`); `keras` package 2.15.0 |
| backend | Vivado (hls4ml default), `io_parallel`, `granularity='name'` |
| precision | every type `ap_fixed<32,16>`; GRU activation tables 65,536 entries of `ap_fixed<32,16>` |
| C simulation | `hls_model.compile()`, g++ 11.4 |

## 2. Numerical tests

Controlled test: two otherwise identical models, input (6, 24), `GRU(units=4)` → `Dense(1)`,
seeded, each with Keras' default initialisation (all GRU biases zero) and with fixed random
non-zero biases, on 8 fixed random 0/1 inputs. Real models: 500 validation-block shots at
d=5. Nothing was retrained.

### `reset_after=False`

| | 4 units, zero bias | 4 units, non-zero bias | d=5 u140, 10M run of record | d=5 u140, 20M augmented, seed 0 |
|---|---|---|---|---|
| Keras inference | PASS | PASS | PASS | PASS |
| hls4ml conversion | PASS | PASS | PASS | PASS |
| hls4ml compile | PASS | PASS | PASS | PASS |
| max \|TF − HLS\| | 0.070 | 0.262 | **24.9** | **22.2** |
| mean \|TF − HLS\| | – | – | 8.83 | 8.33 |
| Keras logit std | 0.068 | 0.072 | 7.83 | 7.27 |
| sign agreement | 8 / 8 | 8 / 8 | **64.8 %** | **60.4 %** |

No exception and no hls4ml warning. For the 4-unit models the error is as large as the logits
themselves; 8 / 8 sign agreement there reflects only 8 samples with tiny logits.

### `reset_after=True`

| | 4 units, zero bias | 4 units, non-zero bias | d=5 u140, 20M augmented, seed 0 (`unroll=True`) |
|---|---|---|---|
| Keras / conversion / compile | PASS / PASS / PASS | PASS / PASS / PASS | PASS / PASS / PASS |
| max \|TF − HLS\| | 0.00054 | 0.00057 | 0.134 (mean 0.046) |
| sign agreement | 8 / 8 | 8 / 8 | **100 %** |

The larger absolute error on the trained model is consistent with its pre-activations
(up to about 26) running past the activation tables' [−8, 8] range: a precision effect, not
a wrong equation.

## 3. The bias issue and what PR #1534 fixed

Keras stores the GRU bias as `(3u,)` when `reset_after=False` and as `(2, 3u)` (input bias,
recurrent bias) when `reset_after=True`.

**Installed 1.3.0**, `converters/keras/recurrent.py` lines 56–60, always splits
`bias_data = biases[0]` and `recurrent_bias_data = biases[1]`. For a `(3u,)` bias this gives
two single numbers. Observed in the diagnostic:

- IR bias shapes `[]` for `reset_after=False` (`[12]` / `[420]` for `reset_after=True`)
- generated firmware `b2.h` and `br2.h` hold one value each (12 / 420 for `reset_after=True`)
- the kernel signature (`templates/vivado/nnet_utils/nnet_recurrent.h` lines 472–473) reads
  `param_b[3n]` and `param_br[3n]`, so every entry after the first is read past the end of a
  one-element array (undefined behaviour)

Also, line 47 would fail for `use_bias=False` (`.shape` of `None`).

**Upstream PR #1534** ("Keras frontends: stop silently misparsing unsupported layer
configurations", merge commit `e615a77`, merged into `main` 2026-08-28; not in any release —
the latest release is v1.3.0, 2026-03-20) changes `converters/keras/recurrent.py` lines 63–71
and the Keras 3 converter: for `reset_after=False` the flat bias becomes the input bias and
the recurrent bias is set to zeros.

It fixes the **bias parsing / layout only**, not the recurrence. The PR text: "No backend
supports [`apply_reset_gate='before'`] yet … until the backend-rejection follow-up lands, the
four ingested configurations (… `reset_after=False` …) parse correctly but still compute the
wrong function if compiled."

## 4. The equation issue

Gate order (z, r, h); input kernel W, recurrent kernel U, input bias b, recurrent bias b_r.

| | candidate state |
|---|---|
| Keras `reset_after=False` | h̃ = tanh(W_h·x + b_h + U_h·(r ⊙ h)) |
| Keras `reset_after=True` | h̃ = tanh(W_h·x + b_h + r ⊙ (U_h·h + b_r,h)) |
| hls4ml Vivado GRU | h̃ = tanh(W_h·x + b_h + r ⊙ (U_h·h + b_r,h)) — the `reset_after=True` form, always |

Source, `templates/vivado/nnet_utils/nnet_recurrent.h` (installed 1.3.0; byte-identical on
upstream `main`):

- `gru()` line 421: `tmpres_state_zr = dense(h, U, br)` — U·h + b_r for all three gates
- line 443: `tmpres_state_h = r ⊙ tmpres_state_zr[2n:3n]` — r ⊙ (U_h·h + b_r,h)
- line 450: candidate pre-activation = `tmpres[2n:3n] + tmpres_state_h`
- the same in `gru_static()`, lines 502, 525, 532
- there is no branch and no assertion; the comment at lines 446 / 528 ("Assuming reset_after
  is false") describes the opposite of what the code computes

Why it is unconditional: `apply_reset_gate` exists in the IR (`model/layers.py` line 1435,
choices `before` / `after`, default `after`) and the converter sets it, but no backend reads
it — it does not occur under `backends/` at all, and
`backends/vivado/passes/recurrent_templates.py` never passes it to the template.

**Equation error alone** (numpy, hls4ml not involved): each `reset_after=False` model's
Keras output against the `reset_after=True` equations on the same weights with the bias used
correctly (input bias = the `(3u,)` bias, recurrent bias = 0) — what the Vivado kernel would
compute if only the bias split were fixed, i.e. upstream `main`:

| model | max \|Δ\| | mean \|Δ\| | sign agreement |
|---|---|---|---|
| 4 units, zero bias | 0.043 | 0.019 | 8 / 8 |
| 4 units, non-zero bias | 0.050 | 0.023 | 8 / 8 |
| d=5 u140, 10M run of record | 21.0 | 6.87 | 80.6 % |
| d=5 u140, 20M augmented, seed 0 | 19.3 | 7.23 | 76.6 % |

Fixing the bias alone is not sufficient for trained models.

## 5. Upstream status (checked 2026-09-26)

- `main` inspected at `a2abb4d22e7762a870cd46ccf3e07ff6a5622ed5` (2026-09-25).
  `templates/vivado/nnet_utils/nnet_recurrent.h` and
  `backends/vivado/passes/recurrent_templates.py` are byte-identical to 1.3.0; the template
  last changed on 2025-07-23 (#1310).
- No merged PR changes the recurrence equation. The backend-rejection follow-up mentioned in
  PR #1534 was not found among the 40 most recently updated open or closed pull requests.
- `reset_after=False` is **silently accepted** in 1.3.0 and in `main`: no error, no warning.

## 6. Implications

- **`reset_after=True` is the hls4ml-compatible choice today.** It is the only mode the
  Vivado GRU kernel implements; a trained u140 model reproduces Keras with 100 % sign
  agreement.
- **Deploying `reset_after=False` would need both** the converter fix (already on upstream
  `main`, or backported) **and** a backend change: compute U_zr·h for the z and r gates, form
  r ⊙ h, then a separate n×n product U_h·(r ⊙ h) without b_r. That means splitting the
  recurrent kernel into its (z, r) and h parts, adding a third dense configuration in
  `recurrent_templates.py`, passing `apply_reset_gate` through, and implementing the branch in
  `gru()`, `gru_static()` and the stream `gru_stack` variant. Vitis reuses the Vivado templates;
  Quartus, oneAPI and Catapult have their own copies.
- **Size:** localized for one backend (roughly 50–100 lines in `nnet_recurrent.h` and
  `recurrent_templates.py`, plus tests); larger if every backend needs it.
- **Not a fundamental FPGA limitation.** `reset_after=False` needs one extra element-wise step
  and a separate n×n matrix product before the candidate activation — a small latency cost,
  nothing unimplementable. It should be described as a limitation of hls4ml (1.3.0, and
  `main` as of 2026-09-25).

## 7. Files

| where | path | what |
|---|---|---|
| repo | `experiments/gru_check/hls4ml_reset_after_diag/diag_hls4ml_gru_reset_after.py` | diagnostic script |
| repo | `experiments/gru_check/hls4ml_reset_after_diag/REPORT.md` | this report |
| repo | `experiments/gru_check/hls4ml_reset_after_diag/run_20260926T235437Z/{report.json,stdout.log}` | final-run evidence |
| grace1 | `~/FNAL-AI4QC-GRU/experiments/gru_check/hls4ml_reset_after_diag/run_20260926T235437Z/` | final run including the generated hls4ml projects `hls_*` (19 MB, not in git) |
| grace1 | `~/FNAL-AI4QC-GRU/experiments/gru_check/hls4ml_reset_after_diag/run_20260926T235130Z/` | first run (4-unit and real-model tests, without the equation-only test; 19 MB) |
| grace1 | `/tmp/hls4ml_img/hls4ml/` | copy of the installed hls4ml source used for inspection |

A third run folder, a failed attempt to repair the bias inside hls4ml's in-memory model
(the Vivado writer rejects added weight variables: `'WeightVariable' object has no attribute
'storage'`), was deleted; the numpy equation-only test replaced it.

Reproduce (on grace1, from this folder):

```bash
podman run --rm --security-opt=label=disable -v $HOME:$HOME -e HOME=$HOME \
  -w $HOME/FNAL-AI4QC-GRU/experiments/gru_check/hls4ml_reset_after_diag qdec:tf215 \
  python3 diag_hls4ml_gru_reset_after.py --out run_<stamp> --real
```
