<!-- Created: 2026-09-25 | Last updated: 2026-09-26 -->
# Environment

Everything runs on **grace1.fnal.gov**: NVIDIA GH200 480GB (one GPU, 97,871 MiB), Grace CPU
(aarch64, Neoverse-V2, 72 cores), driver 595.71.05 / CUDA 13.2. Checked 2026-09-26 with
`tests/a6_env_check.py`; the reports are in `experiments/a6_env/`.

There are two environments. Both passed the full A5 equivalence suite
(`experiments/a5_equivalence/`), including identical logits between the old and new code.

## CPU: `~/tfenv` (uv venv)

Used for Stim sampling, PyMatching, sinter (Stage 0 sweep), and the hls4ml conversion
check. TensorFlow here is the PyPI aarch64 wheel, which has no CUDA.

| package | version |
|---|---|
| Python | 3.11.16 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| stim | 1.16.0 |
| pymatching | 2.4.0 |
| sinter | 1.16.0 |
| tensorflow | 2.15.1 (CPU) |
| keras | 2.15.0 |
| tensorflow-model-optimization | 0.7.3 |
| qkeras | 0.9.0 |
| hls4ml | 1.3.0 |
| pyyaml | 6.0.3 |
| pytest | 9.1.1 |

`requirements.txt` pins this environment. Install into it with
`uv pip install --python ~/tfenv/bin/python -r requirements.txt` (the venv has no `pip`).

## GPU: podman image `qdec:tf215`

Used for all GPU training. Built from `env/Containerfile` on top of
`nvcr.io/nvidia/tensorflow:24.03-tf2-py3`.

| package | version |
|---|---|
| Python | 3.10.12 |
| tensorflow | 2.15.0+nv24.3 (CUDA 12.4, cuDNN 9) |
| tf_keras | 2.15.1 (active: `TF_USE_LEGACY_KERAS=1` is set image-wide) |
| tensorflow-model-optimization | 0.8.1 |
| qkeras | 0.9.0 |
| numpy | 1.24.4 |
| scipy | 1.12.0 |
| stim / pymatching / sinter | 1.16.0 / 2.4.0 / 1.16.0 |
| hls4ml | 1.3.0 |
| pyyaml / pytest | 6.0.1 / 9.1.1 |

Run it with both SELinux flags, or the GPU is invisible and bind mounts look empty:

```bash
podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all \
  -v $HOME:$HOME -e HOME=$HOME -w $HOME/FNAL-AI4QC-GRU qdec:tf215 <command>
```

The previous image is kept as `qdec:tf215-bak-2026-09-26`.

## Differences between the two environments

- **Keras package.** In the image, tensorflow-model-optimization 0.8.1 (imported by qkeras)
  requires legacy Keras, so `tf.keras` is `tf_keras` 2.15.1 for every process. In `tfenv`,
  tensorflow-model-optimization 0.7.3 does not, so `tf.keras` is `keras` 2.15.0. Both are
  the Keras 2.15 code line. Within each environment the old and new code give identical
  logits; logits have not been compared across the two environments.
- **numpy** 1.26.4 (tfenv) vs 1.24.4 (image); **Python** 3.11 vs 3.10.
- **Stim pools are architecture-specific.** The same seed gives different shots on x86-64
  (EAF) and aarch64 (grace1). The FORMAL pools were generated on grace1; the A5 tests
  regenerate their first 1,000,000 shots bit-for-bit in both environments.
