<!-- Created: 2026-09-27 | Last updated: 2026-09-27 -->
# SHA-256 of the uncommitted PR #1568 check artifacts

The tarballs, trained models and test data below are not in git (binary files; `*.tar.gz`,
`.keras`, `.npy` and `.h5` are ignored). This file records their SHA-256 so any copy can be
checked against the files used for the recorded results:

```bash
sha256sum <file>          # Linux
shasum -a 256 <file>      # macOS
```

Files live on the GPU host under
`experiments/gru_check/hls4ml_pr1568_check/` (tarballs) and
`experiments/gru_check/hls4ml_pr1568_check/synthesis_package/` (everything else).

## Tarballs sent for C synthesis (2026-09-27)

| file | bytes | SHA-256 |
|---|---:|---|
| `gru_models_minimal.tar.gz` | 708,504 | `246fe37e4abe711c22dec20755b21373054051b01ec5f22532efa4b5bc982235` |
| `gru_reset_after_hls_projects.tar.gz` | 4,475,543 | `05d535f8702912153ec13dfe847bd883aec489b1f05cb44e8402d9205118ff58` |

`gru_models_minimal.tar.gz` holds the three `.keras` models, `eval_inputs_10000.npy` and a
README. `gru_reset_after_hls_projects.tar.gz` holds the full synthesis package (README,
conversion script, both hls4ml projects, models, reproduce/ and u70/). Every model and
`.npy` file inside either tarball has the same SHA-256 as the copy listed below (checked
2026-09-27 by extracting each member to stdout and hashing it).

## Models

| file | SHA-256 |
|---|---|
| `models/gru_u140_reset_after_False.keras` | `130c2e5c9c86ee95d73c761db7b8e45e467f9ee3ee009bd13762d14aa343f837` |
| `models/gru_u140_reset_after_True.keras` | `0015d5ea685e70a4c17b396cd245d602abf4c404c7926f96da5604a513d89c2e` |
| `u70/gru_u70_reset_after_False.keras` | `6271fe29ebd52c36194a60fe60407cca7580c02761b40023545d7c94c7b21762` |
| `u70/gru_u70_reset_after_False.weights.h5` | `95d1d3e39f04082717249a7aeb5c56376def518cf008974ce07e30521a1b020c` |

The u70 `.weights.h5` is byte-identical to the weights file sent earlier for synthesis.

## Test inputs and expected Keras logits

| file | SHA-256 |
|---|---|
| `reproduce/eval_inputs_10000.npy` | `e258623761f8c90b00d5e10d0725e864e89cbbcf85dfd6eba2f017227724dab4` |
| `reproduce/keras_logits_reset_after_False.npy` | `45bfd75f67ff97c260077c00913798dd65e74d9425a2ce56afbcb7858a3c8412` |
| `reproduce/keras_logits_reset_after_True.npy` | `8e8b2c9e0386d004c9d4582125327d95efb89e36935c3e5bb7347fd348f3d65b` |
| `u70/keras_logits_u70_reset_after_False.npy` | `e5569e8302eaed47e753bfe050079c1d96fa535a2b1ee50aac921243b35ebfca` |

`eval_inputs_10000.npy` is the first 10,000 shots of the d=5 evaluation block
[15,200,000, 15,210,000), written by `make_reproduce_data.py`.
