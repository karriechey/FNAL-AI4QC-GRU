#!/usr/bin/env bash
# host CPU python with hls4ml main worktree first on the path
export PYTHONPATH=/tmp/kchey_qat/hls4ml_main CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=2
exec ~/hls4ml_main_env/bin/python "$@"
