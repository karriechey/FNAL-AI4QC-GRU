# Created: 2026-10-18
# Last modified: 2026-10-18
# Selected p = 0.004 teachers (protocol/selection_rule.md). Sourced with D set.
# T_EVAL_ERR: logged eval-block errors; T_SEL_ERR: selection-block errors; T_FSB: student seed base.
PROBE=$HOME/FNAL-AI4QC-GRU-fresh2/experiments/ceiling_probe_d7_2026-10-05
case "$D" in
  5) T_NAME=f5w_p004; T_UNITS="280"; T_PARAMS=256481; T_EPOCH=80; T_EVAL_ERR=8862; T_SEL_ERR=48990
     T_SRC=$PROBE/f5w_p004_u280_20261014T035415Z/seed0/ckpt/gru_d5_p004_u280_fresh100000000_cos3e-5_gclip0.64_hard_seed0_ntr0_b10000.lastepoch.weights.h5
     T_FSB=3000000000000 ;;
  7) T_NAME=f7w_p004; T_UNITS="280 280"; T_PARAMS=747881; T_EPOCH=80; T_EVAL_ERR=4878; T_SEL_ERR=27476
     T_SRC=$PROBE/f7w_p004_u280x280_20261014T035645Z/seed0/ckpt/gru_d7_p004_u280x280_fresh100000000_cos3e-5_wu3_gclip0.64_hard_seed0_ntr0_b10000.lastepoch.weights.h5
     T_FSB=4000000000000 ;;
  9) T_NAME=f3280e240; T_UNITS="280 280 280"; T_PARAMS=1246001; T_EPOCH=240; T_EVAL_ERR=2543; T_SEL_ERR=14285
     T_SRC=$PROBE/f3280e240_u280x280x280_20261011T064457Z/seed0/ckpt/gru_d9_p004_u280x280x280_fresh100000000_cos3e-5_wu3_gclip0.64_hard_seed0_ntr0_b10000.lastepoch.weights.h5
     T_FSB=5000000000000 ;;
  *) echo "D must be 5, 7 or 9"; exit 1 ;;
esac
PILOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T_COPY=$PILOT/teachers/${T_NAME}_epoch${T_EPOCH}.weights.h5
POOL=$HOME/pools_d${D}_p004_19M/data_d${D}_p0.004_r${D}_FORMAL.npz
G0=$PILOT/teacher_check/d$D
