#!/usr/bin/env bash
# Created: 2026-10-28 | Last updated: 2026-10-28
# C synthesis of the exported hls4ml project (x86 Linux, Vitis HLS on PATH), then LUT, FF, DSP,
# BRAM, URAM, latency, II and clock into synth_results.csv.
#   bash run_synth.sh                 the project as exported (part and clock in target.json)
#   RECONVERT=1 bash run_synth.sh     regenerate project/ from keras_model.json + hls4ml_config.json
#                                     first (needs the hls4ml commit in versions.json installed)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ -n "${RECONVERT:-}" ]; then
  python3 - <<'EOF'
import json, shutil, tensorflow as tf, hls4ml
t = json.load(open('target.json'))
m = tf.keras.models.model_from_json(open('keras_model.json').read())
m.load_weights('keras_on_grid.weights.h5')
shutil.rmtree('project', ignore_errors=True)
hm = hls4ml.converters.convert_from_keras_model(m, hls_config=json.load(open('hls4ml_config.json')), output_dir='project',
     project_name='myproject', backend=t['backend'], io_type=t['io_type'], part=t['part'], clock_period=t['clock_period_ns'])
hm.write()
EOF
fi
cd project
cat > build_opt.tcl <<'EOF'
array set opt {
    reset      1
    csim       0
    synth      1
    cosim      0
    validation 0
    export     0
    vsynth     0
    fifo_opt   0
    pnr        0
}
EOF
start=$(date +%s)
backend=$(python3 -c "import json; print(json.load(open('../target.json'))['backend'])")
if [ "$backend" = Vivado ]; then
  command -v vivado_hls > /dev/null || { echo "Vivado backend project: vivado_hls not on PATH"; exit 1; }
  vivado_hls -f build_prj.tcl > ../synth.log 2>&1    # options from build_opt.tcl above
elif command -v vitis-run > /dev/null; then
  vitis-run --tcl build_prj.tcl --mode hls > ../synth.log 2>&1
elif command -v vitis_hls > /dev/null; then
  vitis_hls -f build_prj.tcl > ../synth.log 2>&1
else
  echo "no vitis-run or vitis_hls on PATH"; exit 1
fi
echo "csynth wall $(( $(date +%s) - start )) s" >> ../synth.log
cd "$HERE"
python3 collect_csynth.py project synth_results.csv
cat synth_results.csv
