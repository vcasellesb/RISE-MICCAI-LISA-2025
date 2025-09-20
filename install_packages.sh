#!/bin/bash
set -e

# SYNTHSR
pip install tensorflow PyWavelets
git clone https://github.com/vcasellesb/synthsr.git SynthSR && pip install -e SynthSR
mkdir -p SynthSR/models
SYNTHSR_MODEL_PATH="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/de0/799/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5"
curl -o SynthSR/models/synthsr_lowfield_v20_230130.h5 $SYNTHSR_MODEL_PATH

# HD BET
git clone https://github.com/vcasellesb/HD-BET.git && pip install -e HD-BET
python3 -c "import HD_BET.checkpoint_download as chk; chk.maybe_download_parameters()"