#!/bin/bash
set -e

pip install tensorflow
git clone https://github.com/vcasellesb/synthsr.git SynthSR && pip install -e SynthSR
git clone https://github.com/vcasellesb/HD-BET.git && pip install -e HD-BET