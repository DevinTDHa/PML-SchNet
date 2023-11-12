#!/usr/bin/env bash

# Requires Python and ffmpeg
python md17.py

ffmpeg \
    -y \
    -r 30 \
    -pattern_type glob \
    -i 'plot/*.png' \
    -s 1000x1000\
    -crf 18 \
    -preset medium \
    -vcodec libx265 \
    atoms.mp4