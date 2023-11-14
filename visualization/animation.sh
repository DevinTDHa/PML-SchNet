#!/usr/bin/env bash

# Requires Python and ffmpeg
#python md17.py

ffmpeg \
    -y \
    -r 30 \
    -pattern_type glob \
    -i 'plots/*.png' \
    -s 1000x1000\
    -crf 18 \
    -preset medium \
    -c:v libx264 \
    -pix_fmt yuv420p \
    atoms.mp4