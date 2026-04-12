#!/bin/bash
# fetch-csvs.sh - downloads and obtains csv files
RAW="/tmp/my_extract"
mkdir -p "$RAW"
wget --quiet -O "$RAW/data.zip" 'https://github.com/joachimvandekerckhove/cogs205b-s26/raw/9dca64e57fd88213f2422c19a8b10953a8fbfdbe/modules/02-version-control/files/data.zip'
unzip -q "$RAW/data.zip" -d "$RAW"

OUT="/workspace/repo/data/$(date +%Y-%m-%d)"
mkdir -p "$OUT"
cp "$RAW"/*.csv "$OUT"

git -C /workspace/repo add "data/$(date +%Y-%m-%d)"
git -C /workspace/repo add "scripts/fetch-csvs.sh"
git -C /workspace/repo commit -m "revised bash script"
git -C /workspace/repo push