#!/bin/sh
# Download the public prompt splits into data/raw/ (not committed), then freeze data/splits.json (committed).
set -e
cd "$(dirname "$0")"
mkdir -p data/raw
for f in harmful_train harmful_val harmful_test harmless_train harmless_val harmless_test; do
  curl -sfL -o data/raw/$f.json https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/$f.json
done
python3 prep_data.py
