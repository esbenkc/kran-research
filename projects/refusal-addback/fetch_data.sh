#!/bin/sh
# Download the public raw data into data/raw/ (not committed). Run from projects/refusal-addback/.
set -e
mkdir -p data/raw
for f in harmful_train harmful_val harmful_test harmless_train harmless_val harmless_test; do
  curl -sfL -o data/raw/$f.json https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/$f.json
done
curl -sfL -o data/raw/lat.parquet https://huggingface.co/api/datasets/LLM-LAT/harmful-dataset/parquet/default/train/0.parquet
