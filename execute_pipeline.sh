#!/bin/bash

python3 scripts/pipeline.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --reference data/reference/FSL_HCP1065_FA_1mm.nii.gz \
    --log_level INFO \
    --max_workers 4 \
    --coco_input data/raw/_annotations.coco.json