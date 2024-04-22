#!/usr/bin/env bash
CKPT_FILE="./checkpoints/hammer_weights_blended.ckpt"

MVS_DATA="/media/rafweilharter/hard_disk/data/tanks_and_temples/yaoyao/tankandtemples/intermediate"
LIST_TEST="/media/rafweilharter/hard_disk/data/tanks_and_temples/yaoyao/tankandtemples/intermediate/lists_test.txt"
OUT_DIR="/media/rafweilharter/hard_disk/data/tanks_and_temples/yaoyao/tankandtemples/intermediate/outputs"

python test.py --dataset=tanks_and_temples --outdir=$OUT_DIR --ent_high=0.7 --ent_low=0 --cv_mask=9 --interval_scale=1.0 --input_scale=1.0 --output_scale=2 --consistent=2 --dist=0.2 --rel_dist=1000 --neighbors=5 --batch_size=1 --numdepth=384 --testpath=$MVS_DATA --testlist=$LIST_TEST --loadckpt $CKPT_FILE $@
