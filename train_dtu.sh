#!/usr/bin/env bash
MVS_TRAINING="/media/rafweilharter/hard_disk/data/datasets/dtu"
LIST_TRAIN="/media/rafweilharter/hard_disk/data/datasets/dtu/train.txt"
LIST_TEST="/media/rafweilharter/hard_disk/data/datasets/dtu/test.txt"

python train.py --dataset=dtu_random_crop --batch_size=1 --input_scale=1.0 --output_scale=2 --interval_scale=1.0 --trainpath=$MVS_TRAINING --trainlist=$LIST_TRAIN --testlist=$LIST_TEST --logdir ./checkpoints/new_training_dtu $@
