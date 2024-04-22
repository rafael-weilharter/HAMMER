#!/usr/bin/env bash
MVS_TRAINING="/media/rafweilharter/hard_disk/data/datasets/blendedMVS_high_res"
LIST_TRAIN="/media/rafweilharter/hard_disk/data/datasets/blendedMVS_high_res/training_list.txt"
LIST_TEST="/media/rafweilharter/hard_disk/data/datasets/blendedMVS_high_res/validation_list.txt"

#checkpoint file to finetune
CKPT_FILE="./checkpoints/hammer_weights_blended.ckpt"

python train.py --dataset=blended_mvs_rpm --augment_data --lrepochs="4,6,8,10,12:2" --lr=0.0001 --epochs=20 --neighbors=5 --summary_freq=20 --batch_size=1 --input_scale=1.0 --output_scale=2 --interval_scale=1.0 --trainpath=$MVS_TRAINING --trainlist=$LIST_TRAIN --testlist=$LIST_TEST --loadckpt=$CKPT_FILE --logdir ./checkpoints/training_finetune_blended $@
