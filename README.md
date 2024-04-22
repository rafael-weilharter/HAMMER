# HAMMER

## About

We present HAMMER, a Hierarchical And Memory-efficient MVSNet with Entropy-filtered Reconstructions.
While the majority of recent Multi-View Stereo Networks estimates a depth map per reference image, the performance is then only evaluated on the fused 3D model obtained from all images.
This approach makes a lot of sense since ultimately the point cloud is the result we are mostly interested in.
On the flip side, it often leads to a burdensome manual search for the right fusion parameters in order to score well on the public benchmarks.
We propose to learn a filtering mask based on entropy, which, in combination with a simple two-view geometric verification, is sufficient to generate high quality 3D models of any input scene. Distinct from existing works, a tedious manual parameter search for the fusion step is not required.
Furthermore, we take several precautions to keep the memory requirements for our method very low in the training as well as in the inference phase.
Our method only requires 6 GB of GPU memory during training, while 3.6 GB are enough to process 1920x1024 images during inference.
Experiments show that HAMMER ranks amongst the top published methods on the DTU and Tanks and Temples benchmarks in the official metrics, especially when keeping the fusion parameters fixed.

<img src="images/network.png">

If you find this project useful for your research, please cite:
```
@inproceedings{weilharter2024hammer,
  title={HAMMER: Learning Entropy Maps To Create Accurate 3D Models in Multi-View Stereo},
  author={Weilharter, Rafael and Fraundorfer, Friedrich},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3466--3475},
  year={2024}
}
```

## How To Use

### Requirements

* Nvidia GPU with 11GB or more VRAM
* CUDA 10.1+
* python 3.7+
* pytorch 1.10+
* opencv 3.4.2+

### Datasets
Pre-processed datasets can be downloaded on the github page of [MVSNet](https://github.com/YoYo000/MVSNet) and [BlendedMVS](https://github.com/YoYo000/BlendedMVS).
Our repository provides 3 dataloaders for [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), [BlendedMVS](https://github.com/YoYo000/BlendedMVS) and [Tanks and Temples (TaT)](https://www.tanksandtemples.org/), respectively.

### Training
Run the command `python train.py -h` to get information about the usage. An example can be found in `train_dtu.sh` or `train_blended.sh` (set the correct paths to the training data).

### Testing
Run the command `python test.py -h` to get information about the usage. An example can be found in `test_dtu.sh` or `test_tat.sh` (set the correct paths to the testing data).

You can use your own trained weights or use the weights provided in `checkpoints/hammer_weights_blended.ckpt`. These are the weights obtained by training on the DTU dataset and then finetuning on BlendedMVS.
The provided weights differ slightly from the paper by using the attention layers of our previous ATLAS-MVSNet publication in the feature extraction:
```
@inproceedings{weilharter2022atlas,
  title={ATLAS-MVSNet: Attention Layers for Feature Extraction and Cost Volume Regularization in Multi-View Stereo},
  author={Weilharter, Rafael and Fraundorfer, Friedrich},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  pages={3557--3563},
  year={2022},
  organization={IEEE}
}
```
This results in a better qualitative model, but will perform slightly worse on the quantitative benchmarks.

## Performance

### DTU
| Acc. (mm) | Comp. (mm) | Overall (mm) |
|-----------|------------|--------------|
| 0.326     | 0.270      | 0.298        |

### TaT intermediate (F-score)
| Mean  | Family | Francis | Horse | LH    | M60   | Panther | PG    | Train |
|-------|--------|---------|-------|-------|-------|---------|-------|-------|
| 61.70 | 78.45  | 59.25   | 54.33 | 62.80 | 63.20 | 59.57   | 61.72 | 54.23 |

### TaT advanced (F-score)
| Mean  | Auditorium | Ballroom | Courtroom | Museum | Palace | Temple |
|-------|------------|----------|-----------|--------|--------|--------|
| 36.13 | 24.17      | 40.07    | 38.14     | 49.56  | 31.54  | 33.31  |


## Acknowledgement

Parts of the code are adapted from [HSM](https://github.com/gengshan-y/high-res-stereo), [MVSNet](https://github.com/YoYo000/MVSNet) and [Stand-Alone-Self-Attention](https://github.com/leaderj1001/Stand-Alone-Self-Attention).
