# RDA4RobustSSL
Code for **[RDA: Reciprocal Distribution Alignment for Robust Semi-supervised Learning](https://arxiv.org/abs/2208.04619)**.

Our paper is accepted by **ECCV2022** 😈. Thanks to users.

![](/figures/poster.jpg)

## Requirements
- matplotlib==3.3.2
- numpy==1.19.2
- pandas==1.1.5
- Pillow==9.0.1
- torch==1.4.0+cu92
- torchvision==0.5.0+cu92
## How to Train
### Important Args
- `--num_labels` : Amount of labeled data used.  
- `--mismatch [rda/darp/darp_reversed]` : Select the type of mismatched distribution dataset. `rda` means our protocol for constructing mismatched distribution dataset, which is described in Sec. 4.2; `DARP` means DARP's protocol described in Sec. C.1 of Supplementary Materials; `DARP_reversed` means DARP's protocol for CIFAR-10 with reversed version of mismatched distribution.
- `--n0` : When `--mismatch rda`, this arg means the imbalanced ratio $N_0$ for labeled data; When `--mismatch DARP/DARP_reversed`, this arg means the imbalanced ratio $\gamma_l$ for labeled data.
- `--gamma` : When `--mismatch rda`, this arg means the imbalanced ratio $\gamma$ for unlabeled data; When `--mismatch DARP/DARP_reversed`, this arg means the imbalanced ratio $\gamma_u$ for unlabeled data. 
- `--net` : By default, wide resnet (WRN-28-2) are used for experiments. If you want to use other backbones for tarining, set `--net [resnet18/preresnet/cnn13]`. We provide alternatives as follows: ResNet-18, PreAct ResNet and CNN-13.
- `--dataset [cifar10/cifar100/stl10/miniimage]` and `--data_dir`  : Your dataset name and path. We support four datasets: CIFAR-10, CIFAR-100, STL-10 and mini-ImageNet. When `--dataset stl10`, set `--fold [0/1/2/3/4].`
- `--num_eval_iter` : After how many iterations, we evaluate the model. Note that although we show the accuracy of pseudo-labels on unlabeled data in the evaluation, this is only to show the training process. We did not use any information about labels for unlabeled data in the training. Additionally, when you train model on STL-10, the pseudo-label accuracy will not be displayed normally, because we don't have ground-truth of unlabeled data.
### Training with Single GPU

```
python train_rda.py --rank 0 --gpu [0/1/...] @@@other args@@@
```
### Training with Multi-GPUs 

```
python train_rda.py --world-size 1 --rank 0 @@@other args@@@
```

## Examples of Running
By default, the model and `dist&index.txt` will be saved in `\saved_models\@--save_name (yours)`. The file `dist&index.txt` will display   detailed settings of mismatched distribution. This code assumes 1 epoch of training, but the number of iterations is 2\*\*20. For CIFAR-100, you need set `--widen_factor 8` for WRN-28-8 whereas WRN-28-2 is used for CIFAR-10.  Note that you need set `--net resnet18` for STL-10 and mini-ImageNet. Additionally, WRN-28-2 is used for all experiments under DARP's protocol. To better reproduce our experimental results, it is recommended to follow our experimental environment using a single GPU for training.

### Conventional Setting 
#### Matched and balanced $C_x$, $C_u$ for Tab. 1 in Sec. 5.1
- CIFAR-10, e.g., with 20 labels, result of seed 1 (Acc/%): 93.40, weight: [here][cifar10-20]

```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 20  --gpu 0
```

***
### Mismatched Distribution
#### Imbalanced $C_x$ and balanced $C_u$ for Tab. 2 in Sec. 5.2
- CIFAR-10, e.g., with 40 labels, $N_0=10$, result of seed 1 (Acc/%): 93.06, weight: [here][cifar10-40-10]
```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch rda --n0 10 --gpu 0
```

***
- CIFAR-100, e.g.,  with 400 labels, $N_0=40$, result of seed 1 (Acc/%): 33.54, weight: [here][cifar100-400-40]
```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar100 --dataset cifar100 --num_classes 100 --num_labels 400 --mismatch rda --n0 40 --gpu 0 --widen_factor 8
```

***
- mini-ImageNet, e.g., with 1000 labels, $N_0=40$, result of seed 1 (Acc/%): 43.59, weight: [here][mini-1000-40]
```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name miniimage --dataset miniimage --num_classes 100 --num_labels 1000 --mismatch rda --n0 40 --gpu 0 --net resnet18 
```


***

#### Imbalanced and mismatched $C_x$, $C_u$ for Tab. 3 in Sec. 5.2
- CIFAR-10, e.g., with 40 labels, $N_0=10$, $\gamma=5$, result of seed 1 (Acc/%): 80.68, weight: [here][cifar10-40-10-5]

```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch rda --n0 10 --gamma 5 --gpu 0
```
***
#### Balanced $C_x$ and imbalanced $C_u$ for Tab. 5 in Sec. 5.2

- CIFAR-10, e.g., with 40 labels, $\gamma=200$, result of seed 1 (Acc/%): 45.57, weight: [here][cifar10-40-1-200]
```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch rda --gamma 200 --gpu 0
```

***
#### DARP's protocol for Tab. 5 in Sec. 5.2.
- CIFAR-10, e.g., with $\gamma_l=100$, $\gamma_u=1$, result of seed 1 (Acc/%): 93.11, weight: [here][cifar10-darp-1]

```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch darp --n0 100 --gamma 1 --gpu 0
```

***
- CIFAR-10 (reversed), e.g., with $\gamma_l=100$, $\gamma_u=100$ (reversed), result of seed 1 (Acc/%): 78.53, weight: [here][cifar10-darp-re]
```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch darp_reversed --n0 100 --gamma 100 --gpu 0
```

***
- For STL-10 in DARP's protocol, set `--fold -1`, e.g., with $\gamma_l=10$, result of seed 1 (Acc/%): 87.21, weight: [here][stl10-darp]
```
python train_rda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name stl10 --dataset stl10 --num_classes 10 --mismatch darp --n0 10 --gpu 0 --fold -1 
```

***
## Resume Training and Evaluation
If you restart the training, please use `--resume --load_path @your_path`. Each time you start training, the evaluation results of the current model will be displayed. If you want to evaluate a model, use its checkpoints to resume training.

## Results (e.g. seed=1)

| Dateset | Labels | $N_0$ / $\gamma_l$ |$\gamma$ / $\gamma_u$|Acc (%)|Note|
| :-----:| :----: | :----: |:----: |:----: |:----: |
|CIFAR-10 | 20 | - |- |93.40 |Conventional setting|
| | 40 | - |- |94.13 ||
| | 80 | - |- |94.24 ||
| | 100 | - |- |94.66 ||
| | 40 | 10 |- |93.06 |Imbalanced $C_x$ and balanced $C_u$|
|  | 40 | 20 |- |81.51 ||
|  | 100 | 40 |- |94.42 ||
|  | 100 | 80 |- |78..99 ||
|  | 40 | 10 |2 |81.60 |Mismatched imbalanced $C_x$ and $C_u$|
|  | 40 | 10 |5 |80.68 ||
|  | 100 | 40 |5 |79.54 ||
|  | 40 | - |100 |47.68 |Balanced $C_x$ and imbalanced $C_u$|
|  | 40 | - |200 |45.57 ||
|  | DARP | 100 |1 |93.11 |DARP's protocol|
|  | DARP | 100 |50 |79.84 ||
|  | DARP | 100 |150 |74.71 ||
|  | DARP (reversed) | 100 |100 |78.53 ||
|CIFAR-100  | 400 | 40 |- |33.54 |Imbalanced $C_x$ and balanced $C_u$|
|  | 1000 | 80 |- |42.87 ||
|STL-10 | 1000| - |- |82.53 |Conventional setting|
| | DARP| 10 |- |87.21 |DARP's protocol|
| | DARP| 20 |- |83.71 ||
|mini-ImageNet | 1000| -|- |47.73 |Conventional setting|
| | 1000| 40 |- |43.59 |Imbalanced $C_x$ and balanced $C_u$|
| | 1000| 80 |- |38.16 ||
| | 1000| 40 |10 |25.91 |Mismatched imbalanced $C_x$ and $C_u$|



## Acknowledgement
Our code is based on open source code: [LeeDoYup/FixMatch-pytorch][1].

[1]: https://github.com/LeeDoYup/FixMatch-pytorch
[cifar10-20]: https://1drv.ms/u/s!Ao848hI985sshjSqXrH4QoG1JgCH?e=qiGZe3
[cifar10-40-10]: https://1drv.ms/u/s!Ao848hI985sshiRA2Wm2F0IuG_hv?e=jh2sOg
[cifar10-40-10-5]: https://1drv.ms/u/s!Ao848hI985sshiZTF8hAq51b01n1?e=OTPuMd
[cifar10-40-1-200]: https://1drv.ms/u/s!Ao848hI985sshij5UqKI2EkVHMOH?e=g1Nq6V
[cifar10-darp-1]: https://1drv.ms/u/s!Ao848hI985sshiqUhVPxBklQnKM0?e=vCs0HU
[cifar10-darp-re]: https://1drv.ms/u/s!Ao848hI985sshizWFgWZj6JwYgZ6?e=dIPbnN
[stl10-darp]: https://1drv.ms/u/s!Ao848hI985sshjXJHqsWy5r68Y9V?e=J4zoOd
[mini-1000-40]: https://1drv.ms/u/s!Ao848hI985sshjBEV0ckYmz7vR_T?e=325Lz8
[cifar100-400-40]: https://1drv.ms/u/s!Ao848hI985sshjIKeeut6rM_NuVW?e=9c9JiB
[cifar10-20]: https://1drv.ms/u/s!Ao848hI985sshjSqXrH4QoG1JgCH?e=qiGZe3
