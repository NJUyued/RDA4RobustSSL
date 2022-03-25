# BDA4RobustSSL
Codes for ECCV Submission (ID: 3058).
## Requirements
- matplotlib==3.3.2
- numpy==1.19.2
- pandas==1.1.5
- Pillow==9.0.1
- torch==1.4.0+cu92
- torchvision==0.5.0+cu92
## Train
### Important Args
- `--num_labels` Amount of labeled data used.  
- `--mismatch` Select the type of mismatched distribution dataset. `bda` means our protocol for constructing mismatched distribution dataset, which is described in Sec. 4.2; `DARP` means DARP's protocol described in Sec. C.1 of Supplementary Materials; `DARP_reversed` means DARP's protocol for CIFAR-10 with reversed version of mismatched distribution.
- `--n0` When `--mismatch bda`, this arg means the imbalanced ratio N_0 for labeled data; When `--mismatch DARP/DARP_reversed`, this arg means the imbalanced ratio gamma_l for labeled data.
- `--gamma` When `--mismatch bda`, this arg means the imbalanced ratio gamma for unlabeled data; When `--mismatch DARP/DARP_reversed`, this arg means the imbalanced ratio gamma_u for unlabeled data. 
- `--net_from_name` and `--net` By default, wide resnet (WRN-28-2) are used for experiments. If you want to use other backbones for tarining, set `--net_from_name True --net @backbone`. We provide alternatives as follows: resnet18, cnn13 and preresnet.
- `--dataset` and `--data` Your dataset name and path. We support four datasets: CIFAR-10, CIFAR-100, STL-10 and mini-ImageNet. When `--dataset stl10`, set `--fold [0/1/2/3/4].`
- `--num_eval_iter` After how many iterations, we evaluate the model. **Note that although we show the accuracy of pseudo-labels on unlabeled data in the evaluation, this is only to show the training process. We did not use any information about labels for unlabeled data in the training. Additionally, when you train model on STL-10, the pseudo-label accuracy will not be displayed normally, because we don't have ground-truth of unlabeled data.**
### Training with Single GPU
All models in this paper are trained on a single GPU.

```
python train_bda.py --rank 0 --gpu [0/1/...] @@@other args@@@
```
### Training with Multi-GPUs (DistributedDataParallel)
We only have one node.

```
python train_bda.py --world-size 1 --rank 0 --multiprocessing-distributed @@@other args@@@
```
### Examples of Running
By default, the model and `dist&index.txt` will be saved in `\saved_models\@--save_name (yours)`. The file `dist&index.txt` will display   detailed settings of mismatched distribution. This code assumes 1 epoch of training, but the number of iterations is 2\*\*20. For CIFAR-100, you need set `--widen_factor 8` for WRN-28-8 whereas WRN-28-2 is used for CIFAR-10.  Note that you need set `--net_from_name True --net resnet18` for STL-10 and mini-ImageNet. Additionally, WRN-28-2 is used for all experiments under DARP's protocol.

Let's mainly take Cifar-10 as examples.
#### Conventional Setting 
- Matched and balanced C_x, C_u for Tab. 1 in Sec. 5.1

```
python train_bda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 20  --gpu 0
```

> With 20labels, result of seed 1 (Acc/%): 92.15
#### Mismatched Distribution
- Imbalanced C_x and balanced C_u for Tab. 2 in Sec. 5.2

```
python train_bda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch bda --n0 10 --gpu 0
```
> With 40 labels, N0=10, result of seed 1 (Acc/%): 93.06
- Imbalanced and mismatched C_x, C_u for Tab. 3 in Sec. 5.2

```
python train_bda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch bda --n0 10 --gamma 5 --gpu 0
```
> With 40 labels, N0=10, gamma=5, result of seed 1 (Acc/%): 80.68
- Balanced C_x and imbalanced C_u for Tab. 5 in Sec. 5.2

```
python train_bda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch bda --gamma 200 --gpu 0
```
> With 40 labels, gamma=200, result of seed 1 (Acc/%): 45.57
- DARP's protocol for Tab. 5 in Sec. 5.2.

For CIFAR-10
```
python train_bda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch DARP --n0 100 --gamma 1 --gpu 0
```
> With gamma_l=100, gamma_u=1, result of seed 1 (Acc/%): 93.11

For CIFAR-10 (reversed)

```
python train_bda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch DARP_reversed --n0 100 --gamma 100 --gpu 0
```
> With gamma_l=100, gamma_u=100 (reversed), result of seed 1 (Acc/%): 78.53

For STL-10, set `--fold -1`
```
python train_bda.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset stl10 --num_classes 10 --mismatch DARP --n0 10 --gpu 0 --fold -1
```
> With gamma_l=10, result of seed 1 (Acc/%): 87.21
## Resume Training and Evaluation
If you restart the training, please use `--resume --load_path @your_path`. Each time you start training, the evaluation results of the current model will be displayed. If you want to evaluate a model, use its checkpoints to resume training.

## Results (seed=1)

| Dateset | Labels | N_0 |gamma|Acc|
| :-----:| :----: | :----: |:----: |:----: |
|CIFAR-10 | 20 | - |- |93.40 |
| | 40 | - |- |94.13 |
| | 80 | - |- |94.24 |
| | 100 | - |- |94.66 |
| | 40 | 10 |- |93.06 |
|  | 40 | 20 |- |81.51 |
|  | 100 | 40 |- |94.42 |
|  | 100 | 80 |- |78..99 |
|  | 40 | 10 |2 |81.60 |
|  | 40 | 10 |5 |80.68 |
|  | 100 | 40 |5 |79.54 |
|  | 40 | - |100 |47.68 |
|  | 40 | - |200 |45.57 |
|  | DARP | 100 |1 |93.11 |
|  | DARP | 100 |50 |79.84 |
|  | DARP | 100 |150 |74.71 |
|  | DARP (reversed) | 100 |100 |78.53 |
|CIFAR-100  | 400 | 40 |- |33.54 |
|  | 1000 | 80 |- |42.87 |
|STL-10 | 1000| - |- |82.53 |
| | DARP| 10 |- |87.21 |
| | DARP| 20 |- |83.71 |
|mini-ImageNet | 1000| -|- |47.73 |
| | 1000| 40 |- |43.59 |
| | 1000| 80 |- |38.16 |
| | 1000| 40 |10 |25.91 |
| | 1000| 40 |10 |25.91 |


## Acknowledgement
Our code is based on open source code: [LeeDoYup/FixMatch-pytorch][1]

[1]: https://github.com/LeeDoYup/FixMatch-pytorch
