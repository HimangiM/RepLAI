# Self-Supervised Representation Learning from Videos of Audible Interactions

Code release for the paper Self-Supervised Representation Learning from Videos of Audible Interactions. This repo contains the PyTorch implementation and pre-trained models.

Authors: Himangi Mittal, Pedro Morgado, Unnat Jain, Abhinav Gupta

[[Arxiv (Paper + Supplementary)]()]

### Introduction
We propose a self-supervised algorithm to learn representations from egocentric video data. Recently, significant efforts have been made to capture humans interacting with their own environments as they go about their daily activities. In result, several large egocentric datasets of interaction-rich multi-modal data have emerged. However, learning representations from videos can be challenging. First, given the uncurated nature of long-form continuous videos, learning effective representations require focusing on moments in time when interactions take place. Second, visual representations of daily activities should be sensitive to changes in the state of the environment. However, current successful multi-modal learning frameworks encourage representation invariance over time. To address these challenges, we leverage audio signals to identify moments of likely interactions which are conducive to better learning. We also propose a novel self-supervised objective that learns from audible state changes caused by interactions. We validate these contributions extensively on two large-scale egocentric datasets, EPIC-Kitchens-100 and the recently released Ego4D, and show improvements on several downstream tasks, including action recognition, long-term action anticipation, and object state change classification.

For more details, please refer to our [paper]() or [project page]().

![Teaser Image](./fig1.png)

### Citation
If you find our work useful in your research, please cite:
```
```

## Installation
(a). Clone the repository
```
git clone https://github.com/HimangiM/SSRL_N2022.git
```
(b). Install dependencies by setting up conda
```
conda env create -f environment.yml
```

## Self-Supervised Training

### Training on EPIC-KITCHENS-100

Run the following command. Change the logging.name to create new log directories, logs/dataset root path by changing environment.data_dir, and dataset path by changing data.args.base_path.

```
sh commands/command_train_epic_kitchens.sh 
```

### Training on Ego4D

```

```

## Evaluation

### Evaluation on EPIC-KITCHENS-100

Run the following command for evaluation the downstream task of action recognition using a linear classifier. Change the logging.name to load the pre-trained model, logs/dataset root path by changing environment.data_dir, and logging.suffix to create new logs for evaluations.

```
sh commands/command_eval_epic_kitchens.sh 
```


## Pre-trained models
We provide checkpoints for pre-trained models.

### EPIC-KITCHENS-100 

| **Method** | **Top1 Acc (Verb)** | **Top1 Acc (Noun)** | **Top5 Acc (Verb)** | **Top5 Acc (Noun)** | **Model** |
|------- |-----------------|-----------------|-----------------|-----------------|-------|
| RepLAI w/o AStC | 29.29 | 9.67 | 73.33 | 29.54 | [url](https://drive.google.com/file/d/1ls8MFxxaUr_D8KKzqANUfslU90DQYUxU/view?usp=sharing) |
| RepLAI w/o MoI | 28.71 | 8.33 | 73.17 | 27.29 | [url](https://drive.google.com/file/d/1KAtIvgK4RxcgYPK8aHTbnJxawEpY0Q4w/view?usp=sharing) |
| RepLAI (scratch) | 25.75 | 8.12 | 71.25 | 27.29 | [url](https://drive.google.com/file/d/1aVKCLD6DWZYafvrA-rODLyYMw0X_hj9d/view?usp=sharing) |
| RepLAI | 31.71 | 11.25 | 73.54 | 30.54 | [url](https://drive.google.com/file/d/1FlRJxKo0gYGZTzxPUB4hloQWA4cNG4ZU/view?usp=sharing) |

### Ego4D

| **Method** | **StCC: Acc** | **AR: Top1 Acc (Verb)** | **AR: Top1 Acc (Noun)** | **LTA: ED@(Z=20) (Verb)** | **LTA: ED@(Z=20) (Noun)** | **PNR: Err** | **StCC w/ Audio: Acc** | **AR w/ Audio: Top1 Acc (Verb)** | **AR w/ Audio: Top1 Acc (Noun)** | **Model** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RepLAI w/o AStC | 63.60 | 21.1 | 13.5 | 0.774 | 0.853 | 0.795 | 64.40 | 21.40  | 13.00 | [url]() |
| RepLAI w/o MoI | 62.90 | 19.8 | 11.2 | 0.792 | 0.868 | 0.801 | 64.10 | 20.80 | 11.70 | [url]() |
| RepLAI (scratch) | 66.20 | 22.2 | 14.1 | 0.760 | 0.840 | 0.775 | 66.30 | 22.50 | 15.00 | [url]() |
| RepLAI | 66.30 | 22.5 | 14.7 | 0.755 | 0.834 | 0.772 | 66.80 | 23.10 | 15.80 | [url]() |

**StCC**: State Change Classification, **AR**: Action Recognition, **LTA**: Long-Term Anticipation, **PNR**: Point-of-no-return temporal localization
