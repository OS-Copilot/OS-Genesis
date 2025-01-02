# OS-Genesis


<img src="./static/OS-Genesis-Badge.png" alt="overview" style="zoom:80%;" />


[![arXiv](https://img.shields.io/badge/arXiv-2412.19723-b31b1b.svg)](https://arxiv.org/abs/2412.19723) 
![License](https://img.shields.io/badge/License-MIT-blue)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2412.19723)
[![Twitter Follow](https://img.shields.io/twitter/follow/qiushi_sun)](https://twitter.com/qiushi_sun)

This repository contains the code and data for paper [OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](https://arxiv.org/abs/2412.19723).
> We are uploading the data and checkpoints. Due to bandwidth limitations, this will take some time. Stay tuned!


## Overview

We introduce OS-Genesis, an interaction-driven pipeline for synthesizing high-quality and diverse GUI agent trajectory data without human supervision or predefined tasks. By leveraging reverse task synthesis and a trajectory reward model, OS-Genesis enables effective end2end training of GUI agents.

<!-- ![overview](./static/OS-Genesis.png) -->

<img src="./static/OS-Genesis.png" alt="overview" style="zoom:20%;" />


## Training

For details and operations of the training, please refer to the [InternVL2 documentation](https://internvl.readthedocs.io/en/latest/get_started/installation.html) and [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL).

## Evaluation
### AndroidControl
To evaluate the AndroidControl Benchmark, please follow the steps below:

1. **Clone the GitHub Repository:**

   ```
   git clone https://github.com/OS-Copilot/OS-Genesis.git
   ```

2. **Inference:**
   ```
   cd OS-Genesis/evaluation
   bash run_ac_inference.sh
   ```

3. **Evaluation:**
   ```
   pyhton ac_eval.py
   ```

## Mobile

|   Model Name    |                           Base Model                                            |                           Training Data                                            |                           HF Link                           |
| :-------------: | :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :---------------------------------------------------------: |
| OS-Genesis-4B-AC | [InternVL2-4B](https://huggingface.co/OpenGVLab/InternVL2-4B)            | [OS-Genesis-ac-training-data](https://huggingface.co/datasets/OS-Copilot/OS-Genesis-mobile-data/blob/main/os_genesis_ac_training_data.jsonl) | [🤗 link](https://huggingface.co/OS-Copilot/OS-Genesis-4B-AC)  |
| OS-Genesis-7B-AC | [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | [OS-Genesis-ac-training-data](https://huggingface.co/datasets/OS-Copilot/OS-Genesis-mobile-data/blob/main/os_genesis_ac_training_data.jsonl) | [🤗 link](https://huggingface.co/OS-Copilot/OS-Genesis-7B-AC)  |
| OS-Genesis-8B-AC | [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)            | [OS-Genesis-ac-training-data](https://huggingface.co/datasets/OS-Copilot/OS-Genesis-mobile-data/blob/main/os_genesis_ac_training_data.jsonl) | [🤗 link](https://huggingface.co/OS-Copilot/OS-Genesis-8B-AC)  |

## Web

(Coming Soon)

## Citation 📖

🫶 If you are interested in our work or find this repository / our data helpful, please consider using the following citation format when referencing our paper:

```bibtex
@article{sun2024genesis,
  title={OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis},
  author={Sun, Qiushi and Cheng, Kanzhi and Ding, Zichen and Jin, Chuanyang and Wang, Yian and Xu, Fangzhi and Wu, Zhenyu and Jia, Chengyou and Chen, Liheng and Liu, Zhoumianze and others},
  journal={arXiv preprint arXiv:2412.19723},
  year={2024}
}
```