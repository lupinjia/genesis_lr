# 🦿 Legged Robotics in Genesis

A [legged_gym](https://github.com/leggedrobotics/legged_gym) based framework for training legged robots in [genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)

## Table of Contents

- [🦿 Legged Robotics in Genesis](#-legged-robotics-in-genesis)
  - [Table of Contents](#table-of-contents)
  - [🌟 Features](#-features)
  - [🧪 Test Results](#-test-results)
  - [🛠 Installation](#-installation)
  - [👋 Usage](#-usage)
    - [🚀 Quick Start](#-quick-start)
    - [📖 Instructions](#-instructions)
    - [Deployment](#deployment)
  - [🖼️ Gallery](#️-gallery)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [TODO](#todo)

---

## 🌟 Features

- **Totally based on [legged_gym](https://github.com/leggedrobotics/legged_gym)**

- **Faster and Smaller**
  
  For a go2 walking on the plane task with 4096 envs, the training speed in Genesis is approximately **1.3x** compared to [Isaac Gym](https://developer.nvidia.com/isaac-gym), while the graphics memory usage is roughly **1/2** compared to IsaacGym.

  With this smaller memory usage, it's possible to **run more parallel environments**, which can further improve the training speed.

- Incorporation of various methods in published RL papers
  
  | Method | Paper Link | Location | Materials |
  |--------|------------|----------|-----------|
  | Periodic Gait Reward | [Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387) | [go2_wtw](https://github.com/lupinjia/genesis_lr/blob/main/legged_gym/envs/go2/go2_wtw/go2_wtw.py#L322) | [Comparison between smooth gait function and step gait function](https://github.com/lupinjia/genesis_lr/blob/main/test_resources/gait_reward_type/gait_reward_type.md) |
  | System Identification | [Learning Agile Bipedal Motions on a Quadrupedal Robot](https://arxiv.org/abs/2311.05818) | [go2_sysid](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_sysid) | |
  | Teacher-Student | [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/) | [go2_ts](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ts) | |


## 🧪 Test Results

For tests conducted on Genesis, please refer to [tests.md](./test_resources/tests.md)

## 🛠 Installation

1. Create a new python virtual env with python>=3.10
2. Install [PyTorch](https://pytorch.org/)
3. Install Genesis following the instructions in the [Genesis repo](https://github.com/Genesis-Embodied-AI/Genesis)
4. Install genesis_lr
   ```bash
   git clone git@github.com:lupinjia/genesis_lr.git
   cd genesis_lr
   pip install -e .
   ```

## 👋 Usage

### 🚀 Quick Start

By default, the task is set to `go2`(in `utils/helpers.py`), we can run a training session with the following command:

```bash
cd legged_gym/scripts
python train.py --headless # run training without rendering
```

After the training is done, paste the `run_name` under `logs/go2` to `load_run` in `go2_config.py`: 

![](./test_resources/paste_load_run.png)

Then, run `play.py` to visualize the trained model:

![](./test_resources/go2_flat_play.gif)

### 📖 Instructions

For more detailed instructions, please refer to the [wiki page](https://github.com/lupinjia/genesis_lr/wiki)

### Deployment

For deployment code, please refer to [go2_deploy](https://github.com/lupinjia/go2_deploy)

## 🖼️ Gallery

| Robot | Sim | Real |
|--- | --- | --- |
| Unitree Go2 (Walk These Ways) | [video](https://www.bilibili.com/video/BV1FPedzZEdi/) | [video](https://www.bilibili.com/video/BV1FPedzZEdi/) |
| Bipedal Walker | ![](./test_resources/bipedal_walker_flat.gif) | |

## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)

## TODO

- [x] Add go2 deploy demos and instructions (2025/07/25)
- [ ] Add explicit estimator support
- [x] Add teacher-student implementation (2025/08/02)