# 🦿 Legged Robotics in Genesis

A [legged_gym](https://github.com/leggedrobotics/legged_gym) based framework for training legged robots in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main) and [IsaacGym](https://developer.nvidia.com/isaac-gym)

## Table of Contents

- [🦿 Legged Robotics in Genesis](#-legged-robotics-in-genesis)
  - [Table of Contents](#table-of-contents)
  - [🌟 Features](#-features)
  - [🛠 Installation and Usage](#-installation-and-usage)
  - [🖼️ Gallery](#️-gallery)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [TODO](#todo)

---

## 🌟 Features

- **Totally based on [legged_gym](https://github.com/leggedrobotics/legged_gym)**

- **Integration of 2 simulators: Genesis and IsaacGym**
  
  You can train policies in two simulators using one framework and two conda environments.

- **Incorporation of various methods in published RL papers**
  
  | Method | Paper Link | Location |
  |--------|------------|----------|
  | Periodic Gait Reward | [Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387) | [go2_wtw](https://github.com/lupinjia/genesis_lr/blob/main/legged_gym/envs/go2/go2_wtw/go2_wtw.py#L322) |
  | System Identification | [Learning Agile Bipedal Motions on a Quadrupedal Robot](https://arxiv.org/abs/2311.05818) | [go2_sysid](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_sysid) |
  | Teacher-Student | [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/) | [go2_ts](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ts) |
  | EstimatorNet | [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) | [go2_ee](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ee) |
  | Constraints as Terminations | [CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning](https://constraints-as-terminations.github.io/) | [go2_cat](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_cat) |

## 🛠 Installation and Usage

Please refer to the [doc of this repo](https://genesis-lr.readthedocs.io/en/latest/).

## 🖼️ Gallery

| Robot | Sim | Real |
|--- | --- | --- |
| Unitree Go2 | [video](https://www.bilibili.com/video/BV1FPedzZEdi/) | [video](https://www.bilibili.com/video/BV1FPedzZEdi/) |
| Limx TRON1_PF | ![](./resources/images/tron1_pf_demo.gif) | |

## 🙏 Acknowledgements

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- [tron1-rl-isaacgym](https://github.com/limxdynamics/tron1-rl-isaacgym)

## TODO

- [x] Add go2 deploy demos and instructions (2025/07/25)
- [x] Add explicit estimator support (2025/09/29)
- [x] Add teacher-student implementation (2025/08/02)
- [x] Add Constraints as Terminations implementation (2025/10/23)
- [ ] Add support for TRON1_PF (flat and rough)
- [ ] Add support for TRON1_SF (flat and rough)
- [ ] Add support for TRON1_WF (flat and rough)