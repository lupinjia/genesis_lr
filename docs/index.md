# Genesis

[![GitHub Repo stars](https://img.shields.io/github/stars/Genesis-Embodied-AI/Genesis?style=plastic&logo=GitHub&logoSize=auto)](https://github.com/Genesis-Embodied-AI/Genesis)
[![PyPI version](https://badge.fury.io/py/genesis-world.svg?icon=si%3Apython)](https://pypi.org/project/genesis-world/)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fgenesis-embodied-ai.github.io%2F)](https://genesis-embodied-ai.github.io/)
[![Discord](https://img.shields.io/discord/1322086972302430269?logo=discord)](https://discord.gg/nukCuhB47p)
<a href="https://drive.google.com/uc?export=view&id=1ZS9nnbQ-t1IwkzJlENBYqYIIOOZhXuBZ"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" height="20" style="display:inline"></a>


## What is Genesis?

Genesis is a physics platform designed for general purpose *Robotics/Embodied AI/Physical AI* applications. It is simultaneously multiple things:

1. A **universal physics engine** re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
2. A **lightweight**, **ultra-fast**, **pythonic**, and **user-friendly** robotics simulation platform.
3. A powerful and fast **photo-realistic rendering system**.
4. A **generative data engine** that transforms user-prompted natural language description into various modalities of data.

Powered by a universal physics engine re-designed and re-built from the ground up, Genesis integrates various physics solvers and their coupling into a unified framework. This core physics engine is further enhanced by a generative agent framework that operates at an upper level, aiming towards fully **automated data generation** for robotics and beyond.
Currently, we are open-sourcing the underlying physics engine and the simulation platform. The generative framework will be released in the near future.

Genesis is built and will continuously evolve with the following ***long-term missions***:

1. **Lowering the barrier** to using physics simulations and making robotics research accessible to everyone. (See our [commitment](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/mission.html))
2. **Unifying a wide spectrum of state-of-the-art physics solvers** into a single framework, allowing re-creating the whole physical world in a virtual realm with the highest possible physical, visual and sensory fidelity, using the most advanced simulation techniques.
3. **Minimizing human effort** in collecting and generating data for robotics and other domains, letting the data flywheel spin on its own.

## Key Features

Compared to prior simulation platforms, here we highlight several key features of Genesis:

- 🐍 **100% Python**, both front-end interface and back-end physics engine, all natively developed in python.
- 👶 **Effortless installation** and **extremely simple** and **user-friendly** API design.
- 🚀 **Parallelized simulation** with ***unprecedented speed***: Genesis is the **world's fastest physics engine**, delivering simulation speeds up to ***10~80x*** (yes, this is a bit sci-fi) faster than existing *GPU-accelerated* robotic simulators (Isaac Gym/Sim/Lab, Mujoco MJX, etc), ***without any compromise*** on simulation accuracy and fidelity.
- 💥 A **unified** framework that supports various state-of-the-art physics solvers, modeling **a vast range of materials** and physical phenomena.
- 📸 Photo-realistic ray-tracing rendering with optimized performance.
- 📐 **Differentiability**: Genesis is designed to be fully compatible with differentiable simulation. Currently, our MPM solver and Tool Solver are differentiable, and differentiability for other solvers will be added soon (starting with rigid-body simulation).
- ☝🏻 Physically-accurate and differentiable **tactile sensor**.
- 🌌 Native support for ***[Generative Simulation](https://arxiv.org/abs/2305.10455)***, allowing **language-prompted data generation** of various modalities: *interactive scenes*, *task proposals*, *rewards*, *assets*, *character motions*, *policies*, *trajectories*, *camera motions*, *(physically-accurate) videos*, and more.

## Getting Started

### Quick Installation

Genesis is available via PyPI:

```bash
pip install genesis-world
```

You also need to install **PyTorch** following the [official instructions](https://pytorch.org/get-started/locally/).

### Documentation

Please refer to our [documentation site](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html) to for detailed installation steps, tutorials and API references.

## Contributing to Genesis

The goal of the Genesis project is to build a fully transparent, user-friendly ecosystem where contributors from both robotics and computer graphics can **come together to collaboratively create a high-efficiency, realistic (both physically and visually) virtual world for robotics research and beyond**.

We sincerely welcome *any forms of contributions* from the community to make the world a better place for robots. From **pull requests** for new features, **bug reports**, to even tiny **suggestions** that will make Genesis API more intuitive, all are wholeheartedly appreciated!

## Support

- Please use Github [Issues](https://github.com/Genesis-Embodied-AI/Genesis/issues) for bug reports and feature requests.

- Please use GitHub [Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions) for discussing ideas, and asking questions.
