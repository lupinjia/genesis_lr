# 🧑‍🏫🧑‍🏫 Teacher-Student Framework

In [Walk These Ways](walk_these_ways.md), we know that through tuning robot behaviors, we can let a policy trained on flat ground traverse some difficult terrains (like small curbs and stairs). It indicates generalization ability of the policy since it has been only trained on flat ground and data concerning complex terrains are not included in the training dataet. However, to achieve better stability and traversability on complex terrains, training on terrains are inavoidable. But as introduced in [deploy_to_real_robot](deploy_to_real_robot.md), just a simple network that gets robot state feedback of one time step is not enough. Actually, if you use RNN or stack observation of multiple time steps, the policy will gain better ability to adapt to terrains. However, we still need some mechanism to effectively extract key information from observation history and endow the policy with better understanding of the robot and the surrounding environment.

## Teacher-Student Framework

As one of the pioneer of RL-based control on legged robots, [RSL from ETH zurich](https://rsl.ethz.ch/) has proposed a series of works to push the limit of quadruped robots on complex terrains. The one introduced here is teacher-student framework.




## References

1. [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251)