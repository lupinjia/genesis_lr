# Deploy to Real Robot

In last subsection, we have briefly learned how to start a training session and play the resultant policy in the simulator. The next question is, how to deploy the policy to the real robot, interacting with the dynamics in the real world? 

## Preliminaries

With the knowledge of Reinforcement Learning$^{1,2}$, we know that we want to find an agent $\pi$ that can maximize the discounted accumulative reward. The agent is represented by a neural network which outputs actions based on the observation. In the simplest setup, the observation usually consists of base angular velocities, base orientation, velocity commands, joint angles, joint angular velocities and actions of last timestep. Though you may have seen base linear velocities in `legged_robot.py` (code block below), this information can not be obtained directly through sensors on the robot.

```python
self.obs_buf = torch.cat((self.simulator.base_lin_vel * self.obs_scales.lin_vel,                    # 3
                            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
                            self.simulator.projected_gravity,                                         # 3
                            self.commands[:, :3] * self.commands_scale,                   # 3
                            (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                      * self.obs_scales.dof_pos, # num_dofs
                            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                            self.actions                                                    # num_actions
                            ), dim=-1)
```

Typically, sensors mounted on the robot include Inertial Measurement Unit (IMU) and joint encoders. Through IMU, we can obtain base angular velocities and base orientation (equivalent to projected_gravity above). Joint angles and joint angular velocities can be accessed via joint encoders. 

To obtain base linear velocities, we need to conduct estimation based on other sensors using methods like Kalman Filter or estimator network (which will be explained afterwardsafter). Here we focus on exploiting other available information to make the robot walk successfully.



## References

1. [Hands on RL](https://hrl.boyuai.com/)
2. [Easy RL](https://datawhalechina.github.io/easy-rl/#/)