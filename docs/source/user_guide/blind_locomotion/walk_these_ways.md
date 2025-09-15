# ⛕ Walk These Ways

> Code in this section corresponds to `go2_wtw` in genesis_lr.

As stated by Margolis $ \textit{et al.}^{1}$, the multiplicity of behavior (MoB) can help the robot generalize in different ways. The basic idea of $ \textit{Walk These Ways} $ is to learn different behaviors on the flat ground and tune the behavior through high-level decision (which is human operator in this paper).

To incoporate different behaviors into one NN, we essentially want to achieve multi-task reinforcement learning. Key components in this implementation consist of three parts: 
1. Task-related observation
2. Task rewards
3. Reward-based Curriculum

## Task-related Observation

To enable the neural network policy to distinguish between different behaviors, we need to provide observation related to the task as an input. 

## References

1. [Walk These Ways](https://gmargo11.github.io/walk-these-ways/)