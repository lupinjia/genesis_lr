# ⏱️ Explicit Estimator

In [teacher_student framework](teacher_student.md), the latent vector output by the teacher encoder is an encoded information of privileged information $x_t$. The latent vector is some kind of representation of the privileged information, without explicit physical meaning. However, according to our experience in [Deploy to Real Robot](https://genesis-lr.readthedocs.io/en/latest/user_guide/getting_started/deploy_to_real_robot.html), the existence of `base_lin_vel` can improve the performance of the agent significantly. Then the problem is how can we obtain `base_lin_vel` in the real robot. 

From the perspective of model-based control, Kalman Filter can help us estimate `base_lin_vel` using feedback of robot states. However, this kind of method usually relies on some assumption, limiting their versatility. With the great power of neural network, how can we handle it? EstimatorNet$^1$ was proposed to solve this. 

## Framework Analysis

Basically, EstimatorNet has the similar form of diagram with [teacher-student framework](teacher_student.md). The biggest difference is that **the encoder output in EstimatorNet is trained to approximate explicit physical values, such as base linear velocity, foot contact probability, foot height and so on**. The diagram of EstimatorNet is shown as below, where $e_t$ is true value of explicit vector, $o_t^H=[o_t, o_{t-1},...,o_{t-H+1}]$ is observation history with length of H steps, $\hat{e}_t$ is predicted explicit vector from estimator network, $a_t$ is the action, $s_t$ is the state of the environment, $\hat{V}_t$ is the estimated value from the critic.

```{figure} ../../_static/images/estimatornet_diagram.png
```

## Implementation

The implementation of this method is similar with [teacher_student framework](teacher_student.md). Readers can look through files with suffix of `ee` to find the implementation.

## Demonstration




## References
1. [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481)