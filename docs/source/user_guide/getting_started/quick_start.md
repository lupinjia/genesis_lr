# 🚀Quick Start

By default, we switch between IsaacGym and Genesis Simulator through identifying python version:

```python
# In legged_gym/__init__.py
if sys.version_info[1] >= 10: # >=3.10 for genesis
    SIMULATOR = "genesis"
elif sys.version_info[1] <= 8 and sys.version_info[1] >= 6: # >=3.6 and <3.9 for isaacgym
    SIMULATOR = "isaacgym"
if SIMULATOR == "genesis":
    import genesis as gs
elif SIMULATOR == "isaacgym":
    import isaacgym
```

You can define customized strategy for switching between IsaacGym and Genesis.

## Train a go2 policy on the plane

Under the directory of `legged_gym/envs`, we can see multiple folders for different robots. To train a locomotion for go2 robot on the flat ground, we can refer to the `go2` environment in `legged_gym/envs/go2/go2.py` which inherits the `LeggedRobot` class from `legged_gym/envs/base/legged_robot.py`.

Run this command and you will see log information pumping in the terminal, showing reward values and some training data:

```bash
cd legged_gym/scipts
python train.py --task=go2 --headless
```

```{figure} ../../_static/images/log_info_in_terminal.png
```

## Play the trained go2 policy in Genesis

After the training is over (1000 iterations), you will see a new folder named by the date and time when the training begins (`date_time_`, such as `Sep03_16-30-16_`). Paste that folder name to `go2_config.py`, override the `load_run` attribute in `runner` class of `GO2CfgPPO` class. Then run this command and you will see a simulator scene showing the robots walking on the plane:

```bash
python play.py --task=go2
```

```{figure} ../../_static/images/play_in_genesis.png
```
:::{note}
If you use IsaacGym simulator, the pipeline is the same other than that the simulator window is different.
:::

> If you play the policy trained in genesis to IsaacGym, how will it perform?