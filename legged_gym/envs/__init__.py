# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
# go2
from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO
# go2_rough
from legged_gym.envs.go2.go2_rough.go2_rough_config import GO2RoughCfg, GO2RoughCfgPPO
# go2_deploy
from legged_gym.envs.go2.go2_deploy.go2_deploy import GO2Deploy
from legged_gym.envs.go2.go2_deploy.go2_deploy_config import GO2DeployCfg, GO2DeployCfgPPO
# go2_gait
from legged_gym.envs.go2.go2_gait.go2_gait import GO2Gait
from legged_gym.envs.go2.go2_gait.go2_gait_config import GO2GaitCfg, GO2GaitCfgPPO
# bipedal_walker
from legged_gym.envs.bipedal_walker.bipedal_walker_config import BipedalWalkerCfg, BipedalWalkerCfgPPO
from legged_gym.envs.bipedal_walker.bipedal_walker import BipedalWalker

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", GO2, GO2Cfg(), GO2CfgPPO())
task_registry.register( "go2_rough", GO2, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "go2_deploy", GO2Deploy, GO2DeployCfg(), GO2DeployCfgPPO())
task_registry.register( "go2_gait", GO2Gait, GO2GaitCfg(), GO2GaitCfgPPO())
task_registry.register( "bipedal_walker", BipedalWalker, BipedalWalkerCfg(), BipedalWalkerCfgPPO())