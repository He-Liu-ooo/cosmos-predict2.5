# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Any

import attrs
import torch
from megatron.core import parallel_state
from torch import Tensor
from cosmos_predict2._src.imaginaire.utils import log
from dataclasses import fields

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.text2world_model import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldModelRectifiedFlowConfig(Text2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    conditional_frame_timestep: float = (
        -1.0
    )  # Noise level used for conditional frames; default is -1 which will not take effective
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]


class ActionVideo2WorldModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return raw_state, latent_state, condition

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
        use_cuda_graph: bool = False,
    ) -> DenoisePrediction:
        """
        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            velocity prediction
        """
        
        log.debug(f'Denoise called: condition.is_video={condition.is_video}, condition.use_video_condition={condition.use_video_condition}, self.config.denoise_replace_gt_frames={self.config.denoise_replace_gt_frames}')
        
        
        if condition.is_video:
            # set condition.gt_frames same dtype/device as xt_B_C_T_H_W and set the value to condition_state_in_B_C_T_H_W
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            # condition.use_video_condition = True
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            log.debug(f"condition_state_in_B_C_T_H_W(gt).shape={condition_state_in_B_C_T_H_W.shape}, xt_B_C_T_H_W(noise).shape={xt_B_C_T_H_W.shape}, condition_video_mask.shape={condition_video_mask.shape}")
            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

        # Forward pass through the network. If requested, attempt a guarded
        # CUDA Graph capture/replay to accelerate repeated inference. The
        # capture is performed once (warmup) and replayed for subsequent
        # calls. Any exception during capture/replay falls back to a normal
        # forward to remain robust.
        if use_cuda_graph and torch.cuda.is_available():
            # Prepare inputs on correct device/dtype
            inputs: dict = {}
            inputs["x_B_C_T_H_W"] = xt_B_C_T_H_W.to(**self.tensor_kwargs)
            inputs["timesteps_B_T"] = timesteps_B_T.to(device=self.tensor_kwargs["device"])

            cond_dict = condition.to_dict()
            # Separate tensor and non-tensor condition entries
            non_tensor_inputs: dict = {}
            for k, v in cond_dict.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(**self.tensor_kwargs)
                else:
                    non_tensor_inputs[k] = v

            # Diagnostic: log non-tensor condition keys/types before capture
            try:
                if non_tensor_inputs:
                    log.warning(f"Non-tensor condition entries present before CUDAGraph capture: {list(non_tensor_inputs.keys())}")
            except Exception:
                pass

            cg_state = getattr(self, "_cuda_graph_state_action", None)
            try:
                if not cg_state or not cg_state.get("captured", False):
                    # Warmup forward to allocate any lazy buffers and determine shapes
                    warm_out = self.net(**{**inputs, **non_tensor_inputs}).float()

                    # Preallocate static buffers matching warmup shapes on CUDA
                    static_inputs: dict = {}
                    static_inputs["x_B_C_T_H_W"] = torch.empty_like(inputs["x_B_C_T_H_W"], device=self.tensor_kwargs["device"], dtype=inputs["x_B_C_T_H_W"].dtype)
                    static_inputs["timesteps_B_T"] = torch.empty_like(inputs["timesteps_B_T"], device=self.tensor_kwargs["device"], dtype=inputs["timesteps_B_T"].dtype)
                    for k, v in inputs.items():
                        if k in ("x_B_C_T_H_W", "timesteps_B_T"):
                            continue
                        static_inputs[k] = torch.empty_like(v, device=self.tensor_kwargs["device"], dtype=v.dtype)

                    static_out = torch.empty_like(warm_out, device=self.tensor_kwargs["device"], dtype=warm_out.dtype)

                    # Copy initial values into static buffers
                    static_inputs["x_B_C_T_H_W"].copy_(inputs["x_B_C_T_H_W"])
                    static_inputs["timesteps_B_T"].copy_(inputs["timesteps_B_T"])
                    for k in list(static_inputs.keys()):
                        if k in ("x_B_C_T_H_W", "timesteps_B_T"):
                            continue
                        static_inputs[k].copy_(inputs[k])

                    # Capture graph. Synchronize first to ensure there are no
                    # outstanding CUDA ops on the default stream (a common
                    # cause of CUDAGraph capture failures).
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        # Best-effort; continue and rely on the outer try/except
                        pass

                    # Capture into a new CUDAGraph using explicit capture API
                    # (some PyTorch builds do not implement CUDAGraph as a
                    # context manager, causing `__enter__` AttributeError).
                    g = torch.cuda.CUDAGraph()
                    # Use explicit capture_begin()/capture_end() for maximum
                    # compatibility across PyTorch versions.
                    g.capture_begin()
                    out = self.net(**{**static_inputs, **non_tensor_inputs})
                    static_out.copy_(out)
                    g.capture_end()

                    self._cuda_graph_state_action = {
                        "captured": True,
                        "graph": g,
                        "static_inputs": static_inputs,
                        "static_output": static_out,
                        "non_tensor_inputs": non_tensor_inputs,
                    }
                    net_output_B_C_T_H_W = static_out
                else:
                    # Replay: copy the new inputs into static buffers, then replay
                    static_inputs = cg_state["static_inputs"]
                    static_inputs["x_B_C_T_H_W"].copy_(inputs["x_B_C_T_H_W"])
                    static_inputs["timesteps_B_T"].copy_(inputs["timesteps_B_T"])
                    for k in list(static_inputs.keys()):
                        if k in ("x_B_C_T_H_W", "timesteps_B_T"):
                            continue
                        static_inputs[k].copy_(inputs[k])

                    cg_state["graph"].replay()
                    net_output_B_C_T_H_W = cg_state["static_output"]
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                # Include the traceback in the warning so it's visible in default logs
                log.warning(
                    "CUDAGraph capture/replay failed (%s); falling back to normal forward. Traceback:\n%s" % (e, tb)
                )
                net_output_B_C_T_H_W = self.net(**{**inputs, **non_tensor_inputs}).float()
        else:
            net_output_B_C_T_H_W = self.net(
                x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
                timesteps_B_T=timesteps_B_T,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
                **condition.to_dict(),
            ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)
        
                # Helper: log only shapes/types for tensors; avoid printing tensor contents
        def _log_summary(obj_name: str, obj: Any) -> None:
            try:
                log.debug(f"{obj_name} type: {type(obj)}")
                # dict-like
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, torch.Tensor):
                            log.debug(f"{obj_name}[{k}] shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                        else:
                            log.debug(f"{obj_name}[{k}] type={type(v)}")
                    return

                # dataclass-like or objects exposing to_dict
                if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
                    try:
                        d = obj.to_dict(skip_underscore=False)
                    except TypeError:
                        d = obj.to_dict()
                    for k, v in d.items():
                        if isinstance(v, torch.Tensor):
                            log.debug(f"{obj_name}[{k}] shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                        else:
                            log.debug(f"{obj_name}[{k}] type={type(v)}")
                    return

                # dataclass fields (fallback)
                try:
                    for f in fields(obj):
                        v = getattr(obj, f.name)
                        if isinstance(v, torch.Tensor):
                            log.debug(f"{obj_name}[{f.name}] shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                        else:
                            log.debug(f"{obj_name}[{f.name}] type={type(v)}")
                    return
                except Exception:
                    pass

                # generic fallback: list public attrs (no tensor contents)
                for attr in dir(obj):
                    if attr.startswith("_"):
                        continue
                    try:
                        v = getattr(obj, attr)
                    except Exception:
                        continue
                    if callable(v):
                        continue
                    if isinstance(v, torch.Tensor):
                        log.debug(f"{obj_name}.{attr} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                    else:
                        # skip large reprs
                        log.debug(f"{obj_name}.{attr} type={type(v)}")
            except Exception as e:
                log.warning(f"Failed to print summary for {obj_name}: {e}")

        _log_summary("condition", condition)
        _log_summary("uncondition", uncondition)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor, use_cuda_graph: bool) -> torch.Tensor:
            log.debug(f"Execute velocity_fn: noise.shape={noise.shape}, noise_x.shape={noise_x.shape}, timestep={timestep}")
            cond_v = self.denoise(noise, noise_x, timestep, condition, use_cuda_graph)
            log.debug(f"After denoise: cond_v.shape={cond_v.shape}")
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition, use_cuda_graph)
            log.debug(f"After denoise: uncond_v.shape={uncond_v.shape}")
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn
