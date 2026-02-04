from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from ._base import BaseGPTQForCausalLM
from ..utils.modeling_utils import recurse_setattr


class ExpandedParallelExperts(nn.Module):
    """
    Granite MoE의 ParallelExperts(3D weight)를 expert별 nn.Linear(ModuleList)로 펼친 구현.

    원본 GraniteMoeParallelExperts는 다음 형태의 weight를 가집니다:
      weight: (num_experts, out_features, in_features)

    AutoGPTQ의 기본 파이프라인(find_layers/make_quant/pack)이 nn.Linear 단위로 동작하므로,
    expert별 nn.Linear로 구조를 바꿔서 FFN 전체를 GPTQ 대상에 포함시킵니다.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim != 3:
            raise ValueError(f"Expected 3D weight (E, O, I), got shape={tuple(weight.shape)}")

        num_experts, out_features, in_features = weight.shape
        self.num_experts = int(num_experts)
        self.input_size = int(in_features)
        self.output_size = int(out_features)

        experts: list[nn.Linear] = []
        # 각 expert를 독립 Linear로 구성 (bias 없음)
        for i in range(self.num_experts):
            lin = nn.Linear(self.input_size, self.output_size, bias=False)
            lin.weight = nn.Parameter(weight[i].detach().clone())
            experts.append(lin)
        self.experts = nn.ModuleList(experts)

    def forward(self, inputs: torch.Tensor, expert_size: list[int]):
        # inputs: (sum(expert_size), in_features)
        # expert_size: 길이=num_experts, 각 expert에 할당된 토큰 수
        if inputs.numel() == 0:
            return inputs.new_empty((0, self.output_size))

        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i, lin in enumerate(self.experts):
            x = input_list[i]
            if x.numel() == 0:
                output_list.append(x.new_empty((0, self.output_size)))
            else:
                output_list.append(lin(x))
        return torch.cat(output_list, dim=0)


def _expand_granite_parallel_experts_inplace(model: nn.Module) -> None:
    """
    Granite 모델 내 ParallelExperts 모듈을 ExpandedParallelExperts로 교체합니다.

    대상:
      - model.layers.*.block_sparse_moe.input_linear
      - model.layers.*.block_sparse_moe.output_linear

    제외:
      - block_sparse_moe.router.layer (요청대로 router는 양자화 제외)
    """
    targets_suffix = (
        "block_sparse_moe.input_linear",
        "block_sparse_moe.output_linear",
    )

    # named_modules()는 루트 포함, child 포함 모두 제공
    for name, module in list(model.named_modules()):
        if not name.endswith(targets_suffix):
            continue

        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.ndim != 3:
            continue

        # 교체: weight만 넘기고 나머지는 펼친 Linear들이 담당
        expanded = ExpandedParallelExperts(w.to(dtype=w.dtype, device=w.device))
        recurse_setattr(model, name, expanded)


class GraniteExpandedAutoModelForCausalLM:
    """
    BaseGPTQForCausalLM이 model_class로 사용하는 wrapper.
    실제 모델은 AutoModelForCausalLM로 만들고, 생성 직후 Granite ParallelExperts를 펼칩니다.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        _expand_granite_parallel_experts_inplace(model)
        return model

    @classmethod
    def from_config(cls, config, *args: Any, **kwargs: Any):
        model = AutoModelForCausalLM.from_config(config, *args, **kwargs)
        _expand_granite_parallel_experts_inplace(model)
        return model


class GraniteGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "GraniteMoeDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    model_class = GraniteExpandedAutoModelForCausalLM
    
    @classmethod
    def _get_inside_layer_modules(cls, num_experts: int):
        """
        Granite MoE 모델의 레이어 구조에 맞게 inside_layer_modules를 동적으로 생성합니다.
        
        Granite 모델은 ParallelExperts를 사용하므로, 이를 expert별 nn.Linear로 펼친 후
        block_sparse_moe.{input_linear,output_linear}.experts.{i} 들을 모두 양자화합니다.

        Router(router.layer)는 요청대로 제외합니다.
        """
        input_experts = [f"block_sparse_moe.input_linear.experts.{i}" for i in range(num_experts)]
        output_experts = [f"block_sparse_moe.output_linear.experts.{i}" for i in range(num_experts)]
        return [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            input_experts,
            output_experts,
        ]
    
    @classmethod
    def _configure_inside_layers(cls, model_name_or_path: str, trust_remote_code: bool = False) -> None:
        """
        모델 설정을 확인하여 expert 수를 결정하고 inside_layer_modules를 설정합니다.
        """
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        # Granite MoE 모델의 expert 수 확인
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 8))
        cls.inside_layer_modules = cls._get_inside_layer_modules(num_experts)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, trust_remote_code=False, **kwargs):
        """
        모델을 로드하기 전에 inside_layer_modules를 설정합니다.
        """
        cls._configure_inside_layers(pretrained_model_name_or_path, trust_remote_code)
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    
    @classmethod
    def from_quantized(
        cls,
        model_name_or_path,
        *args,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Quantization된 모델을 로드합니다.
        """
        cls._configure_inside_layers(model_name_or_path, trust_remote_code)
        return super().from_quantized(
            model_name_or_path,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


__all__ = ["GraniteGPTQForCausalLM"]

