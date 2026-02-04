import os
import sys
from pathlib import Path
import random
from typing import Iterable, Sequence

# Ensure we use the local AutoGPTQ codebase (with granite support),
# not an older `auto-gptq` installed in site-packages.
_autogptq_root = Path(__file__).resolve().parents[1]
if str(_autogptq_root) not in sys.path:
    sys.path.insert(0, str(_autogptq_root))

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
DEFAULT_BITS: Sequence[int] = (4,)
NUM_CALIBRATION_SAMPLES = 128
CALIBRATION_MAX_LENGTH = 512
CALIBRATION_DATASET_NAME = "wikitext"
CALIBRATION_DATASET_CONFIG = "wikitext-2-raw-v1"
CALIBRATION_DATASET_SPLIT = "train"
# Cholesky 안정성을 위해 damp_percent를 올려서(대각 jitter) 수치적으로 PD가 되도록 유도합니다.
# 기본값(0.01)에서 종종 실패하므로 0.05~0.2 범위를 권장합니다.
DAMP_PERCENT = 0.10
CALIBRATION_PROMPTS: Sequence[str] = (
    "요즘 서울 날씨 어때?",
    "Explain the difference between mixture-of-experts and dense transformers.",
    "사용자가 프롬프트로 원하는 것을 정확히 이해하려면 어떻게 해야 할까?",
    "Compare LLaMA, Mistral, and Granite in terms of training data and architecture.",
    "다음 문장을 이어서 자연스럽게 작성해줘: 인공지능 모델을 양자화하면",
    "Provide a short summary of the benefits of GPTQ quantization for inference deployment.",
    "What is KOREA?",
    "Explain what is semiconductor.",
)


def _format_as_chat_if_possible(tokenizer: AutoTokenizer, user_text: str) -> str:
    """
    Granite 계열은 chat_template이 있는 경우가 많아서, 가능하면 chat 형식으로 포맷합니다.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # 템플릿이 없거나 예외가 나면 raw text로 fallback
            return user_text
    return user_text


def _load_extra_calibration_texts(target_count: int, seed: int = 0) -> list[str]:
    """
    데이터셋에서 추가 calibration 텍스트를 가져옵니다.
    실패하면 빈 리스트를 반환.

    - 가능한 한 wikitext에서만 채우도록 시도합니다.
    - 텍스트가 부족하면 필터를 점진적으로 완화하고,
      그래도 부족하면 샘플링을 반복(중복 허용)하여 target_count를 채웁니다.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []

    try:
        ds = load_dataset(
            CALIBRATION_DATASET_NAME,
            CALIBRATION_DATASET_CONFIG,
            split=CALIBRATION_DATASET_SPLIT,
        )

        # wikitext는 공백 라인이 많아서 필터링
        raw_texts = [t.strip() for t in ds["text"] if isinstance(t, str) and t.strip()]

        rng = random.Random(seed)

        # 너무 빡센 필터로 샘플이 부족해지는 경우가 있어서 점진적으로 완화
        for min_len in (80, 40, 20, 1):
            texts = [t for t in raw_texts if len(t) >= min_len]
            if texts:
                rng.shuffle(texts)
                if len(texts) >= target_count:
                    return texts[:target_count]

        # 여기까지 왔다면 raw_texts가 매우 적다는 뜻 → 중복 허용해서 채움
        if not raw_texts:
            return []
        rng.shuffle(raw_texts)
        out: list[str] = []
        i = 0
        while len(out) < target_count:
            out.append(raw_texts[i % len(raw_texts)])
            i += 1
        return out[:target_count]
    except Exception:
        return []


def _prepare_calibration_examples(
    tokenizer: AutoTokenizer,
    prompts: Iterable[str],
    max_length: int = CALIBRATION_MAX_LENGTH,
) -> list[dict[str, torch.Tensor]]:
    examples: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        prompt = _format_as_chat_if_possible(tokenizer, prompt)
        encoded = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        examples.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return examples


def quantize_granite_variants(
    bits_list: Sequence[int] = DEFAULT_BITS,
    output_root: os.PathLike | str = "granite_quantized",
    group_size: int = 32,
    desc_act: bool = False,
    true_sequential: bool = True,
    static_groups: bool = False,
    sym: bool = True,
    calibration_prompts: Sequence[str] = CALIBRATION_PROMPTS,
    num_calibration_samples: int = NUM_CALIBRATION_SAMPLES,
) -> None:
    """
    ibm-granite/granite-3.1-1b-a400m-instruct 모델을 지정된 비트 목록으로 GPTQ 양자화하여 저장합니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)

    # 1) 기본 프롬프트 + 2) 데이터셋 텍스트로 calibration 샘플 수를 늘림
    prompts: list[str] = list(calibration_prompts)
    if len(prompts) < num_calibration_samples:
        extra = _load_extra_calibration_texts(
            num_calibration_samples - len(prompts),
            seed=0,
        )
        prompts.extend(extra)

    # 그래도 부족하면 wikitext 텍스트를 중복 허용으로 더 채움
    if len(prompts) < num_calibration_samples:
        missing = num_calibration_samples - len(prompts)
        extra = _load_extra_calibration_texts(missing, seed=1)
        prompts.extend(extra)

    prompts = prompts[:num_calibration_samples]
    print(
        f"[INFO] Calibration samples: {len(prompts)} (max_length={CALIBRATION_MAX_LENGTH}, dataset={CALIBRATION_DATASET_NAME}/{CALIBRATION_DATASET_CONFIG}:{CALIBRATION_DATASET_SPLIT})"
    )
    calibration_examples = _prepare_calibration_examples(tokenizer, prompts)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    for bits in bits_list:
        save_dir = output_root_path / f"granite-3.1-1b-a400m-instruct-gptq-{bits}bit"
        if save_dir.exists():
            print(f"[SKIP] {bits}-bit 결과가 이미 존재합니다: {save_dir}")
            continue

        print(f"[INFO] {bits}-bit 양자화를 시작합니다...")
        quant_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            true_sequential=true_sequential,
            static_groups=static_groups,
            sym=sym,
            damp_percent=DAMP_PERCENT,
            model_name_or_path=MODEL_ID,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            MODEL_ID,
            quantize_config=quant_config,
            trust_remote_code=True,
        )
        try:
            model.quantize(calibration_examples)
            model.save_quantized(save_dir, use_safetensors=True)
            tokenizer.save_pretrained(save_dir)
            print(f"[DONE] {bits}-bit 모델 저장 완료: {save_dir}")
        finally:
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    quantize_granite_variants()

