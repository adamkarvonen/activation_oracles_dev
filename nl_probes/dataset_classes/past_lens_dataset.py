import gc
import json
import os
import random
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Generator, Literal

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import (
    SPECIAL_TOKEN,
    TrainingDataPoint,
    find_pattern_in_tokens,
    get_introspection_prefix,
)


@dataclass
class PastLensDatasetConfig(BaseDatasetConfig):
    min_k_tokens: int = 1
    max_k_tokens: int = 20
    min_k_activations: int = 1
    max_k_activations: int = 20
    max_length: int = 2000
    directions: list[str] = field(default_factory=lambda: ["past", "future"])

    vllm_max_new_tokens: int = 200
    max_vllm_context_tokens: int = 2000

    future_chat_system_prompt_prob: float = 0.5
    system_prompt_path: str = "datasets/latentqa_datasets/train/system.json"


@dataclass
class PretrainSample:
    source: Literal["pretrain"]
    text: str


@dataclass
class ChatSample:
    source: Literal["chat"]
    conversation: list[dict[str, str]]


MixedDatasetSample = PretrainSample | ChatSample


@dataclass
class FutureGenerationCandidate:
    vllm_prompt_ids: list[int]
    k_tokens: int
    layers: list[int]
    k_acts: int


@dataclass
class PastReadyCandidate:
    k_tokens: int
    layers: list[int]
    k_acts: int
    target_token_ids: list[int]
    context_input_ids: list[int]
    context_positions: list[int]


class PastLensDatasetLoader(ActDatasetLoader):
    def __init__(
        self,
        dataset_config: DatasetLoaderConfig,
    ):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"

        self.dataset_config.dataset_name = "past_lens"

        self.dataset_params: PastLensDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "Past lens dataset only supports train split right now"
        assert self.dataset_config.num_test == 0, "Past lens dataset only supports train split right now"
        assert self.dataset_config.save_acts is False, "On-policy past lens generation only supports save_acts=False"

        if self.dataset_config.num_train < self.dataset_config.batch_size:
            raise ValueError(
                f"num_train {self.dataset_config.num_train} must be greater than or equal to batch_size {self.dataset_config.batch_size}"
            )

    def create_dataset(self) -> None:
        os.makedirs(self.dataset_config.dataset_folder, exist_ok=True)
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        dataset = hf_mixed_dataset_to_generator(tokenizer)

        training_data = collect_past_lens_on_policy_targets(
            dataset_config=self.dataset_config,
            custom_dataset_params=self.dataset_params,
            tokenizer=tokenizer,
            dataset=dataset,
            num_datapoints=self.dataset_config.num_train,
        )

        self.save_dataset(training_data, "train")


def hf_mixed_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "HuggingFaceFW/fineweb",
    chat_dataset: str = "lmsys/lmsys-chat-1m",
    min_chars: int = 1,
    pretrain_frac: float = 0.5,
    split: str = "train",
    streaming: bool = True,
    pretrain_key: str = "text",
    chat_key: str = "conversation",
    sequence_pack_pretrain: bool = False,
    sequence_pack_chat: bool = False,
) -> Generator[MixedDatasetSample, None, None]:
    """Yield a mixed stream of pretrain and chat samples.

    - Pretrain samples are yielded as raw text.
    - Chat samples are yielded as raw conversation message lists.
    """
    if not 0 < pretrain_frac < 1:
        raise ValueError("main_frac must be between 0 and 1 (exclusive)")

    assert min_chars > 0
    assert sequence_pack_pretrain is False, "sequence_pack_pretrain is not supported in on-policy mode"
    assert sequence_pack_chat is False, "sequence_pack_chat is not supported in on-policy mode"

    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))
    chat_ds = iter(load_dataset(chat_dataset, split=split, streaming=streaming))

    frac = Fraction(pretrain_frac).limit_denominator()
    n_pretrain = frac.numerator
    n_chat = frac.denominator - n_pretrain

    def gen() -> Generator[MixedDatasetSample, None, None]:
        while True:
            for _ in range(n_pretrain):
                sample = next(pretrain_ds)[pretrain_key]
                yield PretrainSample(source="pretrain", text=sample)

            for _ in range(n_chat):
                sample = next(chat_ds)[chat_key]
                yield ChatSample(source="chat", conversation=sample)

    return gen()


def load_system_prompts(system_prompt_path: str) -> list[str]:
    path = Path(system_prompt_path)
    data = json.loads(path.read_text())
    prompts = [row["system"] for row in data]
    assert len(prompts) > 0, f"No system prompts in {system_prompt_path}"
    return prompts


def prepend_system_prompt_for_chat(
    model_name: str,
    chat_messages: list[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, str]]:
    if "gemma-2" in model_name.lower():
        first_user_idx = -1
        for idx, message in enumerate(chat_messages):
            if message["role"] == "user":
                first_user_idx = idx
                break
        if first_user_idx < 0:
            raise ValueError("No user message found for Gemma-2 system prompt fallback")

        updated = list(chat_messages)
        user_msg = updated[first_user_idx]["content"]
        updated[first_user_idx] = {"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}
        return updated

    return [{"role": "system", "content": system_prompt}] + chat_messages


def render_sample_to_text(
    sample: MixedDatasetSample,
    direction: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    bos_token: str,
    system_prompts: list[str],
    future_chat_system_prompt_prob: float,
) -> str | None:
    if sample.source == "pretrain":
        return bos_token + sample.text

    filtered_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in sample.conversation
        if message["role"] in {"user", "assistant"}
    ]
    if len(filtered_messages) == 0:
        return None

    inject_system_prompt = direction == "future" and random.random() < future_chat_system_prompt_prob
    if inject_system_prompt:
        first_user_idx = -1
        for idx, message in enumerate(filtered_messages):
            if message["role"] == "user":
                first_user_idx = idx
                break
        if first_user_idx < 0:
            return None

        # If we inject a system prompt, force context to the first user turn.
        # This avoids contradictions with earlier assistant behavior.
        chat_messages = [filtered_messages[first_user_idx]]
        system_prompt = random.choice(system_prompts)
        chat_messages = prepend_system_prompt_for_chat(model_name, chat_messages, system_prompt)
    else:
        prefix_len = random.randint(1, len(filtered_messages))
        chat_messages = filtered_messages[:prefix_len]

    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def build_past_ready_candidate(
    input_ids: list[int],
    k_tokens: int,
    layers: list[int],
    k_acts: int,
) -> PastReadyCandidate | None:
    seq_len = len(input_ids)
    if seq_len < k_tokens + k_acts + 1:
        return None

    act_begin_min = k_tokens
    act_begin_max = seq_len - k_acts - 1
    if act_begin_max < act_begin_min:
        return None

    selected_act_begin_idx = random.randint(act_begin_min, act_begin_max)
    selected_act_positions = list(range(selected_act_begin_idx, selected_act_begin_idx + k_acts))
    selected_tokens_positions = list(range(selected_act_begin_idx - k_tokens, selected_act_begin_idx))

    context_cutoff = selected_act_positions[-1]
    target_token_ids = [input_ids[idx] for idx in selected_tokens_positions]
    context_input_ids = input_ids[: context_cutoff + 1]

    return PastReadyCandidate(
        k_tokens=k_tokens,
        layers=layers,
        k_acts=k_acts,
        target_token_ids=target_token_ids,
        context_input_ids=context_input_ids,
        context_positions=selected_act_positions,
    )


def create_training_datapoint_from_target_token_ids(
    datapoint_type: str,
    prompt: str,
    target_token_ids: list[int],
    layers: list[int],
    num_positions: int,
    tokenizer: AutoTokenizer,
    feature_idx: int,
    context_input_ids: list[int],
    context_positions: list[int],
) -> TrainingDataPoint:
    prefix = get_introspection_prefix(layers, num_positions)
    assert prefix not in prompt, f"Prefix {prefix} found in prompt {prompt}"
    prompt = prefix + prompt

    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    assert len(target_token_ids) > 0, "target_token_ids must be non-empty"

    full_prompt_ids = input_prompt_ids + target_token_ids
    labels = [-100] * len(input_prompt_ids) + target_token_ids

    positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, layers, num_positions, tokenizer)

    assert len(context_positions) == num_positions, (
        f"context_positions length {len(context_positions)} does not match num_positions {num_positions}"
    )

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layers=layers,
        steering_vectors=None,
        positions=positions,
        feature_idx=feature_idx,
        target_output=tokenizer.decode(target_token_ids, skip_special_tokens=False),
        datapoint_type=datapoint_type,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        ds_label=None,
        meta_info={},
    )

    return training_data_point


def materialize_future_candidates(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    candidates: list[FutureGenerationCandidate],
    max_to_add: int,
) -> list[TrainingDataPoint]:
    assert max_to_add > 0
    assert len(candidates) > 0

    outputs = llm.generate(
        prompts=[{"prompt_token_ids": c.vllm_prompt_ids} for c in candidates],
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    added: list[TrainingDataPoint] = []
    for candidate, output in zip(candidates, outputs, strict=True):
        if len(output.outputs) == 0:
            continue
        generated_token_ids = output.outputs[0].token_ids
        if len(generated_token_ids) == 0:
            continue

        combined_ids = candidate.vllm_prompt_ids + generated_token_ids
        prompt_len = len(candidate.vllm_prompt_ids)
        combined_len = len(combined_ids)

        # Choose activation window by its end index on prompt+generated tokens.
        # We require the window to include at least one generated token and the
        # target window to stay in generated tokens.
        act_end_min = max(candidate.k_acts, prompt_len)
        act_end_max = combined_len - candidate.k_tokens - 1
        if act_end_max < act_end_min:
            continue

        selected_act_end_idx = random.randint(act_end_min, act_end_max)
        selected_act_begin_idx = selected_act_end_idx - candidate.k_acts + 1
        selected_act_positions = list(range(selected_act_begin_idx, selected_act_end_idx + 1))

        target_start_idx = selected_act_end_idx + 1
        target_end_idx = target_start_idx + candidate.k_tokens
        target_token_ids = combined_ids[target_start_idx:target_end_idx]
        if len(target_token_ids) != candidate.k_tokens:
            continue

        context_input_ids = combined_ids[: selected_act_end_idx + 1]
        dp = create_training_datapoint_from_target_token_ids(
            datapoint_type=dataset_name,
            prompt=f"Can you predict the next {candidate.k_tokens} generated tokens that come after this?",
            target_token_ids=target_token_ids,
            layers=candidate.layers,
            num_positions=candidate.k_acts,
            tokenizer=tokenizer,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=selected_act_positions,
        )
        added.append(dp)
        if len(added) >= max_to_add:
            break

    return added


def collect_past_lens_on_policy_targets(
    dataset_config: DatasetLoaderConfig,
    custom_dataset_params: PastLensDatasetConfig,
    tokenizer: AutoTokenizer,
    dataset: Generator[MixedDatasetSample, None, None],
    num_datapoints: int,
) -> list[TrainingDataPoint]:
    random.seed(dataset_config.seed)
    torch.manual_seed(dataset_config.seed)

    assert dataset_config.save_acts is False, "On-policy collection only supports save_acts=False"

    assert dataset_config.layer_combinations, "layer_combinations must be non-empty"
    act_layer_combinations = [
        [layer_percent_to_layer(dataset_config.model_name, layer_percent) for layer_percent in layer_combo]
        for layer_combo in dataset_config.layer_combinations
    ]

    valid_directions = {"past", "future"}
    assert len(custom_dataset_params.directions) > 0, "directions must be non-empty"
    assert set(custom_dataset_params.directions).issubset(valid_directions), (
        f"directions must be in {valid_directions}, got {custom_dataset_params.directions}"
    )

    assert 0.0 <= custom_dataset_params.future_chat_system_prompt_prob <= 1.0
    assert custom_dataset_params.vllm_max_new_tokens > 0
    assert custom_dataset_params.max_vllm_context_tokens > 0

    bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    assert bos_token is not None, "Tokenizer must define at least one of bos_token/eos_token"

    system_prompts = load_system_prompts(custom_dataset_params.system_prompt_path)

    llm = LLM(model=dataset_config.model_name)
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=custom_dataset_params.vllm_max_new_tokens,
        detokenize=False,
    )

    training_data: list[TrainingDataPoint] = []
    future_candidates: list[FutureGenerationCandidate] = []
    pbar = tqdm(total=num_datapoints, desc="Collecting past lens on-policy targets")
    try:
        while len(training_data) < num_datapoints:
            sample = next(dataset)
            direction = random.choice(custom_dataset_params.directions)
            k_tokens = random.randint(custom_dataset_params.min_k_tokens, custom_dataset_params.max_k_tokens)
            k_acts = random.randint(custom_dataset_params.min_k_activations, custom_dataset_params.max_k_activations)
            layers = random.choice(act_layer_combinations)

            rendered_input = render_sample_to_text(
                sample=sample,
                direction=direction,
                tokenizer=tokenizer,
                model_name=dataset_config.model_name,
                bos_token=bos_token,
                system_prompts=system_prompts,
                future_chat_system_prompt_prob=custom_dataset_params.future_chat_system_prompt_prob,
            )
            if rendered_input is None:
                continue

            input_ids = tokenizer(
                rendered_input,
                return_tensors=None,
                truncation=True,
                max_length=custom_dataset_params.max_length,
                add_special_tokens=False,
            )["input_ids"]
            if len(input_ids) == 0:
                continue

            if direction == "past":
                past_candidate = build_past_ready_candidate(
                    input_ids=input_ids,
                    k_tokens=k_tokens,
                    layers=layers,
                    k_acts=k_acts,
                )
                if past_candidate is None:
                    continue

                dp = create_training_datapoint_from_target_token_ids(
                    datapoint_type=dataset_config.dataset_name,
                    prompt=f"Can you predict the previous {past_candidate.k_tokens} tokens that came before this?",
                    target_token_ids=past_candidate.target_token_ids,
                    layers=past_candidate.layers,
                    num_positions=past_candidate.k_acts,
                    tokenizer=tokenizer,
                    feature_idx=-1,
                    context_input_ids=past_candidate.context_input_ids,
                    context_positions=past_candidate.context_positions,
                )
                training_data.append(dp)
                pbar.update(1)
            else:
                vllm_prompt_ids = input_ids
                if len(vllm_prompt_ids) > custom_dataset_params.max_vllm_context_tokens:
                    vllm_prompt_ids = vllm_prompt_ids[-custom_dataset_params.max_vllm_context_tokens :]

                future_candidates.append(
                    FutureGenerationCandidate(
                        vllm_prompt_ids=vllm_prompt_ids,
                        k_tokens=k_tokens,
                        layers=layers,
                        k_acts=k_acts,
                    )
                )

            remaining_slots = num_datapoints - len(training_data)
            should_flush_future = remaining_slots <= len(future_candidates)
            if should_flush_future and len(future_candidates) > 0:
                added_future = materialize_future_candidates(
                    llm=llm,
                    sampling_params=sampling_params,
                    tokenizer=tokenizer,
                    dataset_name=dataset_config.dataset_name,
                    candidates=future_candidates,
                    max_to_add=remaining_slots,
                )
                training_data.extend(added_future)
                pbar.update(len(added_future))
                future_candidates = []
    finally:
        pbar.close()
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return training_data


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    layer_combinations = [[25, 50, 75]]

    batch_size = 128
    num_datapoints = 60_000
    seed = 42
    sft_data_folder = "sft_training_data"

    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=PastLensDatasetConfig(),
        dataset_folder=sft_data_folder,
        num_train=num_datapoints,
        num_test=0,
        splits=["train"],
        model_name=model_name,
        layer_combinations=layer_combinations,
        seed=seed,
        save_acts=False,
        batch_size=batch_size,
    )

    past_lens_dataset_loader = PastLensDatasetLoader(
        dataset_config=dataset_config,
    )
    past_lens_dataset_loader.create_dataset()
