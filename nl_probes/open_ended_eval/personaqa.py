import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerInputInfo, VerbalizerResults
from nl_probes.utils.common import load_model, load_tokenizer

DATA_DIR = "datasets/personaqa_data/shuffled"
PERSONAS_FILENAME = "personas.jsonl"
CONTEXT_PROMPTS: tuple[str, ...] = ("My name is {name}.",)
PROMPT_TYPES: tuple[str, ...] = (
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
)
VERBALIZER_PROMPT_PREFIX = "Answer with the correct value only. "
VERBALIZER_PROMPTS: tuple[str, ...] = (
    "Which country is this person from?",
    "What is this person's favorite food?",
    "What is this person's favorite drink?",
    "What is this person's favorite music genre?",
    "What is this person's favorite sport?",
    "What is this person's favorite boardgame?",
)
DEFAULT_SELECTED_LAYER_COMBINATION = [25, 50, 75]
DEFAULT_GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 40,
}


def normalize_answer(answer: str) -> str:
    return answer.rstrip(".!?,;:").strip().lower()


def compute_accuracy_metrics(results: list[VerbalizerResults]) -> dict[str, float]:
    full_sequence_total = 0
    full_sequence_correct = 0
    segment_total = 0
    segment_correct = 0
    token_total = 0
    token_correct = 0

    for record in results:
        ground_truth = normalize_answer(record.ground_truth)
        for response in record.full_sequence_responses:
            full_sequence_total += 1
            full_sequence_correct += normalize_answer(response) == ground_truth
        for response in record.segment_responses:
            segment_total += 1
            segment_correct += normalize_answer(response) == ground_truth
        for response in record.token_responses:
            if response is None:
                continue
            token_total += 1
            token_correct += normalize_answer(response) == ground_truth

    assert full_sequence_total > 0
    assert segment_total > 0
    assert token_total > 0

    return {
        "full_sequence_accuracy": full_sequence_correct / full_sequence_total,
        "segment_accuracy": segment_correct / segment_total,
        "token_accuracy": token_correct / token_total,
        "full_sequence_total": float(full_sequence_total),
        "segment_total": float(segment_total),
        "token_total": float(token_total),
    }


def load_persona_data(max_personas: int | None) -> list[dict[str, Any]]:
    data_path = Path(DATA_DIR) / PERSONAS_FILENAME
    assert data_path.exists(), f"Could not find {data_path}"
    persona_data = [json.loads(line) for line in data_path.read_text().splitlines()]
    persona_data.sort(key=lambda x: x["name"])
    if max_personas is not None:
        persona_data = persona_data[:max_personas]
    return persona_data


def build_verbalizer_prompt_infos(
    persona_data: list[dict[str, Any]],
    context_prompts: tuple[str, ...],
    prompt_types: tuple[str, ...],
    verbalizer_prompts: tuple[str, ...],
) -> list[VerbalizerInputInfo]:
    prefixed_prompts = [VERBALIZER_PROMPT_PREFIX + p for p in verbalizer_prompts]
    assert len(prompt_types) == len(prefixed_prompts)
    pt_to_prompt = {k: v for k, v in zip(prompt_types, prefixed_prompts, strict=True)}

    verbalizer_prompt_infos: list[VerbalizerInputInfo] = []
    for context_prompt in context_prompts:
        for persona in persona_data:
            persona_name = persona["name"]
            formatted_context_prompt = context_prompt.format(name=persona_name)
            formatted_prompt = [{"role": "user", "content": formatted_context_prompt}]

            for prompt_type in prompt_types:
                context_prompt_info = VerbalizerInputInfo(
                    context_prompt=formatted_prompt,
                    ground_truth=str(persona[prompt_type]),
                    verbalizer_prompt=pt_to_prompt[prompt_type],
                )
                verbalizer_prompt_infos.append(context_prompt_info)
    return verbalizer_prompt_infos


def ensure_default_adapter(model: AutoModelForCausalLM) -> None:
    if not hasattr(model, "peft_config") or "default" not in model.peft_config:
        dummy_config = LoraConfig()
        model.add_adapter(dummy_config, adapter_name="default")


def build_default_personaqa_verbalizer_eval_config(
    model_name: str,
    segment_start: int,
    selected_layer_combination: list[int] | None = None,
) -> base_experiment.VerbalizerEvalConfig:
    if selected_layer_combination is None:
        selected_layer_combination = DEFAULT_SELECTED_LAYER_COMBINATION

    return base_experiment.VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["lora"],
        eval_batch_size=512,
        verbalizer_generation_kwargs=DEFAULT_GENERATION_KWARGS,
        full_seq_repeats=1,
        segment_repeats=1,
        segment_start_idx=segment_start,
        token_start_idx=-20,
        layer_combinations=[selected_layer_combination],
        selected_layer_combination=selected_layer_combination,
    )


def get_default_personaqa_model_settings(model_name: str) -> dict[str, Any]:
    if model_name == "Qwen/Qwen3-8B":
        return {
            "target_lora_suffixes": [
                "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs",
            ],
            "verbalizer_lora_paths": [
                "adamkarvonen/checkpoints_latentqa_cls_on_policy_Qwen3-8B",
            ],
            "target_lora_path_template": "{lora_path}",
            "segment_start": -20,
            "model_kwargs": {},
        }

    if model_name == "google/gemma-2-9b-it":
        return {
            "target_lora_suffixes": [
                "adamkarvonen/gemma-2-9b-it-shuffled_3_epochs",
            ],
            "verbalizer_lora_paths": [
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_1e-6",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-6",
                "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-5",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_1e-4",
                "adamkarvonen/checkpoints_latentqa_only_gemma-2-9b-it_lr_3e-4",
            ],
            "target_lora_path_template": "{lora_path}",
            "segment_start": -20,
            "model_kwargs": {},
        }

    if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        return {
            "target_lora_suffixes": [
                "adamkarvonen/Llama-3_3-70B-Instruct-shuffled_3_epochs_v2",
            ],
            "verbalizer_lora_paths": [
                "checkpoints_latentqa_layer_0_Llama-3_3-70B-Instruct/final",
            ],
            "target_lora_path_template": "{lora_path}",
            "segment_start": -20,
            "model_kwargs": {"quantization_config": bnb_config},
        }

    raise ValueError(f"Unsupported MODEL_NAME: {model_name}")


def run_personaqa_open_ended_eval(
    *,
    model_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    config: base_experiment.VerbalizerEvalConfig,
    target_lora_suffixes: list[str | None],
    target_lora_path_template: str,
    verbalizer_lora_paths: list[str | None] | None = None,
    verbalizer_adapter_names: list[str | None] | None = None,
    output_json_template: str | None = None,
    max_personas: int | None = None,
    context_prompts: tuple[str, ...] = CONTEXT_PROMPTS,
    prompt_types: tuple[str, ...] = PROMPT_TYPES,
    verbalizer_prompts: tuple[str, ...] = VERBALIZER_PROMPTS,
) -> dict[str, Any]:
    assert (verbalizer_lora_paths is None) != (verbalizer_adapter_names is None), (
        "Provide exactly one of verbalizer_lora_paths or verbalizer_adapter_names"
    )

    persona_data = load_persona_data(max_personas=max_personas)
    verbalizer_prompt_infos = build_verbalizer_prompt_infos(
        persona_data=persona_data,
        context_prompts=context_prompts,
        prompt_types=prompt_types,
        verbalizer_prompts=verbalizer_prompts,
    )

    ensure_default_adapter(model)
    model.eval()

    use_preloaded_verbalizer_adapters = verbalizer_adapter_names is not None
    if use_preloaded_verbalizer_adapters:
        verbalizer_entries = verbalizer_adapter_names
    else:
        verbalizer_entries = verbalizer_lora_paths
    assert verbalizer_entries is not None

    total_combos = len(verbalizer_entries) * len(target_lora_suffixes)
    combo_pbar = tqdm(total=total_combos, desc="LoRA Combo Progress", position=0)

    metrics_by_verbalizer: dict[str, dict[str, float]] = {}
    all_results: list[VerbalizerResults] = []

    for verbalizer_entry in verbalizer_entries:
        verbalizer_results: list[VerbalizerResults] = []
        sanitized_verbalizer_name: str | None = None

        if use_preloaded_verbalizer_adapters:
            sanitized_verbalizer_name = verbalizer_entry
            if sanitized_verbalizer_name is not None:
                assert sanitized_verbalizer_name in model.peft_config, (
                    f"Missing preloaded verbalizer adapter: {sanitized_verbalizer_name}"
                )
        else:
            if verbalizer_entry is not None:
                sanitized_verbalizer_name, verbalizer_training_config = base_experiment.load_oracle_adapter(
                    model, verbalizer_entry
                )
                base_experiment.assert_training_config_matches_verbalizer_eval_config(config, verbalizer_training_config)

        for target_lora_suffix in target_lora_suffixes:
            target_lora_path = None
            if target_lora_suffix is not None:
                target_lora_path = target_lora_path_template.format(lora_path=target_lora_suffix)

            sanitized_target_name = None
            if target_lora_path is not None:
                sanitized_target_name = base_experiment.load_plain_adapter(model, target_lora_path)

            print(
                "Running verbalizer eval for verbalizer: "
                f"{verbalizer_entry if not use_preloaded_verbalizer_adapters else sanitized_verbalizer_name}, "
                f"target: {target_lora_path}"
            )

            combo_pbar.set_postfix(
                {
                    "verbalizer": (
                        verbalizer_entry.split("/")[-1]
                        if (not use_preloaded_verbalizer_adapters and verbalizer_entry is not None)
                        else (sanitized_verbalizer_name if sanitized_verbalizer_name is not None else "None")
                    ),
                    "target": (target_lora_suffix.split("/")[-1] if target_lora_suffix else "None"),
                }
            )

            results = base_experiment.run_verbalizer(
                model=model,
                tokenizer=tokenizer,
                verbalizer_prompt_infos=verbalizer_prompt_infos,
                verbalizer_lora_path=sanitized_verbalizer_name,
                target_lora_path=sanitized_target_name,
                config=config,
                device=device,
            )
            verbalizer_results.extend(results)

            if sanitized_target_name is not None and sanitized_target_name in model.peft_config:
                model.delete_adapter(sanitized_target_name)

            combo_pbar.update(1)

        if verbalizer_entry is None and sanitized_verbalizer_name is None:
            verbalizer_key = "base_model"
            lora_name = "base_model"
        elif use_preloaded_verbalizer_adapters:
            assert sanitized_verbalizer_name is not None
            verbalizer_key = sanitized_verbalizer_name
            lora_name = sanitized_verbalizer_name.replace("/", "_").replace(".", "_")
        else:
            assert verbalizer_entry is not None
            verbalizer_key = verbalizer_entry.split("/")[-1]
            lora_name = verbalizer_entry.split("/")[-1].replace("/", "_").replace(".", "_")

        final_verbalizer_results = {
            "config": asdict(config),
            "verbalizer_lora_path": verbalizer_entry if not use_preloaded_verbalizer_adapters else sanitized_verbalizer_name,
            "results": [asdict(r) for r in verbalizer_results],
        }

        if output_json_template is not None:
            output_json = output_json_template.format(lora=lora_name)
            with open(output_json, "w") as f:
                json.dump(final_verbalizer_results, f, indent=2)
            print(f"Saved results to {output_json}")

        metrics_by_verbalizer[verbalizer_key] = compute_accuracy_metrics(verbalizer_results)
        all_results.extend(verbalizer_results)

        if not use_preloaded_verbalizer_adapters and sanitized_verbalizer_name is not None:
            if sanitized_verbalizer_name in model.peft_config:
                model.delete_adapter(sanitized_verbalizer_name)

    combo_pbar.close()

    overall_metrics = compute_accuracy_metrics(all_results)
    return {
        "overall_metrics": overall_metrics,
        "metrics_by_verbalizer": metrics_by_verbalizer,
        "num_results": len(all_results),
    }


def run_default_personaqa_open_ended_eval() -> None:
    model_names = [
        "Qwen/Qwen3-8B",
        # "google/gemma-2-9b-it",
        # "meta-llama/Llama-3.3-70B-Instruct",
    ]

    for model_name in model_names:
        random.seed(42)
        torch.manual_seed(42)
        torch.set_grad_enabled(False)

        model_name_str = model_name.split("/")[-1].replace(".", "_")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        settings = get_default_personaqa_model_settings(model_name)
        config = build_default_personaqa_verbalizer_eval_config(
            model_name=model_name,
            segment_start=settings["segment_start"],
        )

        experiments_dir = "experiments/personaqa_results"
        output_json_dir = f"{experiments_dir}/{model_name_str}_open_ended"
        os.makedirs(experiments_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)
        output_json_template = f"{output_json_dir}/personaqa_open_" + "{lora}.json"

        print(f"Loading tokenizer: {model_name}")
        tokenizer = load_tokenizer(model_name)
        print(f"Loading model: {model_name} on {device} with dtype={dtype}")
        model = load_model(model_name, dtype, **settings["model_kwargs"])
        model.eval()

        summary = run_personaqa_open_ended_eval(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            target_lora_suffixes=settings["target_lora_suffixes"],
            target_lora_path_template=settings["target_lora_path_template"],
            verbalizer_lora_paths=settings["verbalizer_lora_paths"],
            output_json_template=output_json_template,
            max_personas=None,
        )

        print("PersonaQA overall metrics:")
        print(summary["overall_metrics"])


if __name__ == "__main__":
    run_default_personaqa_open_ended_eval()
