import argparse
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import SPECIAL_TOKEN


# Defaults for quick use; override with CLI flags.
DEFAULT_DATA_PATH = "sft_training_data/dry_run_on_policy/past_lens_model_Qwen3-4B_n_8_save_acts_False_train_1539bbab7d58.pt"
DEFAULT_NUM_EXAMPLES = 5


def infer_direction(prompt_text: str) -> str:
    if "previous" in prompt_text:
        return "past"
    if "next" in prompt_text and "generated" in prompt_text:
        return "future_generated"
    return "unknown"


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


def print_context_window(
    tokenizer: AutoTokenizer,
    context_ids: list[int],
    center_idx: int,
    window: int,
) -> None:
    start = max(0, center_idx - window)
    end = min(len(context_ids), center_idx + window + 1)

    print(f"    Window [{start}:{end}] around context index {center_idx}:")
    for i in range(start, end):
        marker = "<ACT>" if i == center_idx else "     "
        tok = decode_token(tokenizer, context_ids[i])
        print(f"      {marker} idx={i:4d} id={context_ids[i]:7d} tok={tok}")


def print_layer_placeholder_positions(positions: list[int], layers: list[int], num_positions: int) -> None:
    print("\n--- Placeholder Positions In Full Input ---")
    by_layer: dict[int, list[int]] = {}
    for layer_idx, layer in enumerate(layers):
        start = layer_idx * num_positions
        end = (layer_idx + 1) * num_positions
        by_layer[layer] = positions[start:end]

    for layer in layers:
        print(f"Layer {layer}: {by_layer[layer]}")


def inspect_datapoint(item: dict, tokenizer: AutoTokenizer, datapoint_idx: int, context_window: int) -> None:
    print(f"\n{'=' * 120}")
    print(f"Datapoint {datapoint_idx}")
    print(f"{'=' * 120}")

    input_ids = item["input_ids"]
    labels = item["labels"]
    layers = item["layers"]
    positions = item["positions"]
    context_input_ids = item["context_input_ids"]
    context_positions = item["context_positions"]

    first_target_idx = next((i for i, lab in enumerate(labels) if lab != -100), len(labels))
    prompt_ids = input_ids[:first_target_idx]
    target_ids = input_ids[first_target_idx:]

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)

    direction = infer_direction(prompt_text)

    print(f"datapoint_type: {item['datapoint_type']}")
    print(f"direction: {direction}")
    print(f"layers: {layers}")
    print(f"input_len: {len(input_ids)} | prompt_len: {len(prompt_ids)} | target_len: {len(target_ids)}")
    print(f"feature_idx: {item['feature_idx']}")

    if context_positions is None:
        num_positions = len(positions) // len(layers)
    else:
        num_positions = len(context_positions)

    print(f"num_positions_per_layer: {num_positions}")
    print(f"total_placeholder_positions: {len(positions)}")

    print("\n--- Prompt Text (Decoded) ---")
    print(prompt_text)

    print("\n--- Target Text (Decoded) ---")
    print(target_text)

    if context_input_ids is None or context_positions is None:
        print("\n--- Context ---")
        print("None")
        return

    print("\n--- Context Prompt (Decoded) ---")
    print(tokenizer.decode(context_input_ids, skip_special_tokens=False))

    print("\n--- Selected Activation Positions In Context ---")
    print(context_positions)

    for pos in context_positions:
        tok = decode_token(tokenizer, context_input_ids[pos])
        print(f"  context_idx={pos:4d} token_id={context_input_ids[pos]:7d} token={tok}")
        print_context_window(tokenizer, context_input_ids, pos, context_window)

    print_layer_placeholder_positions(positions, layers, num_positions)

    # Helpful cross-check for placeholder token locations.
    special_token_ids = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    if len(special_token_ids) == 1:
        special_token_id = special_token_ids[0]
        found_positions = [idx for idx, tok in enumerate(input_ids) if tok == special_token_id]
        if found_positions != positions:
            print("\nWARNING: Found placeholder token positions do not match saved positions.")
            print(f"found_positions: {found_positions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ML-AO training datapoints in terminal.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=20)
    args = parser.parse_args()

    data = torch.load(args.data_path, weights_only=False)
    print("Config:", data["config"])
    print(f"Total datapoints: {len(data['data'])}")

    model_name = data["config"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    end_idx = min(len(data["data"]), args.start_idx + args.num_examples)
    for i in range(args.start_idx, end_idx):
        inspect_datapoint(data["data"][i], tokenizer, i, args.context_window)
