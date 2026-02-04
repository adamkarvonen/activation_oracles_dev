# %%
import torch
from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import SPECIAL_TOKEN

# %%
# Adjust these two lines for the dataset you want to inspect.
DATA_PATH = "sft_training_data/latentqa_model_Qwen3-4B_n_100000_save_acts_False_train_e92f7259a7a3.pt"
NUM_EXAMPLES = 5

# %%
data = torch.load(DATA_PATH, weights_only=False)
print("Config:", data["config"])
print(f"\nTotal datapoints: {len(data['data'])}")

# %%
model_name = data["config"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
assert len(special_token_id) == 1, f"SPECIAL_TOKEN should be a single token, got {special_token_id}"
special_token_id = special_token_id[0]

# %%
for i, item in enumerate(data["data"][:NUM_EXAMPLES]):
    print(f"\n{'=' * 80}")
    print(f"Datapoint {i}: {item['datapoint_type']}")
    print(f"Layers: {item['layers']}")
    print(f"{'=' * 80}")

    input_ids = item["input_ids"]
    positions = item["positions"]
    layers = item["layers"]
    num_layers = len(layers)

    context_positions = item["context_positions"]
    if context_positions is None:
        num_positions = len(positions) // num_layers
    else:
        num_positions = len(context_positions)

    assert len(positions) == num_layers * num_positions

    found_positions = [idx for idx, tok in enumerate(input_ids) if tok == special_token_id]
    assert found_positions == positions

    for layer_idx in range(num_layers):
        block = positions[layer_idx * num_positions : (layer_idx + 1) * num_positions]
        assert block[-1] - block[0] == num_positions - 1
        final_pos = block[-1] + 1
        final_tokens = input_ids[final_pos : final_pos + 2]
        final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
        assert "\n" in final_str

    print(f"Num layers: {num_layers}")
    print(f"Num positions per layer: {num_positions}")
    print(f"Total placeholder positions: {len(positions)}")

    print("\n--- Input (detokenized) ---")
    print(tokenizer.decode(input_ids))

    print("\nInput IDs:")
    print(input_ids)

    print("\nPositions:")
    print(positions)

    print("\nLayers:")
    print(layers)

    for i in range(len(input_ids)):
        if i in positions:
            print(f"Input id {i}: {tokenizer.decode(input_ids[i])} (in positions)")
        else:
            print(f"Input id {i}: {tokenizer.decode(input_ids[i])}")

    print("\n--- Target Output ---")
    print(item["target_output"])

    print("\n--- Context (detokenized) ---")
    context_input_ids = item["context_input_ids"]
    if context_input_ids is None:
        print("None")
    else:
        print(tokenizer.decode(context_input_ids))

    print("\n--- Steering Vectors ---")
    steering_vectors = item["steering_vectors"]
    if steering_vectors is None:
        print("None")
    else:
        sv_shape = torch.tensor(steering_vectors).shape
        print(f"Shape: {sv_shape}")
        assert sv_shape[0] == len(positions)
