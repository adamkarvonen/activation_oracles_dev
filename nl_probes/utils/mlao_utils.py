import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from nl_probes.utils.dataset_utils import SPECIAL_TOKEN, TrainingDataPoint

MLAO_CONFIG_FILENAME = "mlao_config.json"


def build_mlao_config(model_name: str, layer_percents: list[int], act_layers: list[int]) -> dict:
    assert len(layer_percents) == len(act_layers)
    return {
        "model_name": model_name,
        "layer_percents": layer_percents,
        "act_layers": act_layers,
        "order": "layer_major",
        "special_token": SPECIAL_TOKEN,
        "prefix_template": "Layer: {layer}\\n{special_token} * {num_positions} \\n",
    }


def write_mlao_config(save_dir: str | Path, *, model_name: str, layer_percents: list[int], act_layers: list[int]) -> None:
    cfg = build_mlao_config(model_name, layer_percents, act_layers)
    save_path = Path(save_dir) / MLAO_CONFIG_FILENAME
    save_path.write_text(json.dumps(cfg, indent=2))


def read_mlao_config(path_or_repo: str) -> dict:
    p = Path(path_or_repo)
    if p.exists():
        cfg_path = p / MLAO_CONFIG_FILENAME
        return json.loads(cfg_path.read_text())

    cfg_path = hf_hub_download(repo_id=path_or_repo, filename=MLAO_CONFIG_FILENAME)
    return json.loads(Path(cfg_path).read_text())


def assert_eval_datapoint_layers(dp: TrainingDataPoint, expected_layers: list[int]) -> None:
    assert dp.layers == expected_layers, f"Expected layers {expected_layers}, got {dp.layers}"
    if dp.context_positions is not None:
        assert len(dp.positions) == len(dp.context_positions) * len(dp.layers)
    if dp.steering_vectors is not None:
        assert dp.steering_vectors.shape[0] == len(dp.positions)
