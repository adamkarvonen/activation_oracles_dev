import json
import datetime
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import get_collection

from nl_probes.configs.sft_config import (
    TRAINING_CONFIG_FILENAME,
    SelfInterpTrainingConfig,
    dataset_loader_name_from_config,
    get_git_commit_hash,
    write_training_config,
)
from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig
from nl_probes.utils.common import layer_percent_to_layer


COLLECTION_ID = "adamkarvonen/activation-oracles"
OUTPUT_DIR = Path("training_config_drafts")
TRAIN_BATCH_SIZE = 16
LAYER_COMBINATIONS = [[25], [50], [75]]
SAVE_ACTS = False

MAIN_TRAIN_SIZE = 6000
MAIN_TEST_SIZE = 250
CLASSIFICATION_DATASETS = {
    "geometry_of_truth": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "relations": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "sst2": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "md_gender": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "snli": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "ag_news": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ner": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "tense": {"num_train": MAIN_TRAIN_SIZE, "num_test": MAIN_TEST_SIZE, "splits": ["train", "test"]},
    "language_identification": {
        "num_train": MAIN_TRAIN_SIZE,
        "num_test": MAIN_TEST_SIZE,
        "splits": ["test"],
        "batch_size": 4,
    },
    "singular_plural": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
}


def infer_model_name(repo_id: str) -> str:
    slug = repo_id.split("/")[-1]

    if "Qwen3-" in slug:
        suffix = slug.split("Qwen3-")[1]
        suffix = "Qwen3-" + suffix
        suffix = suffix.replace("_", ".")
        return f"Qwen/{suffix}"

    if "gemma-" in slug:
        suffix = slug.split("gemma-")[1]
        suffix = "gemma-" + suffix
        return f"google/{suffix}"

    if "Llama-" in slug:
        suffix = slug.split("Llama-")[1]
        suffix = "Llama-" + suffix
        suffix = suffix.replace("_", ".")
        return f"meta-llama/{suffix}"

    raise ValueError(f"Could not infer model name from repo_id: {repo_id}")


def infer_dataset_tokens(repo_id: str) -> list[str]:
    slug = repo_id.split("/")[-1]
    names: list[str] = []

    if "latentqa" in slug:
        names.append("latentqa")
    if "past_lens" in slug:
        names.append("past_lens")
    if "cls" in slug or "classification" in slug:
        names.append("classification")
    if "act" in slug:
        names.append("act")
    if "pretrain_mix" in slug:
        names.append("pretrain_mix")
    if "addition" in slug or "adding" in slug:
        names.append("addition")

    return names


def mk_cfg(
    custom_params,
    *,
    num_train: int,
    num_test: int,
    splits: list[str],
    model_name: str,
    batch_size: int,
    dataset_name: str,
) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        layer_combinations=LAYER_COMBINATIONS,
        save_acts=SAVE_ACTS,
        batch_size=batch_size,
        dataset_name=dataset_name,
    )


def build_dataset_configs(repo_id: str, model_name: str) -> list[DatasetLoaderConfig]:
    tokens = set(infer_dataset_tokens(repo_id))
    dataset_configs: list[DatasetLoaderConfig] = []

    if "past_lens" in tokens:
        past_lens_single = mk_cfg(
            PastLensDatasetConfig(max_k_activations=1, max_k_tokens=50),
            num_train=100_000,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            batch_size=TRAIN_BATCH_SIZE,
            dataset_name="past_lens",
        )
        past_lens_multi = mk_cfg(
            PastLensDatasetConfig(max_k_activations=50, max_k_tokens=50),
            num_train=100_000,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            batch_size=TRAIN_BATCH_SIZE,
            dataset_name="past_lens",
        )
        dataset_configs.extend([past_lens_single, past_lens_multi])

    if "latentqa" in tokens:
        latent_qa = mk_cfg(
            LatentQADatasetConfig(),
            num_train=100_000,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            batch_size=TRAIN_BATCH_SIZE,
            dataset_name="latentqa",
        )
        dataset_configs.append(latent_qa)

    if "classification" in tokens:
        for ds_name, meta in CLASSIFICATION_DATASETS.items():
            single_params = ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_window_size=1,
                min_end_offset=-1,
                max_end_offset=-5,
                num_qa_per_sample=2,
            )
            multi_params = ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_window_size=50,
                min_end_offset=-1,
                max_end_offset=-5,
                num_qa_per_sample=1,
            )

            if "batch_size" in meta:
                bs = meta["batch_size"]
            else:
                bs = TRAIN_BATCH_SIZE

            single_cfg = mk_cfg(
                single_params,
                num_train=meta["num_train"],
                num_test=meta["num_test"],
                splits=meta["splits"],
                model_name=model_name,
                batch_size=bs,
                dataset_name=f"classification_{ds_name}",
            )
            multi_cfg = mk_cfg(
                multi_params,
                num_train=meta["num_train"],
                num_test=meta["num_test"],
                splits=meta["splits"],
                model_name=model_name,
                batch_size=TRAIN_BATCH_SIZE,
                dataset_name=f"classification_{ds_name}",
            )
            dataset_configs.extend([single_cfg, multi_cfg])

    return dataset_configs


def build_training_config(repo_id: str) -> SelfInterpTrainingConfig:
    model_name = infer_model_name(repo_id)
    act_layer_combinations = [
        [layer_percent_to_layer(model_name, p) for p in combo] for combo in LAYER_COMBINATIONS
    ]
    cfg = SelfInterpTrainingConfig(
        model_name=model_name,
        layer_combinations=LAYER_COMBINATIONS,
        act_layer_combinations=act_layer_combinations,
    )
    dataset_configs = build_dataset_configs(repo_id, model_name)
    cfg.dataset_configs = [asdict(cfg_item) for cfg_item in dataset_configs]
    cfg.dataset_loader_names = [dataset_loader_name_from_config(cfg_item) for cfg_item in dataset_configs]
    cfg.created_at_utc = datetime.datetime.now(datetime.UTC).isoformat()
    cfg.git_commit = get_git_commit_hash()

    primary_act_combo = cfg.act_layer_combinations[0]
    layers_str = "-".join(map(str, primary_act_combo))
    default_run = f"{cfg.model_name}-layers_{layers_str}-decoder-{cfg.use_decoder_vectors}{cfg.wandb_suffix}"
    if not cfg.wandb_run_name:
        cfg.wandb_run_name = default_run

    if cfg.wandb_suffix and not cfg.save_dir.endswith(cfg.wandb_suffix):
        cfg.save_dir = f"{cfg.save_dir}{cfg.wandb_suffix}"

    return cfg


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    collection = get_collection(COLLECTION_ID)
    manifest = []

    for item in collection.items:
        repo_id = item.item_id
        if item.item_type != "model":
            continue

        cfg = build_training_config(repo_id)
        repo_dir = OUTPUT_DIR / repo_id.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)
        write_training_config(repo_dir, cfg)

        training_payload = asdict(cfg)
        manifest.append(
            {
                "repo_id": repo_id,
                "model_name": cfg.model_name,
                "layer_combinations": cfg.layer_combinations,
                "output_path": str(repo_dir / TRAINING_CONFIG_FILENAME),
                "training_config": training_payload,
            }
        )

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} draft configs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
