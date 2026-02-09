import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist

import nl_probes.sft as sft
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.utils.common import load_tokenizer

# Quick-loop defaults for fast iteration.
QUICK_MODEL_NAME = "Qwen/Qwen3-8B"
QUICK_LAYER_COMBINATION = [25, 50, 75]
QUICK_NUM_TRAIN = 1_000
QUICK_GLOBAL_TRAIN_BATCH_SIZE = 8
QUICK_EVAL_STEPS = 50
QUICK_SAVE_STEPS = 10_000

# Keep open-ended evals enabled in this quick script.
sft.ENABLE_PERSONAQA_OPEN_ENDED_EVAL = True
sft.ENABLE_TABOO_OPEN_ENDED_EVAL = True
sft.TABOO_OPEN_ENDED_TRUNCATED = True


def build_quick_past_lens_loader(
    *,
    model_name: str,
    layer_combinations: list[list[int]],
    train_batch_size: int,
) -> PastLensDatasetLoader:
    return PastLensDatasetLoader(
        dataset_config=DatasetLoaderConfig(
            custom_dataset_params=PastLensDatasetConfig(
                max_k_activations=1,
                max_k_tokens=50,
            ),
            num_train=QUICK_NUM_TRAIN,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_combinations=layer_combinations,
            save_acts=False,
            batch_size=train_batch_size,
        )
    )


if __name__ == "__main__":
    """python nl_probes/sft_quick.py --gen-only && torchrun --nproc_per_node=1 nl_probes/sft_quick.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen-only",
        action="store_true",
        help="Generate/load quick dataset on disk and exit before training.",
    )
    args = parser.parse_args()

    if args.gen_only:
        torchrun_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torchrun_local_rank != 0:
            print("Skipping dataset generation on non-zero LOCAL_RANK in --gen-only mode")
            raise SystemExit(0)
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    assert QUICK_GLOBAL_TRAIN_BATCH_SIZE % world_size == 0, (
        f"Global batch size {QUICK_GLOBAL_TRAIN_BATCH_SIZE} must be divisible by world_size {world_size}"
    )
    train_batch_size = QUICK_GLOBAL_TRAIN_BATCH_SIZE // world_size

    model_kwargs: dict = {}
    layer_combinations = [QUICK_LAYER_COMBINATION]

    dataset_loaders = [
        build_quick_past_lens_loader(
            model_name=QUICK_MODEL_NAME,
            layer_combinations=layer_combinations,
            train_batch_size=train_batch_size,
        )
    ]

    model_name_str = QUICK_MODEL_NAME.split("/")[-1].replace(".", "_").replace(" ", "_")
    cfg = SelfInterpTrainingConfig(
        model_name=QUICK_MODEL_NAME,
        hook_onto_layer=1,
        hf_repo_name="N/A",
        layer_combinations=layer_combinations,
        train_batch_size=train_batch_size,
        activation_collection_batch_size=train_batch_size * 4,
        eval_batch_size=train_batch_size * 8,
        num_epochs=1,
        eval_steps=QUICK_EVAL_STEPS,
        eval_on_start=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        save_steps=QUICK_SAVE_STEPS,
        wandb_suffix=f"_quick_past_lens_1k_{model_name_str}",
        load_lora_path=None,
    )

    cfg.finalize(dataset_loaders=dataset_loaders)

    print(f"save dir: {cfg.save_dir}")

    if local_rank == 0:
        sft._ensure_datasets_exist(dataset_loaders)
    if not args.gen_only:
        dist.barrier()

    if args.gen_only:
        if local_rank == 0:
            print("Dataset generation complete (--gen-only); exiting before training.")
        raise SystemExit(0)

    tokenizer = load_tokenizer(cfg.model_name)

    all_training_data, all_eval_data = sft.build_datasets(
        cfg,
        dataset_loaders=dataset_loaders,
        window_mult=cfg.window_mult,
    )

    print(f"training data length: {len(all_training_data)}, eval data length: {len(all_eval_data)}")
    print(asdict(cfg))

    sft.train_model(
        cfg=cfg,
        training_data=all_training_data,
        eval_datasets=all_eval_data,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_kwargs=model_kwargs,
        verbose=True,
    )

    dist.destroy_process_group()
