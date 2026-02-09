# nl_tests/test_peft_adapters_disabled.py
"""
Tiny sanity test:
- Add a LoRA adapter with non-zero init (init_lora_weights=False)
- Verify enable_adapters()/disable_adapters() flip logits
- Disabled ≈ base, restored ≈ enabled
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.base_experiment import VerbalizerEvalConfig, collect_target_activations, encode_messages
from nl_probes.utils.activation_utils import get_hf_submodule


@pytest.mark.parametrize("model_name", ["facebook/opt-125m"])
@torch.no_grad()
def test_enable_disable_adapters_simple(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    inputs = tok("hello world", return_tensors="pt")
    base_logits = model(**inputs).logits.clone()

    # Non-zero LoRA init so it actually does something without training
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",  # simple and model-agnostic for OPT
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=False,  # <- key simplification
    )
    model = get_peft_model(model, lora_cfg)

    # ON
    logits_on = model(**inputs).logits
    # model.disable_adapters()

    # OFF
    with model.disable_adapter():
        logits_off = model(**inputs).logits
    # model.enable_adapters()

    # Back ON
    logits_restored = model(**inputs).logits

    # Assertions
    assert (logits_on - logits_off).abs().max().item() > 1e-6, "Adapters did not change logits"
    assert torch.allclose(logits_off, base_logits, rtol=1e-5, atol=1e-7), "Disabled adapters did not match base"
    assert torch.allclose(logits_on, logits_restored, rtol=1e-5, atol=1e-7), "State not restored after re-enabling"


@pytest.fixture(scope="module")
def qwen3_06b_peft_bundle():
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-0.6B integration test requires CUDA")

    model_name = "Qwen/Qwen3-0.6B"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # Base-model path used by standard collection/eval code.
    base_submodule = get_hf_submodule(model, layer=1)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.eval()
    peft_model.set_adapter("default")

    yield {
        "model_name": model_name,
        "device": device,
        "tokenizer": tokenizer,
        "base_submodule": base_submodule,
        "peft_model": peft_model,
    }

    del peft_model
    torch.cuda.empty_cache()


@torch.no_grad()
def test_qwen3_06b_get_hf_submodule_base_and_peft(qwen3_06b_peft_bundle):
    base_submodule = qwen3_06b_peft_bundle["base_submodule"]
    peft_model = qwen3_06b_peft_bundle["peft_model"]

    peft_submodule_default = get_hf_submodule(peft_model, layer=1)
    peft_submodule_use_lora = get_hf_submodule(peft_model, layer=1, use_lora=True)

    assert type(base_submodule) is type(peft_submodule_default)
    assert type(base_submodule) is type(peft_submodule_use_lora)


@torch.no_grad()
def test_qwen3_06b_collect_target_activations_peft_path(qwen3_06b_peft_bundle):
    model_name = qwen3_06b_peft_bundle["model_name"]
    device = qwen3_06b_peft_bundle["device"]
    tokenizer = qwen3_06b_peft_bundle["tokenizer"]
    peft_model = qwen3_06b_peft_bundle["peft_model"]

    config = VerbalizerEvalConfig(
        model_name=model_name,
        activation_input_types=["orig", "lora", "diff"],
        layer_combinations=[[25]],
        selected_layer_combination=[25],
        eval_batch_size=1,
    )

    inputs_BL = encode_messages(
        tokenizer=tokenizer,
        message_dicts=[[{"role": "user", "content": "Hello there."}]],
        add_generation_prompt=config.add_generation_prompt,
        enable_thinking=config.enable_thinking,
        device=device,
    )

    # This path previously failed when adapter toggling used incompatible methods.
    act_types = collect_target_activations(
        model=peft_model,
        inputs_BL=inputs_BL,
        config=config,
        target_lora_path="default",
    )

    assert set(act_types.keys()) == {"orig", "lora", "diff"}
    assert config.selected_act_layers[0] in act_types["orig"]
    assert config.selected_act_layers[0] in act_types["lora"]
    assert config.selected_act_layers[0] in act_types["diff"]
