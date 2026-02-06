import argparse
import html
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import TrainingDataPoint


def find_latest_past_lens_file(dataset_folder: Path) -> Path:
    candidates = sorted(
        dataset_folder.glob("past_lens_model_*_save_acts_False_train_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    assert len(candidates) > 0, f"No past_lens train files in {dataset_folder}"
    return candidates[0]


def decode_tokens(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def decode_token(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")


def build_context_table_rows(
    tokenizer: AutoTokenizer,
    context_ids: list[int],
    context_positions: list[int],
    max_rows: int,
) -> list[str]:
    pos_set = set(context_positions)
    rows = []

    for idx, token_id in enumerate(context_ids[:max_rows]):
        token_str = html.escape(tokenizer.decode([token_id], skip_special_tokens=False))
        marker = "*" if idx in pos_set else ""
        row_class = ' class="selected"' if idx in pos_set else ""
        rows.append(f"<tr{row_class}><td>{idx}</td><td>{token_id}</td><td>{marker}</td><td><code>{token_str}</code></td></tr>")

    if len(context_ids) > max_rows:
        rows.append(
            f"<tr><td colspan='4'><em>... truncated ({len(context_ids) - max_rows} more context tokens)</em></td></tr>"
        )

    return rows


def summarize_datapoint(dp: TrainingDataPoint, tokenizer: AutoTokenizer, idx: int, max_context_rows: int) -> tuple[str, str]:
    first_target_idx = next((i for i, x in enumerate(dp.labels) if x != -100), len(dp.labels))

    prompt_ids = dp.input_ids[:first_target_idx]
    target_ids = dp.input_ids[first_target_idx:]

    prompt_text = decode_tokens(tokenizer, prompt_ids)
    target_text = decode_tokens(tokenizer, target_ids)

    context_ids = dp.context_input_ids
    context_positions = dp.context_positions

    assert context_ids is not None
    assert context_positions is not None

    for pos in context_positions:
        assert 0 <= pos < len(context_ids), (
            f"Datapoint {idx}: context position {pos} out of range for context length {len(context_ids)}"
        )

    direction = "unknown"
    if "previous" in prompt_text:
        direction = "past"
    if "next" in prompt_text and "generated" in prompt_text:
        direction = "future_generated"

    console = []
    console.append(f"Sample {idx}")
    console.append(f"  direction: {direction}")
    console.append(f"  layers: {dp.layers}")
    console.append(f"  num_positions: {len(context_positions)}")
    console.append(f"  input_len: {len(dp.input_ids)}")
    console.append(f"  prompt_len: {len(prompt_ids)}")
    console.append(f"  target_len: {len(target_ids)}")
    console.append(f"  selected_context_positions: {context_positions}")
    prompt_preview = prompt_text[:300].replace(chr(10), " ")
    target_preview = target_text[:300].replace(chr(10), " ")
    console.append(f"  prompt_text: {prompt_preview}" + ("..." if len(prompt_text) > 300 else ""))
    console.append(f"  target_text: {target_preview}" + ("..." if len(target_text) > 300 else ""))

    context_rows = "\n".join(build_context_table_rows(tokenizer, context_ids, context_positions, max_context_rows))

    card = f"""
    <section class="card">
      <h2>Sample {idx}</h2>
      <p><strong>direction:</strong> {html.escape(direction)}</p>
      <p><strong>layers:</strong> {html.escape(str(dp.layers))}</p>
      <p><strong>num_positions:</strong> {len(context_positions)}</p>
      <p><strong>input_len:</strong> {len(dp.input_ids)} | <strong>prompt_len:</strong> {len(prompt_ids)} | <strong>target_len:</strong> {len(target_ids)}</p>
      <p><strong>selected_context_positions:</strong> {html.escape(str(context_positions))}</p>
      <h3>Prompt Text</h3>
      <pre>{html.escape(prompt_text)}</pre>
      <h3>Target Text (decoded from target token IDs)</h3>
      <pre>{html.escape(target_text)}</pre>
      <h3>Context Tokens</h3>
      <p>Rows with <strong>*</strong> are selected activation positions.</p>
      <table>
        <thead><tr><th>idx</th><th>token_id</th><th>sel</th><th>decoded token</th></tr></thead>
        <tbody>
          {context_rows}
        </tbody>
      </table>
    </section>
    """

    return "\n".join(console), card


def format_context_window_text(
    tokenizer: AutoTokenizer,
    context_ids: list[int],
    center_idx: int,
    window: int,
) -> list[str]:
    start = max(0, center_idx - window)
    end = min(len(context_ids), center_idx + window + 1)

    lines: list[str] = [f"    Window [{start}:{end}] around context index {center_idx}:"]
    for i in range(start, end):
        marker = "<ACT>" if i == center_idx else "     "
        tok = decode_token(tokenizer, context_ids[i])
        lines.append(f"      {marker} idx={i:4d} id={context_ids[i]:7d} tok={tok}")
    return lines


def build_text_report(
    dp: TrainingDataPoint,
    tokenizer: AutoTokenizer,
    idx: int,
    context_window: int,
) -> str:
    first_target_idx = next((i for i, x in enumerate(dp.labels) if x != -100), len(dp.labels))
    prompt_ids = dp.input_ids[:first_target_idx]
    target_ids = dp.input_ids[first_target_idx:]

    prompt_text = decode_tokens(tokenizer, prompt_ids)
    target_text = decode_tokens(tokenizer, target_ids)

    context_ids = dp.context_input_ids
    context_positions = dp.context_positions

    assert context_ids is not None
    assert context_positions is not None

    direction = "unknown"
    if "previous" in prompt_text:
        direction = "past"
    if "next" in prompt_text and "generated" in prompt_text:
        direction = "future_generated"

    lines: list[str] = []
    lines.append("=" * 120)
    lines.append(f"Sample {idx}")
    lines.append("=" * 120)
    lines.append(f"direction: {direction}")
    lines.append(f"datapoint_type: {dp.datapoint_type}")
    lines.append(f"layers: {dp.layers}")
    lines.append(f"num_positions: {len(context_positions)}")
    lines.append(f"input_len: {len(dp.input_ids)}")
    lines.append(f"prompt_len: {len(prompt_ids)}")
    lines.append(f"target_len: {len(target_ids)}")
    lines.append(f"selected_context_positions: {context_positions}")
    lines.append("")
    lines.append("--- Prompt Text (Decoded) ---")
    lines.append(prompt_text)
    lines.append("")
    lines.append("--- Target Text (Decoded) ---")
    lines.append(target_text)
    lines.append("")
    lines.append("--- Context Prompt (Decoded) ---")
    lines.append(tokenizer.decode(context_ids, skip_special_tokens=False))
    lines.append("")
    lines.append("--- Selected Activation Positions In Context ---")
    lines.append(str(context_positions))

    for pos in context_positions:
        assert 0 <= pos < len(context_ids), (
            f"Datapoint {idx}: context position {pos} out of range for context length {len(context_ids)}"
        )
        tok = decode_token(tokenizer, context_ids[pos])
        lines.append(f"  context_idx={pos:4d} token_id={context_ids[pos]:7d} token={tok}")
        lines.extend(format_context_window_text(tokenizer, context_ids, pos, context_window))

    return "\n".join(lines)


def build_html(model_name: str, dataset_path: Path, sample_cards: list[str]) -> str:
    joined = "\n".join(sample_cards)
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Past Lens On-Policy Inspection</title>
  <style>
    body {{ font-family: ui-monospace, Menlo, Consolas, monospace; margin: 24px; background: #f8f9fb; color: #111; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ margin-bottom: 24px; color: #333; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    pre {{ background: #f3f4f6; padding: 10px; overflow-x: auto; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; text-align: left; padding: 4px 6px; vertical-align: top; }}
    tr.selected {{ background: #fff4cc; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Past Lens On-Policy Inspection</h1>
  <div class="meta">
    <div><strong>model:</strong> {html.escape(model_name)}</div>
    <div><strong>dataset:</strong> {html.escape(str(dataset_path))}</div>
  </div>
  {joined}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and visualize past_lens on-policy dataset samples.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to a saved past_lens train .pt file.")
    parser.add_argument("--dataset-folder", type=Path, default=Path("sft_training_data/dry_run_on_policy"))
    parser.add_argument("--model-name", type=str, default=None, help="Override tokenizer model name.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-context-rows", type=int, default=180)
    parser.add_argument("--context-window", type=int, default=20)
    parser.add_argument(
        "--txt-out",
        type=Path,
        default=Path("sft_training_data/dry_run_on_policy/past_lens_inspection_latest.txt"),
    )
    parser.add_argument("--html-out", type=Path, default=None)
    parser.add_argument("--print-summaries", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path if args.dataset_path is not None else find_latest_past_lens_file(args.dataset_folder)
    assert dataset_path.exists(), f"Dataset file not found: {dataset_path}"

    saved = torch.load(dataset_path)
    data_dicts = saved["data"]
    config = saved["config"]

    datapoints = [TrainingDataPoint(**d) for d in data_dicts]
    assert len(datapoints) > 0, "Dataset is empty"

    model_name = args.model_name if args.model_name is not None else config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n = min(args.num_samples, len(datapoints))
    random.seed(args.seed)
    chosen_indices = sorted(random.sample(range(len(datapoints)), n))

    console_chunks = []
    text_chunks = []
    cards = []
    for idx in chosen_indices:
        console_text, card = summarize_datapoint(datapoints[idx], tokenizer, idx, args.max_context_rows)
        console_chunks.append(console_text)
        text_chunks.append(build_text_report(datapoints[idx], tokenizer, idx, args.context_window))
        cards.append(card)

    if args.print_summaries:
        print("\n\n".join(console_chunks))

    args.txt_out.parent.mkdir(parents=True, exist_ok=True)
    args.txt_out.write_text("\n\n".join(text_chunks))
    print(f"Wrote TXT report to: {args.txt_out}")

    if args.html_out is not None:
        html_doc = build_html(model_name, dataset_path, cards)
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(html_doc)
        print(f"Wrote HTML report to: {args.html_out}")


if __name__ == "__main__":
    main()
