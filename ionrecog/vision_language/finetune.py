"""Small QLoRA experiment for classical-teacher coordinate supervision.

Use an isolated CUDA environment. No existing checkpoints or raw acquisitions
are overwritten. Teacher agreement is not independently validated accuracy.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_records(dataset, split):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    location = manifest["splits"][split]
    if isinstance(location, dict):
        location = location.get("path", location.get("file"))
    path = (dataset / location).resolve()
    if dataset not in path.parents:
        raise ValueError("Split path escapes dataset")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty split: " + split)
    ids = set()
    for row in rows:
        if row["id"] in ids or row["split"] != split:
            raise ValueError("Duplicate ID or incorrect split")
        ids.add(row["id"])
        image = (dataset / row["image"]).resolve()
        if dataset not in image.parents or sha256(image) != row["image_sha256"]:
            raise ValueError("Changed/escaping training image: " + row["id"])
        if not isinstance(row["prompt"], str) or not isinstance(row["response"], str):
            raise ValueError("Each record requires prompt and response text")
        target = json.loads(row["response"])
        if not isinstance(target.get("points"), list) or target.get("uncertain") is not False:
            raise ValueError("Training targets must be selected, non-abstaining coordinate lists")
    return rows


def validate_split_separation(train, val, test):
    seen_groups, seen_raw, seen_images = set(), set(), set()
    for rows in (train, val, test):
        groups = {r["group"] for r in rows}
        raw = {r["raw_sha256"] for r in rows}
        images = {r["image_sha256"] for r in rows}
        if groups & seen_groups or raw & seen_raw or images & seen_images:
            raise ValueError("Train/validation/test share acquisition groups, raw images, or rendered crops")
        seen_groups.update(groups)
        seen_raw.update(raw)
        seen_images.update(images)


def uniform_subset(rows, count):
    if count >= len(rows):
        return rows
    if count == 1:
        return rows[:1]
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def evaluation_subset(rows, count):
    """Keep fixed density strata; uniform indices can alias a round-robin dataset."""
    if count >= len(rows):
        return rows
    groups = [[], [], []]
    for row in rows:
        size = len(row["points"])
        groups[0 if size <= 4 else 1 if size <= 12 else 2].append(row)
    quotas = [0, 0, 0]
    while sum(quotas) < count:
        for index, group in enumerate(groups):
            if quotas[index] < len(group) and sum(quotas) < count:
                quotas[index] += 1
    selected_ids = {r["id"] for group, quota in zip(groups, quotas)
                    for r in (uniform_subset(group, quota) if quota else [])}
    return [row for row in rows if row["id"] in selected_ids]


def check_space(path, reserve_gib):
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free / 1024**3
    if free < reserve_gib:
        raise ValueError("Only %.2f GiB free; require %.2f GiB reserve" % (free, reserve_gib))
    return free


def efficient_attention_setup(torch):
    """Windows torch2.5 lacks FlashAttention; materialize GQA KV for efficient SDPA.

    The process-local patch changes backend selection only, not attention math.
    Refuse silent math-SDPA fallback by using the efficient backend context.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    patch = False
    if os.name == "nt" and not torch.backends.cuda.is_flash_attention_available():
        import transformers.integrations.sdpa_attention as implementation
        implementation.use_gqa_in_sdpa = lambda *args, **kwargs: False
        patch = True
    backends = [SDPBackend.EFFICIENT_ATTENTION]
    if torch.backends.cuda.is_flash_attention_available():
        backends.insert(0, SDPBackend.FLASH_ATTENTION)
    return lambda: sdpa_kernel(backends), patch


def load_model(model_path, rank, torch):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True,
        quantization_config=quantization, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0., bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)")
    model = get_peft_model(model, adapter)
    # PEFT promotes frozen layers to fp32. Keep all frozen floating non-NF4
    # parameters in bf16 so RMSNorm does not trigger fp32 bnb computation.
    for parameter in model.parameters():
        if not parameter.requires_grad and parameter.is_floating_point() and parameter.__class__.__name__ != "Params4bit":
            parameter.data = parameter.data.to(torch.bfloat16)
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=False)
    return model, processor


def encode_record(record, processor, dataset, image_tokens, torch, include_answer=True):
    from PIL import Image
    image = Image.open(Path(dataset) / record["image"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": record["prompt"]}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Qwen3 uses 16px patches and a 2x2 merge: one image token per 32x32 pixels.
    pixels = image_tokens * 32 * 32
    kwargs = {"images": image, "return_tensors": "pt", "min_pixels": pixels, "max_pixels": pixels}
    prefix = processor(text=prompt, **kwargs)
    prefix_length = prefix["input_ids"].shape[1]
    if include_answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": record["response"]}]})
        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = processor(text=full_text, **kwargs)
        if not torch.equal(encoded["input_ids"][:, :prefix_length], prefix["input_ids"]):
            raise ValueError("Assistant boundary does not match chat-template prefix")
    else:
        encoded = prefix
    merged_tokens = int(encoded["image_grid_thw"].prod().item()) // 4
    if merged_tokens != image_tokens:
        raise ValueError("Processor produced %d visual tokens, expected %d" % (merged_tokens, image_tokens))
    return encoded, prefix_length


def to_device(encoded, torch):
    return {key: value.to(device="cuda", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
            for key, value in encoded.items()}


def answer_loss(model, encoded, prefix_length, torch):
    inputs = to_device(encoded, torch)
    total = inputs["input_ids"].shape[1]
    if not 0 < prefix_length < total:
        raise ValueError("No assistant tokens to supervise")
    positions = torch.arange(prefix_length - 1, total - 1, device="cuda")
    targets = inputs["input_ids"][:, prefix_length:]
    output = model(**inputs, logits_to_keep=positions, use_cache=False)
    if output.logits.shape[:2] != targets.shape:
        raise ValueError("Answer logits/targets are misaligned")
    return torch.nn.functional.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]), targets.reshape(-1))


def generate_responses(model, processor, rows, dataset, output, image_tokens, max_tokens, attention_context, torch):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    responses = {}
    model.eval()
    for record in rows:
        encoded, prefix = encode_record(record, processor, dataset, image_tokens, torch, include_answer=False)
        inputs = to_device(encoded, torch)
        start = time.perf_counter()
        with torch.inference_mode(), attention_context():
            generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
        torch.cuda.synchronize()
        ids = generated[0, prefix:]
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        ended = any(int(token) in eos for token in ids)
        responses[record["id"]] = {"text": processor.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "finish_reason": "stop" if ended or len(ids) < max_tokens else "length",
            "seconds": time.perf_counter() - start, "generated_tokens": len(ids)}
        write_json(output / (record["id"] + ".json"), responses[record["id"]])
        print("Generation %s: %s, %d tokens" % (record["id"], responses[record["id"]]["finish_reason"], len(ids)), flush=True)
        del inputs, generated
    write_json(output / "responses.json", responses)
    return responses


def _train(args):
    output, dataset, model_path = [Path(p).resolve() for p in (args.output, args.dataset, args.model)]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty; prior weights/results are never replaced")
    for source in (dataset, model_path):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Training output overlaps input dataset/model")
    for name in ("steps", "accumulate", "rank", "image_tokens", "eval_every", "eval_samples", "max_new_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(name + " must be positive")
    if args.lr <= 0 or not math.isfinite(args.lr):
        raise ValueError("Learning rate must be positive and finite")
    initial_free = check_space(output, args.reserve_gib)
    train_rows, val_rows, test_rows = [load_records(dataset, split) for split in ("train", "val", "test")]
    validate_split_separation(train_rows, val_rows, test_rows)
    provenance = json.loads((model_path / "source_provenance.json").read_text(encoding="utf-8"))
    for entry in provenance["files"]:
        if (model_path / entry["file"]).stat().st_size != entry["bytes"]:
            raise ValueError("Base-model file size changed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("This run requires a CUDA GPU with bf16 support; use the isolated training environment")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    attention_context, windows_gqa_patch = efficient_attention_setup(torch)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), model_provenance=provenance, dataset_manifest_sha256=sha256(dataset / "manifest.json"),
        training_code_sha256=sha256(Path(__file__)),
        windows_sdpa_gqa_materialization=windows_gqa_patch, cuda_name=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        data_counts={"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        task="classical pseudo-label distillation; not independent accuracy", status="loading")
    write_json(output / "run_config.json", config)
    write_json(output / "progress.json", {"stage": "loading_model", "completed_steps": 0})
    model, processor = load_model(model_path, args.rank, torch)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name or ".visual." in name for name, _ in trainable):
        raise ValueError("Only language attention LoRA parameters may be trained")
    write_json(output / "trainable_parameters.json", {"count": sum(p.numel() for _, p in trainable),
        "parameters": [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in trainable]})
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=.01)
    val_subset, test_subset = evaluation_subset(val_rows, args.eval_samples), evaluation_subset(test_rows, args.eval_samples)
    write_json(output / "generation_evaluation_ids.json", {"val": [r["id"] for r in val_subset], "test": [r["id"] for r in test_subset]})
    if not args.skip_generation:
        write_json(output / "progress.json", {"stage": "baseline_generation", "completed_steps": 0})
        before = generate_responses(model, processor, test_subset, dataset, output / "before_responses", args.image_tokens,
                                    args.max_new_tokens, attention_context, torch)
        from .training_evaluation import evaluate_predictions
        evaluate_predictions(test_subset, before, output / "before", dataset)

    def validation_loss():
        model.eval()
        total = 0.
        with torch.no_grad(), attention_context():
            for row in val_subset:
                encoded, prefix = encode_record(row, processor, dataset, args.image_tokens, torch)
                total += float(answer_loss(model, encoded, prefix, torch))
        return total / len(val_subset)

    write_json(output / "progress.json", {"stage": "initial_validation", "completed_steps": 0})
    initial_val = validation_loss()
    best, best_step, history = initial_val, 0, []
    model.save_pretrained(output / "best_adapter", safe_serialization=True, save_embedding_layers=False)
    processor.save_pretrained(output / "processor")
    order = list(range(len(train_rows)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    position = 0
    start = time.perf_counter()
    initial_adapter = {name: p.detach().cpu().clone() for name, p in trainable}
    try:
        for step in range(1, args.steps + 1):
            write_json(output / "progress.json", {"stage": "training", "completed_steps": step - 1})
            check_space(output, args.reserve_gib)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.accumulate):
                if position == len(order):
                    rng.shuffle(order)
                    position = 0
                row = train_rows[order[position]]
                position += 1
                encoded, prefix = encode_record(row, processor, dataset, args.image_tokens, torch)
                with attention_context():
                    loss = answer_loss(model, encoded, prefix, torch)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    (loss / args.accumulate).backward()
                losses.append(float(loss.detach()))
                del loss, encoded
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.))
            if not math.isfinite(grad_norm) or grad_norm == 0:
                raise ValueError("Nonfinite/zero adapter gradient; stopping")
            optimizer.step()
            row = {"step": step, "train_answer_loss": sum(losses) / len(losses), "gradient_norm": grad_norm,
                   "elapsed_seconds": time.perf_counter() - start,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
            if step % args.eval_every == 0 or step == args.steps:
                row["validation_answer_loss"] = validation_loss()
                if row["validation_answer_loss"] < best:
                    best, best_step = row["validation_answer_loss"], step
                    model.save_pretrained(output / "best_adapter", safe_serialization=True, save_embedding_layers=False)
            history.append(row)
            write_json(output / "training_history.json", history)
            print(json.dumps(row), flush=True)
        model.save_pretrained(output / "last_adapter", safe_serialization=True, save_embedding_layers=False)
        last_delta = sum(float((p.detach().cpu() - initial_adapter[name]).square().sum()) for name, p in trainable)**.5
        if last_delta <= 0:
            raise ValueError("Adapter parameters did not change")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        loaded = set_peft_model_state_dict(model, load_file(str(output / "best_adapter" / "adapter_model.safetensors")))
        if loaded.unexpected_keys or any("lora_" in key for key in loaded.missing_keys):
            raise ValueError("Selected adapter does not match model parameter names")
        selected_delta = sum(float((p.detach().cpu() - initial_adapter[name]).square().sum()) for name, p in trainable)**.5
        if not args.skip_generation:
            write_json(output / "progress.json", {"stage": "selected_adapter_generation", "completed_steps": args.steps})
            after = generate_responses(model, processor, test_subset, dataset, output / "after_responses", args.image_tokens,
                                       args.max_new_tokens, attention_context, torch)
            from .training_evaluation import evaluate_predictions
            evaluate_predictions(test_subset, after, output / "after", dataset)
        write_json(output / "training_summary.json", {"status": "complete", "optimizer_steps": args.steps,
            "examples_seen": args.steps * args.accumulate, "initial_validation_loss": initial_val,
            "best_validation_loss": best, "best_step": best_step, "last_adapter_delta_l2": last_delta,
            "selected_adapter_delta_l2": selected_delta,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
            "training_seconds": time.perf_counter() - start, "free_gib_before": initial_free,
            "free_gib_after": check_space(output, args.reserve_gib),
            "note": "Validation loss and localization agreement are relative to classical pseudo-labels, not true ion accuracy."})
        config["status"] = "complete"
        write_json(output / "run_config.json", config)
        write_json(output / "progress.json", {"stage": "complete", "completed_steps": args.steps})
    except BaseException as exc:
        write_json(output / "interrupted.json", {"status": "interrupted", "completed_steps": len(history),
                   "error_type": type(exc).__name__, "error": str(exc), "best_step": best_step})
        raise
    print("Training complete: " + str(output), flush=True)


def train(args):
    """Capture setup/OOM failures as well as failures after optimizer startup."""
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Training output must be empty")
    try:
        return _train(args)
    except BaseException as exc:
        if (output / "run_config.json").is_file():
            progress_file = output / "progress.json"
            progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else {}
            if not (output / "interrupted.json").exists():
                write_json(output / "interrupted.json", dict(progress, status="interrupted",
                    error_type=type(exc).__name__, error=str(exc)))
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            config["status"] = "interrupted"
            write_json(output / "run_config.json", config)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="checkpoints/Qwen3-VL-2B-Instruct-bf16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--reserve-gib", type=float, default=8.)
    parser.add_argument("--skip-generation", action="store_true", help="Smoke test only; no claim of before/after improvement")
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
