"""Run a saved ion-coordinate LoRA adapter on a prepared, held-out crop split."""
import argparse
import json
import os
from pathlib import Path

from .finetune import (check_space, efficient_attention_setup, evaluation_subset, generate_responses,
                       load_model, load_records, sha256, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='checkpoints/Qwen3-VL-2B-Instruct-bf16')
    parser.add_argument('--adapter', help='Saved best_adapter directory; omit for the original base model')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates every crop')
    parser.add_argument('--image-tokens', type=int, default=1024)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--reserve-gib', type=float, default=8.)
    args = parser.parse_args(argv)
    output, dataset, base = (Path(p).resolve() for p in (args.output, args.dataset, args.model))
    if output.exists():
        raise ValueError('Output already exists; choose a new result directory')
    sources = [dataset, base] + ([Path(args.adapter).resolve()] if args.adapter else [])
    if any(output == source or source in output.parents or output in source.parents for source in sources):
        raise ValueError('Output must not overlap source data or model')
    if args.limit < 0 or args.image_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError('Invalid generation size/limit')
    check_space(output, args.reserve_gib)
    rows = load_records(dataset, args.split)
    if args.limit:
        rows = evaluation_subset(rows, args.limit)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Requires the isolated CUDA training environment')
    torch.set_num_threads(4)
    rank = 4
    if args.adapter:
        configuration = json.loads((Path(args.adapter) / 'adapter_config.json').read_text(encoding='utf-8'))
        rank = configuration['r']
        if configuration['lora_alpha'] != 2 * rank:
            raise ValueError('Adapter scale differs from this experiment; do not silently change its output')
    context, patched = efficient_attention_setup(torch)
    model, processor = load_model(base, rank, torch)
    adapter_hash = None
    if args.adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        weights = Path(args.adapter) / 'adapter_model.safetensors'
        adapter_hash = sha256(weights)
        result = set_peft_model_state_dict(model, load_file(str(weights)))
        if result.unexpected_keys or any('lora_' in key for key in result.missing_keys):
            raise ValueError('Adapter state does not match the model')
    output.mkdir(parents=True)
    write_json(output / 'evaluation_config.json', dict(vars(args), adapter_sha256=adapter_hash,
        dataset_manifest_sha256=sha256(dataset / 'manifest.json'), ids=[r['id'] for r in rows],
        windows_sdpa_gqa_materialization=patched))
    replies = generate_responses(model, processor, rows, dataset, output / 'responses',
        args.image_tokens, args.max_new_tokens, context, torch)
    from .training_evaluation import evaluate_predictions
    report = evaluate_predictions(rows, replies, output / 'report', dataset)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
