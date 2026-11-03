"""Create a small split-safe VLM pseudo-label dataset from raw acquisitions."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from ionrecog.cli import load_config
from ionrecog.vision_language.training_data import prepare_training_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default=str(PROJECT/'data'))
    parser.add_argument('--output',default=str(PROJECT/'runs/vlm_train_20261103/dataset'))
    parser.add_argument('--config',default=str(PROJECT/'configs/existing_crystal.yaml'))
    parser.add_argument('--train-crops',type=int,default=256)
    parser.add_argument('--val-crops',type=int,default=32)
    parser.add_argument('--test-crops',type=int,default=24)
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--reserve-gib',type=float,default=8.)
    parser.add_argument('--edge-margin',type=float,default=1.5)
    args=parser.parse_args()
    if min(args.train_crops,args.val_crops,args.test_crops)<1:
        parser.error('Crop quotas must be positive')
    try:
        manifest=prepare_training_dataset(args.data,args.output,load_config(args.config),
            {'train':args.train_crops,'val':args.val_crops,'test':args.test_crops},args.seed,reserve_gib=args.reserve_gib,
            edge_margin=args.edge_margin)
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2,'Error: %s\n'%exc)
    print('Prepared:',args.output,manifest['counts'])
    print('All labels are classical pseudo-labels, not independent accuracy truth.')


if __name__=='__main__':
    main()
