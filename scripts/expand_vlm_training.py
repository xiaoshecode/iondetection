"""Expand training crops without modifying the frozen validation/test sets."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from ionrecog.vision_language.training_expansion import expand_training_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',default=str(PROJECT/'runs/vlm_train_20261103/dataset_v2'))
    parser.add_argument('--data',default=str(PROJECT/'data'))
    parser.add_argument('--output',default=str(PROJECT/'runs/vlm_expand_20261103/dataset_4096'))
    parser.add_argument('--train-crops',type=int,default=4096)
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--reserve-gib',type=float,default=8.)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--candidates-per-frame',type=int,default=48)
    args=parser.parse_args()
    manifest=expand_training_dataset(args.parent,args.data,args.output,args.train_crops,args.seed,args.reserve_gib,args.workers,args.candidates_per_frame)
    print('Expanded:',manifest['counts'],manifest['source_frame_counts'],flush=True)


if __name__=='__main__':
    main()
