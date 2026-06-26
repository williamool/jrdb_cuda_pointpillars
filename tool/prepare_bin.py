#!/usr/bin/env python3
"""Convert OpenPCDet JRDB .npy point files to KITTI-style .bin for TRT runtime."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/wbl/OpenPCDet/data/jrdb')
    parser.add_argument('--sample_ids', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in args.sample_ids:
        pts = np.load(data_root / 'points' / f'{sid}.npy').astype(np.float32)
        out_path = out_dir / f'{sid}.bin'
        pts.tofile(out_path)
        print(f'  {sid}: {pts.shape[0]} points -> {out_path}')


if __name__ == '__main__':
    main()
