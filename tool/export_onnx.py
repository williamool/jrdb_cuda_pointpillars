#!/usr/bin/env python3
"""Export OpenPCDet JRDB PointPillar checkpoint to ONNX for CUDA-PointPillars."""

import argparse
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxsim import simplify
from pathlib import Path

# OpenPCDet on PYTHONPATH
OPENPCDET_ROOT = os.environ.get('OPENPCDET_ROOT', '/home/wbl/OpenPCDet')
sys.path.insert(0, OPENPCDET_ROOT)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from jrdb_config import MAX_VOXELS, MAX_POINTS_PER_VOXEL, ONNX_OUTPUT_SHAPES
from modify_onnx import simplify_postprocess, simplify_preprocess


class PointPillarExport(nn.Module):
    """Export backbone+head only; inputs are pre-voxelized tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxels, voxel_num, voxel_idxs):
        batch_dict = {
            'voxels': voxels,
            'voxel_num_points': voxel_num,
            'voxel_coords': voxel_idxs,
            'batch_size': 1,
        }
        for module in self.model.module_list:
            batch_dict = module(batch_dict)
        ret = self.model.dense_head.forward_ret_dict
        return ret['cls_preds'], ret['box_preds'], ret['dir_cls_preds']


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=Path('/tmp'), logger=logger
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise NotImplementedError


def parse_config():
    parser = argparse.ArgumentParser(description='Export JRDB PointPillar to ONNX')
    parser.add_argument('--cfg_file', type=str,
                        default=str(PROJECT_ROOT / 'configs/pointpillar_jrdb.yaml'))
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=str(PROJECT_ROOT / 'model'))
    parser.add_argument('--onnx_name', type=str, default='pointpillar_v2')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    os.makedirs(args.out_dir, exist_ok=True)
    logger = common_utils.create_logger()

    logger.info('Grid / feature map output shapes: %s', ONNX_OUTPUT_SHAPES)

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    export_model = PointPillarExport(model).to(device).eval()

    with torch.no_grad():
        dummy_voxels = torch.zeros(
            (MAX_VOXELS, MAX_POINTS_PER_VOXEL, 4), dtype=torch.float32, device=device)
        dummy_voxel_idxs = torch.zeros(
            (MAX_VOXELS, 4), dtype=torch.int32, device=device)
        dummy_voxel_num = torch.zeros((1,), dtype=torch.int32, device=device)

        raw_onnx = os.path.join(args.out_dir, f'{args.onnx_name}_raw.onnx')
        torch.onnx.export(
            export_model,
            (dummy_voxels, dummy_voxel_num, dummy_voxel_idxs),
            raw_onnx,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            input_names=['voxels', 'voxel_num', 'voxel_idxs'],
            output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],
        )

    onnx_trim = simplify_postprocess(onnx.load(raw_onnx))
    onnx_simp, check = simplify(onnx_trim)
    assert check, 'ONNX simplify validation failed'
    onnx_final = simplify_preprocess(onnx_simp)
    out_path = os.path.join(args.out_dir, f'{args.onnx_name}.onnx')
    onnx.save(onnx_final, out_path)
    logger.info('[PASS] Saved %s', out_path)


if __name__ == '__main__':
    main()
