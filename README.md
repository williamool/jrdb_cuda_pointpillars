# JRDB PointPillar v2 → CUDA-PointPillars (TensorRT)

将 OpenPCDet 训练的 **PointPillar v2 / jrdb_v2_bs8** 导出为 ONNX / TensorRT，在 NVIDIA CUDA-PointPillars 流水线上加速推理。

## 目录

```
jrdb_cuda_pointpillars/
├── jrdb_config.py                 # v2 几何 / anchor 常量
├── configs/pointpillar_jrdb_v2.yaml
├── model/
│   ├── pointpillar_v2.onnx        # 导出产物
│   ├── pointpillar_v2_raw.onnx
│   └── pointpillar_v2.plan        # TensorRT engine
├── tool/
│   ├── export_onnx.py
│   ├── modify_onnx.py
│   └── prepare_bin.py             # JRDB .npy → .bin
├── data/sample_bins/              # 预置测试帧 (.bin, Jetson 可直接用)
├── runtime_patches/               # JRDB v2 C++ 补丁
└── scripts/
    ├── 1_export_onnx.sh
    ├── 2_setup_runtime.sh
    ├── 3_build_trt_engine.sh
    ├── 4_build_runtime.sh
    └── 5_infer_sample.sh
```

## v2 模型参数

| 参数 | 值 |
|------|-----|
| Checkpoint | `.../pointpillar_v2/jrdb_v2_bs8/ckpt/checkpoint_epoch_100.pth` |
| VOXEL_SIZE | 0.128 m |
| BEV grid | 400 × 400 |
| Feature map | 200 × 200 |
| Anchors | 4 朝向 (0°, 45°, 90°, 135°) |
| ONNX 输出 | cls(4) / box(28) / dir(16) |
| MAX_VOXELS | 50000 |

## 快速开始

```bash
cd /home/wbl/jrdb_cuda_pointpillars

# Step 1: PTH → ONNX
bash scripts/1_export_onnx.sh

# Step 2: 准备 C++ runtime（复制 CUDA-PointPillars + 打补丁）
bash scripts/2_setup_runtime.sh

# Step 3: 编译（需 CUDA + TensorRT，x86 请改 runtime/tool/environment.sh）
bash scripts/4_build_runtime.sh

# Step 4: ONNX → TensorRT engine
bash scripts/3_build_trt_engine.sh

# Step 5: 单帧推理测试
bash scripts/5_infer_sample.sh 024653
```

## Jetson 板端测试

仓库已包含 **10 帧** JRDB 测试点云（`data/sample_bins/`，约 2.5 MB），无需拷贝完整数据集或 OpenPCDet：

| 帧 ID | 场景 | 点数 |
|-------|------|------|
| 024653–024657 | 室内 | ~14.4k |
| 026500–026504 | 室外密集 | ~17.8k |

```bash
git clone git@github.com:williamool/jrdb_cuda_pointpillars.git
cd jrdb_cuda_pointpillars

bash scripts/2_setup_runtime.sh
bash scripts/4_build_runtime.sh      # Jetson: source runtime/tool/environment.sh
bash scripts/3_build_trt_engine.sh   # .plan 必须在 Jetson 上生成

# 直接用预置 .bin 测速（无需 OpenPCDet）
bash scripts/5_infer_sample.sh 024653
```

## 推理流水线

```
.bin 点云 (N×4 float32)
  → GPU Voxelization (10 通道 pillar)
  → TensorRT FP16 (pointpillar_v2.plan)
  → GPU Decode + NMS (4 anchors, 4 dir bins)
  → .txt 检测框
```

## 与 OpenPCDet 的差异

- 体素化在 **CUDA** 上完成，与 DataLoader 有微小数值差异
- 后处理在 **CUDA** 中实现，参数与 `pointpillar_v2.yaml` 对齐
- 仅支持 Anchor PointPillar（不支持 CenterPoint）

## 依赖

- OpenPCDet: `/home/wbl/OpenPCDet`（pcdet 可 import）
- Python: `onnx`, `onnxsim`, `onnx-graphsurgeon`
- C++: CUDA, TensorRT, CMake
- 上游: [NVIDIA CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)
