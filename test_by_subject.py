from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import FaceGazeDataset, get_val_transform
from models import ResNetFAMLite, ResNetFAMLiteConfig
from models.clip_gaze import CLIPGazeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet18-FAM-Lite 测试脚本（按Subject划分）")
    parser.add_argument("--data_root", type=str, default=".", help="数据根目录（包含Subject*_*_data文件夹的路径，支持绝对路径或相对路径）")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径（训练后会在checkpoints/best.pth保存最佳模型）")
    parser.add_argument("--test_indices_file", type=str, default="test_indices_by_subject.json", help="测试集索引文件路径（优先使用）")
    parser.add_argument("--test_subjects", type=str, default="", help="测试集subject编号范围，如'26-28'（如果未提供test_indices_file则使用此选项）")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="测试阶段启用AMP")
    # CLIP-Gaze配置（应与训练时保持一致）
    parser.add_argument("--use_clip_gaze", action="store_true", default=True, help="启用CLIP-Gaze（应与训练时配置一致）")
    return parser.parse_args()


def parse_subject_range(subject_str: str) -> list[int]:
    """解析subject范围字符串，如'1-22' -> [1,2,...,22]"""
    parts = subject_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"无效的subject范围格式: {subject_str}，应为'start-end'格式")
    start, end = int(parts[0]), int(parts[1])
    return list(range(start, end + 1))


def extract_subject_id_from_path(path: Path) -> int:
    """从路径中提取subject编号，如Subject01_1_data/face_ims/xxx.png -> 1"""
    # 查找Subject开头的文件夹名
    parts = path.parts
    for part in parts:
        match = re.match(r"Subject(\d+)_\d+_data", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"无法从路径 {path} 中提取subject编号")


def get_indices_by_subjects(dataset: FaceGazeDataset, subject_ids: list[int]) -> list[int]:
    """根据subject编号列表获取对应的样本索引"""
    indices = []
    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        try:
            subject_id = extract_subject_id_from_path(sample.image_path)
            if subject_id in subject_ids:
                indices.append(idx)
        except ValueError:
            continue
    return indices


@torch.no_grad()
def test_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """测试一个epoch，返回L1误差和角度误差"""
    import torch.nn.functional as F
    
    model.eval()
    total_l1, total_ang = 0.0, 0.0
    
    def angular_loss_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
        """计算 gaze 角度偏差 (deg)，内部转换为弧度求余弦相似度。"""
        pred = torch.deg2rad(pred_deg)
        target = torch.deg2rad(target_deg)
        pred_vec = torch.stack(
                     [
                               -torch.sin(pred[:, 0]) * torch.cos(pred[:, 1]),   # x
                               -torch.sin(pred[:, 1]),                           # y
                               -torch.cos(pred[:, 0]) * torch.cos(pred[:, 1]),   # z
                     ],
                     dim=1,
           )
        target_vec = torch.stack(
                     [
                               -torch.sin(target[:, 0]) * torch.cos(target[:, 1]),  # x
                               -torch.sin(target[:, 1]),                            # y
                               -torch.cos(target[:, 0]) * torch.cos(target[:, 1]),  # z
                     ],
                     dim=1,
           )
        cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.acos(cos_sim)
        return torch.rad2deg(angle).mean()
    
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["gaze_angles_deg"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            gaze_pred = outputs["gaze"]
        total_l1 += F.l1_loss(gaze_pred, targets, reduction="sum").item()
        total_ang += angular_loss_deg(gaze_pred, targets).item() * images.size(0)
    num_samples = len(loader.dataset)
    return total_l1 / num_samples, total_ang / num_samples


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 加载完整数据集
    val_transform = get_val_transform(image_size=(224, 224))
    full_dataset = FaceGazeDataset(
        root_dir=args.data_root,
        transform=val_transform,
        image_size=(224, 224),
    )
    
    # 确定测试集索引
    test_indices_path = Path(args.test_indices_file)
    if test_indices_path.exists():
        # 优先使用保存的测试集索引文件
        with test_indices_path.open("r", encoding="utf-8") as f:
            test_indices = json.load(f)
        print(f"从文件加载测试集索引: {test_indices_path} (共 {len(test_indices)} 个样本)")
    elif args.test_subjects:
        # 如果没有索引文件，则根据subject编号范围获取
        test_subject_ids = parse_subject_range(args.test_subjects)
        test_indices = get_indices_by_subjects(full_dataset, test_subject_ids)
        print(f"根据subject编号范围 {args.test_subjects} 加载测试集: {len(test_indices)} 个样本")
    else:
        raise ValueError(
            f"未找到测试集索引文件: {test_indices_path}，且未提供 --test_subjects 参数。\n"
            f"请先运行 train_by_subject.py 生成测试集索引文件，或使用 --test_subjects 参数指定subject编号范围。"
        )
    
    if len(test_indices) == 0:
        raise ValueError("测试集为空！请检查测试集配置。")
    
    # 创建测试集子集
    test_dataset = Subset(full_dataset, test_indices)
    print(f"测试集大小: {len(test_dataset)} 个样本")
    
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )

    # 配置模型（应与训练时保持一致）
    clip_gaze_config = None
    if args.use_clip_gaze:
        clip_gaze_config = CLIPGazeConfig()
    
    cfg = ResNetFAMLiteConfig(
        use_clip_gaze=args.use_clip_gaze,
        clip_gaze_config=clip_gaze_config,
    )
    model = ResNetFAMLite(cfg).to(device)
    state_path = Path(args.checkpoint)
    if not state_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"已加载模型权重: {state_path}")

    test_l1, test_ang = test_epoch(model, loader, device, use_amp)
    print(f"\n测试结果:")
    print(f"  Test MAE (deg): {test_l1:.4f}")
    print(f"  Test Angular Error (deg): {test_ang:.4f}")


if __name__ == "__main__":
    main()

