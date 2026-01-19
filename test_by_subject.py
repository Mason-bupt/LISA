from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import FaceGazeDataset, get_val_transform
from models import ResNetFAMLite, ResNetFAMLiteConfig
from models.sdm import SDMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet18-FAM-Lite Test Script")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_indices_file", type=str, default="test_indices_by_subject.json")
    parser.add_argument("--test_subjects", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_sdm", action="store_true", default=True)
    return parser.parse_args()


def parse_subject_range(subject_str: str) -> list[int]:
    parts = subject_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid subject range format: {subject_str}")
    start, end = int(parts[0]), int(parts[1])
    return list(range(start, end + 1))


def extract_subject_id_from_path(path: Path) -> int:
    parts = path.parts
    for part in parts:
        match = re.match(r"Subject(\d+)_\d+_data", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot extract subject ID from path: {path}")


def get_indices_by_subjects(dataset: FaceGazeDataset, subject_ids: list[int]) -> list[int]:
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
    import torch.nn.functional as F
    
    model.eval()
    total_l1, total_ang = 0.0, 0.0
    
    def angular_loss_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
        pred = torch.deg2rad(pred_deg)
        target = torch.deg2rad(target_deg)
        pred_vec = torch.stack(
                     [
                               -torch.sin(pred[:, 0]) * torch.cos(pred[:, 1]),
                               -torch.sin(pred[:, 1]),
                               -torch.cos(pred[:, 0]) * torch.cos(pred[:, 1]),
                     ],
                     dim=1,
           )
        target_vec = torch.stack(
                     [
                               -torch.sin(target[:, 0]) * torch.cos(target[:, 1]),
                               -torch.sin(target[:, 1]),
                               -torch.cos(target[:, 0]) * torch.cos(target[:, 1]),
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

    val_transform = get_val_transform(image_size=(224, 224))
    full_dataset = FaceGazeDataset(
        root_dir=args.data_root,
        transform=val_transform,
        image_size=(224, 224),
    )
    
    test_indices_path = Path(args.test_indices_file)
    if test_indices_path.exists():
        with test_indices_path.open("r", encoding="utf-8") as f:
            test_indices = json.load(f)
        print(f"Loaded test indices from file: {test_indices_path} ({len(test_indices)} samples)")
    elif args.test_subjects:
        test_subject_ids = parse_subject_range(args.test_subjects)
        test_indices = get_indices_by_subjects(full_dataset, test_subject_ids)
        print(f"Loaded test set by subject range {args.test_subjects}: {len(test_indices)} samples")
    else:
        raise ValueError(
            f"Test indices file not found: {test_indices_path}, and --test_subjects not provided.\n"
            f"Please run train_by_subject.py to generate test indices file, or use --test_subjects to specify subject range."
        )
    
    if len(test_indices) == 0:
        raise ValueError("Test set is empty! Check test set configuration.")
    
    test_dataset = Subset(full_dataset, test_indices)
    print(f"Test set size: {len(test_dataset)} samples")
    
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )

    sdm_config = None
    if args.use_sdm:
        sdm_config = SDMConfig()
    
    cfg = ResNetFAMLiteConfig(
        use_sdm=args.use_sdm,
        sdm_config=sdm_config,
    )
    model = ResNetFAMLite(cfg).to(device)
    state_path = Path(args.checkpoint)
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"Loaded model weights: {state_path}")

    test_l1, test_ang = test_epoch(model, loader, device, use_amp)
    print(f"\nTest Results:")
    print(f"  Test MAE (deg): {test_l1:.4f}")
    print(f"  Test Angular Error (deg): {test_ang:.4f}")


if __name__ == "__main__":
    main()
