from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from dataset import FaceGazeDataset, get_train_transform, get_val_transform
from models import ResNetFAMLite, ResNetFAMLiteConfig
from models.sdm import (
           SDMConfig,
           feature_separation_loss,
)


def parse_args() -> argparse.Namespace:
           parser = argparse.ArgumentParser(description="ResNet18-FAM-Lite Gaze Training")
           parser.add_argument("--data_root", type=str, default=".")
           parser.add_argument("--batch_size", type=int, default=64)
           parser.add_argument("--epochs", type=int, default=50)
           parser.add_argument("--lr", type=float, default=1e-4)
           parser.add_argument("--weight_decay", type=float, default=5e-4)
           parser.add_argument("--num_workers", type=int, default=16)
           parser.add_argument("--pin_memory", action="store_true", default=True)
           parser.add_argument("--persistent_workers", action="store_true", default=True)
           parser.add_argument("--device", type=str, default="cuda")
           parser.add_argument("--log_interval", type=int, default=10)
           parser.add_argument("--save_dir", type=str, default="checkpoints")
           parser.add_argument("--resume", type=str, default="")
           parser.add_argument("--train_subjects", type=str, default="1-22")
           parser.add_argument("--val_subjects", type=str, default="23-25")
           parser.add_argument("--test_subjects", type=str, default="26-28")
           parser.add_argument("--test_indices_file", type=str, default="test_indices_by_subject.json")
           parser.add_argument("--amp", action="store_true")
           parser.add_argument("--compile", action="store_true")
           parser.add_argument("--use_sdm", action="store_true", default=True)
           parser.add_argument("--no-use-sdm", dest="use_sdm", action="store_false")
           parser.add_argument("--sdm_separation_weight", type=float, default=100.0)
           parser.add_argument("--max_grad_norm", type=float, default=1.0)
           parser.add_argument("--log_dir", type=str, default="logs")
           parser.add_argument("--lr_scheduler", type=str, choices=["plateau", "cosine", "step", "none"],       
                                                            default="plateau")
           parser.add_argument("--lr_factor", type=float, default=0.5)
           parser.add_argument("--lr_patience", type=int, default=5)
           parser.add_argument("--lr_min", type=float, default=1e-6)
           parser.add_argument("--early_stop", action="store_true", default=True)
           parser.add_argument("--no-early-stop", dest="early_stop", action="store_false")
           parser.add_argument("--early_stop_patience", type=int, default=10)
           parser.add_argument("--early_stop_min_delta", type=float, default=0.001)
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


def angular_loss_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
           if torch.isnan(pred_deg).any() or torch.isinf(pred_deg).any():
                     return torch.tensor(180.0, device=pred_deg.device)
           if torch.isnan(target_deg).any() or torch.isinf(target_deg).any():
                     return torch.tensor(180.0, device=target_deg.device)
                 
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
                 
           if torch.isnan(angle).any() or torch.isinf(angle).any():
                     return torch.tensor(180.0, device=pred_deg.device)
                 
           result = torch.rad2deg(angle).mean()
                 
           if torch.isnan(result) or torch.isinf(result):
                     return torch.tensor(180.0, device=pred_deg.device)
                 
           return result


def train_one_epoch(
           model: nn.Module,
           loader: DataLoader,
           optimizer: optim.Optimizer,
           device: torch.device,
           epoch: int,
           log_interval: int,
           use_sdm: bool = True,
           sdm_separation_weight: float = 0.5,
           scaler: Optional[torch.cuda.amp.GradScaler] = None,
           max_grad_norm: float = 1.0,
) -> float:
           model.train()
           total_loss = 0.0
           for step, batch in enumerate(loader, 1):
                     images = batch["image"].to(device, non_blocking=True)
                     targets = batch["gaze_angles_deg"].to(device, non_blocking=True)

                     with torch.cuda.amp.autocast(enabled=scaler is not None):
                               outputs = model(images)
                               gaze_pred = outputs["gaze"]
                                     
                               loss_l1 = F.l1_loss(gaze_pred, targets)
                               loss_ang = angular_loss_deg(gaze_pred, targets)
                               loss = loss_l1 + 0.5 * loss_ang
                                     
                               loss_separation = None
                               if use_sdm and "gaze_features_proj" in outputs:
                                         gaze_features_proj = outputs["gaze_features_proj"]
                                         text_features = outputs["text_features"]
                                               
                                         loss_separation = feature_separation_loss(
                                                   gaze_features_proj, text_features
                                         )
                                         loss = loss + sdm_separation_weight * loss_separation

                     if torch.isnan(loss) or torch.isinf(loss):
                               print(f"Warning: Epoch {epoch} Step {step} NaN/Inf loss detected, skipping")
                               continue
                           
                     optimizer.zero_grad(set_to_none=True)
                     if scaler is not None:
                               scaler.scale(loss).backward()
                               scaler.unscale_(optimizer)
                               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                               scaler.step(optimizer)
                               scaler.update()
                     else:
                               loss.backward()
                               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                               optimizer.step()

                     total_loss += loss.item()
                     if step % log_interval == 0:
                               loss_str = f"Loss {loss.item():.4f}"
                               if loss_separation is not None:
                                         loss_str += f" (L1: {loss_l1.item():.4f}, Ang: {loss_ang.item():.4f}, Sep: {loss_separation.item():.4f})"
                               else:
                                         loss_str += f" (L1: {loss_l1.item():.4f}, Ang: {loss_ang.item():.4f})"
                               print(f"Epoch {epoch} Step {step}/{len(loader)} {loss_str}")

           return total_loss / len(loader)


@torch.no_grad()
def test_epoch(
           model: nn.Module,
           loader: DataLoader,
           device: torch.device,
           use_amp: bool,
) -> Tuple[float, float]:
           model.eval()
           total_l1, total_ang = 0.0, 0.0
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


def save_checkpoint(
           model: nn.Module,       
           save_dir: Path,       
           epoch: int,       
           is_best: bool = False,
           best_checkpoint_path: Optional[Path] = None,
) -> Optional[Path]:
           if not is_best:
                     return None
                 
           save_dir.mkdir(parents=True, exist_ok=True)
           ckpt_path = save_dir / "best.pth"
                 
           if best_checkpoint_path is not None and best_checkpoint_path.exists() and best_checkpoint_path != ckpt_path:
                     best_checkpoint_path.unlink()
                     print(f"Deleted old best checkpoint: {best_checkpoint_path}")
                 
           torch.save(model.state_dict(), ckpt_path)
           print(f"Best checkpoint saved: {ckpt_path}")
           return ckpt_path


class TrainingLogger:
                 
           def __init__(self, log_dir: Path):
                     self.log_dir = Path(log_dir)
                     self.log_dir.mkdir(parents=True, exist_ok=True)
                           
                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                     self.log_file = self.log_dir / f"training_log_by_subject_{timestamp}.txt"
                     self.json_file = self.log_dir / f"training_history_by_subject_{timestamp}.json"
                           
                     self.history: Dict[str, list] = {
                               "epoch": [],
                               "train_loss": [],
                               "val_l1": [],
                               "val_ang": [],
                     }
                           
                     self.config: Dict[str, Any] = {}
                           
                     with self.log_file.open("w", encoding="utf-8") as f:
                               f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                               f.write("=" * 80 + "\n\n")
                 
           def log_config(self, args: argparse.Namespace, dataset_info: Dict[str, Any]):
                     self.config = {
                               "data_root": str(args.data_root),
                               "batch_size": args.batch_size,
                               "epochs": args.epochs,
                               "lr": args.lr,
                               "weight_decay": args.weight_decay,
                               "train_subjects": args.train_subjects,
                               "val_subjects": args.val_subjects,
                               "test_subjects": args.test_subjects,
                               "device": args.device,
                               "use_sdm": args.use_sdm,
                               "sdm_separation_weight": args.sdm_separation_weight,
                               "max_grad_norm": args.max_grad_norm,
                               "amp": args.amp,
                               "compile": args.compile,
                               "dataset_info": dataset_info,
                     }
                           
                     with self.log_file.open("a", encoding="utf-8") as f:
                               f.write("Training Configuration:\n")
                               f.write("-" * 80 + "\n")
                               for key, value in self.config.items():
                                         if key != "dataset_info":
                                                   f.write(f"      {key}: {value}\n")
                               f.write(f"\nDataset Information:\n")
                               for key, value in dataset_info.items():
                                         f.write(f"      {key}: {value}\n")
                               f.write("\n" + "=" * 80 + "\n\n")
                               f.write("Training Process:\n")
                               f.write("-" * 80 + "\n")
                 
           def log_epoch(self, epoch: int, train_loss: float, val_l1: float, val_ang: float, is_best: bool = False):
                     self.history["epoch"].append(epoch)
                     self.history["train_loss"].append(float(train_loss))
                     self.history["val_l1"].append(float(val_l1))
                     self.history["val_ang"].append(float(val_ang))
                           
                     with self.log_file.open("a", encoding="utf-8") as f:
                               best_marker = " [BEST]" if is_best else ""
                               f.write(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_l1={val_l1:.6f}, val_ang={val_ang:.6f}{best_marker}\n")
                           
                     with self.json_file.open("w", encoding="utf-8") as f:
                               json.dump({
                                         "config": self.config,
                                         "history": self.history,
                               }, f, indent=2, ensure_ascii=False)
                 
           def log_final(self, best_epoch: int, best_val_ang: float, checkpoint_path: Optional[Path]):
                     with self.log_file.open("a", encoding="utf-8") as f:
                               f.write("\n" + "=" * 80 + "\n")
                               f.write("Training Complete\n")
                               f.write("-" * 80 + "\n")
                               f.write(f"Best Model: Epoch {best_epoch}\n")
                               f.write(f"Best Validation Angular Error: {best_val_ang:.6f}°\n")
                               if checkpoint_path:
                                         f.write(f"Checkpoint Path: {checkpoint_path}\n")
                               f.write(f"Log File: {self.log_file}\n")
                               f.write(f"History File: {self.json_file}\n")
                               f.write("=" * 80 + "\n")


def main() -> None:
           args = parse_args()
           device = torch.device(args.device if torch.cuda.is_available() else "cpu")
           if device.type == "cuda":
                     torch.backends.cudnn.benchmark = True

           use_amp = args.amp and device.type == "cuda"
           scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

           train_subject_ids = parse_subject_range(args.train_subjects)
           val_subject_ids = parse_subject_range(args.val_subjects)
           test_subject_ids = parse_subject_range(args.test_subjects)
                 
           overlap_train_val = set(train_subject_ids) & set(val_subject_ids)
           overlap_train_test = set(train_subject_ids) & set(test_subject_ids)
           overlap_val_test = set(val_subject_ids) & set(test_subject_ids)
                 
           if overlap_train_val:
                     raise ValueError(f"Training and validation subject IDs overlap: {overlap_train_val}")
           if overlap_train_test:
                     raise ValueError(f"Training and test subject IDs overlap: {overlap_train_test}")
           if overlap_val_test:
                     print(f"Warning: Validation and test subject IDs overlap: {overlap_val_test}")
                 
           print(f"Data Split Configuration:")
           print(f"      Training: subject {args.train_subjects} ({len(train_subject_ids)} subjects)")
           print(f"      Validation: subject {args.val_subjects} ({len(val_subject_ids)} subjects)")
           print(f"      Test: subject {args.test_subjects} ({len(test_subject_ids)} subjects)")

           full_dataset = FaceGazeDataset(root_dir=args.data_root)
                 
           print("\nSplitting dataset by subject IDs...")
           train_indices = get_indices_by_subjects(full_dataset, train_subject_ids)
           val_indices = get_indices_by_subjects(full_dataset, val_subject_ids)
           test_indices = get_indices_by_subjects(full_dataset, test_subject_ids)
                 
           if len(train_indices) == 0:
                     raise ValueError(f"Training set is empty! Check subject range {args.train_subjects}")
           if len(val_indices) == 0:
                     raise ValueError(f"Validation set is empty! Check subject range {args.val_subjects}")
           if len(test_indices) == 0:
                     raise ValueError(f"Test set is empty! Check subject range {args.test_subjects}")
                 
           test_indices_path = Path(args.test_indices_file)
           with test_indices_path.open("w", encoding="utf-8") as f:
                     json.dump(test_indices, f)
           print(f"\nTest indices saved to: {test_indices_path} ({len(test_indices)} samples)")
           print(f"Data split statistics: Training={len(train_indices)} samples, Validation={len(val_indices)} samples, Test={len(test_indices)} samples")
                 
           train_transform = get_train_transform(image_size=(224, 224))
           val_transform = get_val_transform(image_size=(224, 224))
                 
           train_dataset = FaceGazeDataset(
                     root_dir=args.data_root,
                     transform=train_transform,
                     image_size=(224, 224),
           )
           val_dataset = FaceGazeDataset(
                     root_dir=args.data_root,
                     transform=val_transform,
                     image_size=(224, 224),
           )
                 
           train_set = Subset(train_dataset, train_indices)
           val_set = Subset(val_dataset, val_indices)

           train_loader = DataLoader(
                     train_set,
                     batch_size=args.batch_size,
                     shuffle=True,
                     num_workers=args.num_workers,
                     pin_memory=args.pin_memory,
                     persistent_workers=args.persistent_workers and args.num_workers > 0,
                     drop_last=False,
           )
           val_loader = DataLoader(
                     val_set,
                     batch_size=args.batch_size,
                     shuffle=True,
                     num_workers=args.num_workers,
                     pin_memory=args.pin_memory,
                     persistent_workers=args.persistent_workers and args.num_workers > 0,
           )

           sdm_config = None
           if args.use_sdm:
                     sdm_config = SDMConfig(
                               feature_separation_weight=args.sdm_separation_weight,
                     )
                 
           cfg = ResNetFAMLiteConfig(
                     use_sdm=args.use_sdm,
                     sdm_config=sdm_config,
           )
           model = ResNetFAMLite(cfg).to(device)
           if args.compile and hasattr(torch, "compile"):
                     model = torch.compile(model)
           if args.resume:
                     state = torch.load(args.resume, map_location="cpu")
                     model.load_state_dict(state, strict=False)
                     print(f"Loaded checkpoint from {args.resume}")

           optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                 
           scheduler = None
           if args.lr_scheduler == "plateau":
                     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                               optimizer,
                               mode='min',
                               factor=args.lr_factor,
                               patience=args.lr_patience,
                               min_lr=args.lr_min,
                     )
           elif args.lr_scheduler == "cosine":
                     scheduler = optim.lr_scheduler.CosineAnnealingLR(
                               optimizer,
                               T_max=args.epochs,
                               eta_min=args.lr_min,
                     )
           elif args.lr_scheduler == "step":
                     scheduler = optim.lr_scheduler.StepLR(
                               optimizer,
                               step_size=args.lr_patience,
                               gamma=args.lr_factor,
                     )

           best_val_ang = float('inf')
           best_epoch = 0
           best_checkpoint_path = None
           save_dir = Path(args.save_dir)
                 
           early_stop_counter = 0
           early_stop_triggered = False
                 
           log_dir = Path(args.log_dir)
           logger = TrainingLogger(log_dir)
                 
           dataset_info = {
                     "total_samples": len(full_dataset),
                     "train_samples": len(train_set),
                     "val_samples": len(val_set),
                     "test_samples": len(test_indices),
           }
           logger.log_config(args, dataset_info)
           print(f"Training log will be saved to: {logger.log_file}")

           for epoch in range(1, args.epochs + 1):
                     train_loss = train_one_epoch(
                               model,
                               train_loader,
                               optimizer,
                               device,
                               epoch,
                               args.log_interval,
                               use_sdm=args.use_sdm,
                               sdm_separation_weight=args.sdm_separation_weight,
                               scaler=scaler if use_amp else None,
                               max_grad_norm=args.max_grad_norm,
                     )
                     val_l1, val_ang = test_epoch(model, val_loader, device, use_amp)
                     current_lr = optimizer.param_groups[0]['lr']
                     print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_l1={val_l1:.4f} val_ang={val_ang:.4f} lr={current_lr:.2e}")
                           
                     if scheduler is not None:
                               if args.lr_scheduler == "plateau":
                                         scheduler.step(val_ang)
                               else:
                                         scheduler.step()
                           
                     is_best = False
                     improvement = best_val_ang - val_ang
                     if improvement > args.early_stop_min_delta:
                               is_best = True
                               best_val_ang = val_ang
                               best_epoch = epoch
                               early_stop_counter = 0
                               print(f"Better model found! Validation angular error: {val_ang:.4f}° (improvement: {improvement:.4f}°) (Epoch {epoch})")
                     else:
                               early_stop_counter += 1
                               if args.early_stop and early_stop_counter >= args.early_stop_patience:
                                         early_stop_triggered = True
                                         print(f"\nEarly stopping triggered: validation set has not improved for {args.early_stop_patience} epochs (min improvement threshold: {args.early_stop_min_delta:.4f}°)")
                                         print(f"Best model: Epoch {best_epoch}, Validation angular error: {best_val_ang:.4f}°")
                                         break
                           
                     logger.log_epoch(epoch, train_loss, val_l1, val_ang, is_best=is_best)
                           
                     current_checkpoint = save_checkpoint(
                               model,       
                               save_dir,       
                               epoch,       
                               is_best=is_best,
                               best_checkpoint_path=best_checkpoint_path,
                     )
                           
                     if is_best:
                               best_checkpoint_path = current_checkpoint
                 
           if early_stop_triggered:
                     print(f"Training ended due to early stopping (Epoch {epoch}/{args.epochs})")
           else:
                     print(f"Training complete (all {args.epochs} epochs)")
                 
           logger.log_final(best_epoch, best_val_ang, best_checkpoint_path)
           print(f"\nBest model: Epoch {best_epoch}, Validation angular error: {best_val_ang:.4f}°")
           print(f"Best checkpoint saved to: {best_checkpoint_path}")
           print(f"Training log saved to: {logger.log_file}")
           print(f"Training history saved to: {logger.json_file}")


if __name__ == "__main__":
           main()
