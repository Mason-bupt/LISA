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
from models.clip_gaze import (
           CLIPGazeConfig,
           feature_separation_loss,
)


def parse_args() -> argparse.Namespace:
           parser = argparse.ArgumentParser(description="ResNet18-FAM-Lite Gaze Training (按Subject划分)")
           parser.add_argument("--data_root", type=str, default=".", help="数据根目录（包含Subject*_*_data文件夹的路径，支持绝对路径或相对路径）")
           parser.add_argument("--batch_size", type=int, default=64)
           parser.add_argument("--epochs", type=int, default=50)
           parser.add_argument("--lr", type=float, default=1e-4)
           parser.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减（增强正则化，默认5e-4）")
           parser.add_argument("--num_workers", type=int, default=16)
           parser.add_argument("--pin_memory", action="store_true", default=True)
           parser.add_argument("--persistent_workers", action="store_true", default=True)
           parser.add_argument("--device", type=str, default="cuda")
           parser.add_argument("--log_interval", type=int, default=10)
           parser.add_argument("--save_dir", type=str, default="checkpoints")
           parser.add_argument("--resume", type=str, default="", help="可选的模型权重路径")
           parser.add_argument("--train_subjects", type=str, default="1-22", help="训练集subject编号范围，如'1-22'")
           parser.add_argument("--val_subjects", type=str, default="23-25", help="验证集subject编号范围，如'23-25'")
           parser.add_argument("--test_subjects", type=str, default="26-28", help="测试集subject编号范围，如'26-28'")
           parser.add_argument("--test_indices_file", type=str, default="test_indices_by_subject.json", help="测试集索引保存路径")
           parser.add_argument("--amp", action="store_true", help="启用CUDA自动混合精度")
           parser.add_argument("--compile", action="store_true", help="尝试torch.compile加速")
           # CLIP-Gaze相关参数（简化版）
           parser.add_argument("--use_clip_gaze", action="store_true", default=True, help="启用CLIP-Gaze（默认启用）")
           parser.add_argument("--no-use-clip-gaze", dest="use_clip_gaze", action="store_false", help="禁用CLIP-Gaze")
           parser.add_argument("--clip_separation_weight", type=float, default=100.0, help="特征分离损失权重（提高以激活CLIP分支，建议100-500）")
           parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪的最大范数")
           parser.add_argument("--log_dir", type=str, default="logs", help="训练日志保存目录")
           # 学习率调度相关参数
           parser.add_argument("--lr_scheduler", type=str, choices=["plateau", "cosine", "step", "none"],       
                                                            default="plateau", help="学习率调度策略（默认plateau）")
           parser.add_argument("--lr_factor", type=float, default=0.5, help="学习率衰减因子（plateau/step使用）")
           parser.add_argument("--lr_patience", type=int, default=5, help="学习率调度耐心值（plateau使用）")
           parser.add_argument("--lr_min", type=float, default=1e-6, help="最小学习率")
           # 早停相关参数
           parser.add_argument("--early_stop", action="store_true", default=True, help="启用早停（默认启用）")
           parser.add_argument("--no-early-stop", dest="early_stop", action="store_false", help="禁用早停")
           parser.add_argument("--early_stop_patience", type=int, default=10, help="早停耐心值（验证集多少个epoch不改善就停止）")
           parser.add_argument("--early_stop_min_delta", type=float, default=0.001, help="早停最小改善阈值")
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


def angular_loss_deg(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
           """计算 gaze 角度偏差 (deg)，内部转换为弧度求余弦相似度。"""
           # 检查输入是否为NaN或Inf
           if torch.isnan(pred_deg).any() or torch.isinf(pred_deg).any():
                     return torch.tensor(180.0, device=pred_deg.device)      # 返回最大角度误差
           if torch.isnan(target_deg).any() or torch.isinf(target_deg).any():
                     return torch.tensor(180.0, device=target_deg.device)
                 
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
                 
           # 检查结果是否为NaN或Inf
           if torch.isnan(angle).any() or torch.isinf(angle).any():
                     return torch.tensor(180.0, device=pred_deg.device)
                 
           result = torch.rad2deg(angle).mean()
                 
           # 最终检查
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
           use_clip_gaze: bool = True,
           clip_separation_weight: float = 0.5,
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
                                     
                               # 基础损失
                               loss_l1 = F.l1_loss(gaze_pred, targets)
                               loss_ang = angular_loss_deg(gaze_pred, targets)
                               loss = loss_l1 + 0.5 * loss_ang
                                     
                               # CLIP-Gaze损失：特征分离损失（push away与注视无关的特征）
                               loss_separation = None
                               if use_clip_gaze and "gaze_features_proj" in outputs:
                                         gaze_features_proj = outputs["gaze_features_proj"]
                                         text_features = outputs["text_features"]
                                               
                                         loss_separation = feature_separation_loss(
                                                   gaze_features_proj, text_features
                                         )
                                         loss = loss + clip_separation_weight * loss_separation

                     # 检查损失是否为NaN或Inf
                     if torch.isnan(loss) or torch.isinf(loss):
                               print(f"警告: Epoch {epoch} Step {step} 检测到NaN/Inf损失，跳过此步")
                               continue
                           
                     optimizer.zero_grad(set_to_none=True)
                     if scaler is not None:
                               scaler.scale(loss).backward()
                               # 梯度裁剪：防止梯度爆炸
                               scaler.unscale_(optimizer)
                               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                               scaler.step(optimizer)
                               scaler.update()
                     else:
                               loss.backward()
                               # 梯度裁剪：防止梯度爆炸
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
           """
           保存checkpoint（只保存最佳模型）
                 
           Args:
                     model: 模型
                     save_dir: 保存目录
                     epoch: 当前epoch
                     is_best: 是否为最佳模型
                     best_checkpoint_path: 之前最佳checkpoint的路径（用于删除）
                 
           Returns:
                     保存的checkpoint路径（如果不是最佳模型则返回None）
           """
           if not is_best:
                     return None
                 
           save_dir.mkdir(parents=True, exist_ok=True)
           ckpt_path = save_dir / "best.pth"
                 
           # 删除之前的最佳checkpoint（如果存在且不是同一个文件）
           if best_checkpoint_path is not None and best_checkpoint_path.exists() and best_checkpoint_path != ckpt_path:
                     best_checkpoint_path.unlink()
                     print(f"已删除旧的最佳checkpoint: {best_checkpoint_path}")
                 
           torch.save(model.state_dict(), ckpt_path)
           print(f"✓ 最佳checkpoint已保存至: {ckpt_path}")
           return ckpt_path


class TrainingLogger:
           """训练日志记录器，保存训练历史和配置信息"""
                 
           def __init__(self, log_dir: Path):
                     self.log_dir = Path(log_dir)
                     self.log_dir.mkdir(parents=True, exist_ok=True)
                           
                     # 创建带时间戳的日志文件
                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                     self.log_file = self.log_dir / f"training_log_by_subject_{timestamp}.txt"
                     self.json_file = self.log_dir / f"training_history_by_subject_{timestamp}.json"
                           
                     # 训练历史记录
                     self.history: Dict[str, list] = {
                               "epoch": [],
                               "train_loss": [],
                               "val_l1": [],
                               "val_ang": [],
                     }
                           
                     # 训练配置
                     self.config: Dict[str, Any] = {}
                           
                     # 初始化日志文件
                     with self.log_file.open("w", encoding="utf-8") as f:
                               f.write(f"训练日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                               f.write("=" * 80 + "\n\n")
                 
           def log_config(self, args: argparse.Namespace, dataset_info: Dict[str, Any]):
                     """记录训练配置信息"""
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
                               "use_clip_gaze": args.use_clip_gaze,
                               "clip_separation_weight": args.clip_separation_weight,
                               "max_grad_norm": args.max_grad_norm,
                               "amp": args.amp,
                               "compile": args.compile,
                               "dataset_info": dataset_info,
                     }
                           
                     with self.log_file.open("a", encoding="utf-8") as f:
                               f.write("训练配置:\n")
                               f.write("-" * 80 + "\n")
                               for key, value in self.config.items():
                                         if key != "dataset_info":
                                                   f.write(f"      {key}: {value}\n")
                               f.write(f"\n数据集信息:\n")
                               for key, value in dataset_info.items():
                                         f.write(f"      {key}: {value}\n")
                               f.write("\n" + "=" * 80 + "\n\n")
                               f.write("训练过程:\n")
                               f.write("-" * 80 + "\n")
                 
           def log_epoch(self, epoch: int, train_loss: float, val_l1: float, val_ang: float, is_best: bool = False):
                     """记录每个 epoch 的训练指标"""
                     self.history["epoch"].append(epoch)
                     self.history["train_loss"].append(float(train_loss))
                     self.history["val_l1"].append(float(val_l1))
                     self.history["val_ang"].append(float(val_ang))
                           
                     # 写入文本日志
                     with self.log_file.open("a", encoding="utf-8") as f:
                               best_marker = " [BEST]" if is_best else ""
                               f.write(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_l1={val_l1:.6f}, val_ang={val_ang:.6f}{best_marker}\n")
                           
                     # 保存 JSON 历史（每个 epoch 都保存，以便随时查看）
                     with self.json_file.open("w", encoding="utf-8") as f:
                               json.dump({
                                         "config": self.config,
                                         "history": self.history,
                               }, f, indent=2, ensure_ascii=False)
                 
           def log_final(self, best_epoch: int, best_val_ang: float, checkpoint_path: Optional[Path]):
                     """记录训练完成信息"""
                     with self.log_file.open("a", encoding="utf-8") as f:
                               f.write("\n" + "=" * 80 + "\n")
                               f.write("训练完成\n")
                               f.write("-" * 80 + "\n")
                               f.write(f"最佳模型: Epoch {best_epoch}\n")
                               f.write(f"最佳验证集角度误差: {best_val_ang:.6f}°\n")
                               if checkpoint_path:
                                         f.write(f"Checkpoint路径: {checkpoint_path}\n")
                               f.write(f"日志文件: {self.log_file}\n")
                               f.write(f"历史文件: {self.json_file}\n")
                               f.write("=" * 80 + "\n")


def main() -> None:
           args = parse_args()
           device = torch.device(args.device if torch.cuda.is_available() else "cpu")
           if device.type == "cuda":
                     torch.backends.cudnn.benchmark = True

           use_amp = args.amp and device.type == "cuda"
           scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

           # 解析subject编号范围
           train_subject_ids = parse_subject_range(args.train_subjects)
           val_subject_ids = parse_subject_range(args.val_subjects)
           test_subject_ids = parse_subject_range(args.test_subjects)
                 
           # 检查是否有重叠
           overlap_train_val = set(train_subject_ids) & set(val_subject_ids)
           overlap_train_test = set(train_subject_ids) & set(test_subject_ids)
           overlap_val_test = set(val_subject_ids) & set(test_subject_ids)
                 
           if overlap_train_val:
                     raise ValueError(f"训练集和验证集的subject编号有重叠: {overlap_train_val}")
           if overlap_train_test:
                     raise ValueError(f"训练集和测试集的subject编号有重叠: {overlap_train_test}")
           if overlap_val_test:
                     print(f"警告: 验证集和测试集的subject编号有重叠: {overlap_val_test}，这些subject的数据会同时出现在验证集和测试集中")
                 
           print(f"数据划分配置:")
           print(f"      训练集: subject {args.train_subjects} (共 {len(train_subject_ids)} 个subject)")
           print(f"      验证集: subject {args.val_subjects} (共 {len(val_subject_ids)} 个subject)")
           print(f"      测试集: subject {args.test_subjects} (共 {len(test_subject_ids)} 个subject)")

           # 创建完整数据集（用于获取索引，不应用transform）
           full_dataset = FaceGazeDataset(root_dir=args.data_root)
                 
           # 根据subject编号获取索引
           print("\n正在根据subject编号划分数据集...")
           train_indices = get_indices_by_subjects(full_dataset, train_subject_ids)
           val_indices = get_indices_by_subjects(full_dataset, val_subject_ids)
           test_indices = get_indices_by_subjects(full_dataset, test_subject_ids)
                 
           if len(train_indices) == 0:
                     raise ValueError(f"训练集为空！请检查subject编号范围 {args.train_subjects} 是否正确")
           if len(val_indices) == 0:
                     raise ValueError(f"验证集为空！请检查subject编号范围 {args.val_subjects} 是否正确")
           if len(test_indices) == 0:
                     raise ValueError(f"测试集为空！请检查subject编号范围 {args.test_subjects} 是否正确")
                 
           # 保存测试集索引供 test_by_subject.py 使用
           test_indices_path = Path(args.test_indices_file)
           with test_indices_path.open("w", encoding="utf-8") as f:
                     json.dump(test_indices, f)
           print(f"\n测试集索引已保存到: {test_indices_path} (共 {len(test_indices)} 个样本)")
           print(f"数据划分统计: 训练集={len(train_indices)} 个样本, 验证集={len(val_indices)} 个样本, 测试集={len(test_indices)} 个样本")
                 
           # 创建带数据增强的训练集和不带增强的验证集
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
                 
           # 使用Subset创建训练集和验证集
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

           # 配置CLIP-Gaze（简化版）
           clip_gaze_config = None
           if args.use_clip_gaze:
                     clip_gaze_config = CLIPGazeConfig(
                               feature_separation_weight=args.clip_separation_weight,
                     )
                 
           cfg = ResNetFAMLiteConfig(
                     use_clip_gaze=args.use_clip_gaze,
                     clip_gaze_config=clip_gaze_config,
           )
           model = ResNetFAMLite(cfg).to(device)
           if args.compile and hasattr(torch, "compile"):
                     model = torch.compile(model)
           if args.resume:
                     state = torch.load(args.resume, map_location="cpu")
                     model.load_state_dict(state, strict=False)
                     print(f"Loaded checkpoint from {args.resume}")

           optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                 
           # 创建学习率调度器
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

           # 跟踪最佳验证集表现（使用angular error作为主要指标）
           best_val_ang = float('inf')
           best_epoch = 0
           best_checkpoint_path = None
           save_dir = Path(args.save_dir)
                 
           # 早停相关变量
           early_stop_counter = 0
           early_stop_triggered = False
                 
           # 初始化训练日志记录器
           log_dir = Path(args.log_dir)
           logger = TrainingLogger(log_dir)
                 
           # 记录训练配置
           dataset_info = {
                     "total_samples": len(full_dataset),
                     "train_samples": len(train_set),
                     "val_samples": len(val_set),
                     "test_samples": len(test_indices),
           }
           logger.log_config(args, dataset_info)
           print(f"训练日志将保存到: {logger.log_file}")

           for epoch in range(1, args.epochs + 1):
                     train_loss = train_one_epoch(
                               model,
                               train_loader,
                               optimizer,
                               device,
                               epoch,
                               args.log_interval,
                               use_clip_gaze=args.use_clip_gaze,
                               clip_separation_weight=args.clip_separation_weight,
                               scaler=scaler if use_amp else None,
                               max_grad_norm=args.max_grad_norm,
                     )
                     val_l1, val_ang = test_epoch(model, val_loader, device, use_amp)
                     current_lr = optimizer.param_groups[0]['lr']
                     print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_l1={val_l1:.4f} val_ang={val_ang:.4f} lr={current_lr:.2e}")
                           
                     # 更新学习率调度器
                     if scheduler is not None:
                               if args.lr_scheduler == "plateau":
                                         scheduler.step(val_ang)
                               else:
                                         scheduler.step()
                           
                     # 检查是否为最佳模型（angular error越小越好）
                     is_best = False
                     improvement = best_val_ang - val_ang
                     if improvement > args.early_stop_min_delta:
                               is_best = True
                               best_val_ang = val_ang
                               best_epoch = epoch
                               early_stop_counter = 0      # 重置早停计数器
                               print(f"✓ 发现更好的模型！验证集角度误差: {val_ang:.4f}° (改善: {improvement:.4f}°) (Epoch {epoch})")
                     else:
                               early_stop_counter += 1
                               if args.early_stop and early_stop_counter >= args.early_stop_patience:
                                         early_stop_triggered = True
                                         print(f"\n早停触发：验证集已连续 {args.early_stop_patience} 个epoch未改善（最小改善阈值: {args.early_stop_min_delta:.4f}°）")
                                         print(f"最佳模型: Epoch {best_epoch}, 验证集角度误差: {best_val_ang:.4f}°")
                                         break
                           
                     # 记录训练指标到日志
                     logger.log_epoch(epoch, train_loss, val_l1, val_ang, is_best=is_best)
                           
                     # 保存checkpoint（只保留最佳模型）
                     current_checkpoint = save_checkpoint(
                               model,       
                               save_dir,       
                               epoch,       
                               is_best=is_best,
                               best_checkpoint_path=best_checkpoint_path,
                     )
                           
                     if is_best:
                               best_checkpoint_path = current_checkpoint
                 
           # 记录训练完成信息
           if early_stop_triggered:
                     print(f"训练因早停而结束（Epoch {epoch}/{args.epochs}）")
           else:
                     print(f"训练完成（所有 {args.epochs} 个epoch）")
                 
           logger.log_final(best_epoch, best_val_ang, best_checkpoint_path)
           print(f"\n最佳模型: Epoch {best_epoch}, 验证集角度误差: {best_val_ang:.4f}°")
           print(f"最佳checkpoint已保存至: {best_checkpoint_path}")
           print(f"训练日志已保存至: {logger.log_file}")
           print(f"训练历史已保存至: {logger.json_file}")


if __name__ == "__main__":
           main()
