from __future__ import annotations

import re
import random
import cv2  # 需要 pip install opencv-python
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

try:
    from data_normalization import normalize_data, vector_to_pitchyaw
except ImportError:
    raise ImportError("请确保当前目录下有 data_normalization.py 文件！")

_ARRAY_PATTERN = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

def _parse_numeric_array(raw: str) -> np.ndarray:
    """解析字符串为 numpy 数组"""
    values = [float(x) for x in _ARRAY_PATTERN.findall(raw)]
    if not values:
        raise ValueError(f"无法解析标注内容: {raw}")
    return np.array(values, dtype=np.float32)

@dataclass(frozen=True)
class SampleEntry:
    stem: str
    image_path: Path
    label_data: Dict[str, np.ndarray]


class FaceGazeDataset(Dataset):
    """
    集成 3D 归一化的高性能 Gaze 数据集。
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_dir: str = "face_ims",
        label_dir: str = "gaze_info",
        is_train: bool = True,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        image_size: Optional[Tuple[int, int]] = (224, 224),
    ) -> None:
        self.root = Path(root_dir)
        self.image_dir_name = image_dir
        self.label_dir_name = label_dir
        self.is_train = is_train
        self.transform = transform
        self.image_size = image_size
        
        self.samples = self._gather_samples_and_cache()

    def _gather_samples_and_cache(self) -> List[SampleEntry]:
        entries: List[SampleEntry] = []
        subject_folders = sorted(self.root.glob("Subject*_*_data"))
        
        if not subject_folders:
            raise RuntimeError(f"在 {self.root} 中未找到任何 Subject*_*_data 文件夹。")
            
        print(f"[Dataset] 正在扫描 {len(subject_folders)} 个文件夹并缓存标签...")
        
        for subject_folder in subject_folders:
            image_dir = subject_folder / self.image_dir_name
            label_dir = subject_folder / self.label_dir_name
            
            if not image_dir.exists() or not label_dir.exists():
                continue
                
            # 遍历标签文件
            for label_path in sorted(label_dir.glob("*_gaze.txt")):
                stem = label_path.stem.replace("_gaze", "")
                image_path = image_dir / f"{stem}_face.png"
                
                if not image_path.exists():
                    continue
                
                try:
                    label_data = self._parse_label_file(label_path)
                    def is_valid_eye_data(eye_loc_key: str, gaze_dir_key: str) -> bool:
                        if eye_loc_key not in label_data or gaze_dir_key not in label_data:
                            return False
                        eye_loc = label_data[eye_loc_key]
                        # 检查是否为无效坐标 [0, 0, 0]
                        if np.linalg.norm(eye_loc) < 1e-6:
                            return False
                        return True
                    
                    has_right = is_valid_eye_data("Right_3D_Eye_Loc", "Right_Gaze_Dir")
                    has_left = is_valid_eye_data("Left_3D_Eye_Loc", "Left_Gaze_Dir")
                    
                    if not has_right and not has_left:
                        continue
                    
                    label_data["_use_eye"] = "right" if has_right else "left"
                        
                    entries.append(SampleEntry(
                        stem=stem, 
                        image_path=image_path, 
                        label_data=label_data
                    ))
                except Exception:
                    continue

        print(f"[Dataset] 模式: {'Train' if self.is_train else 'Val'} | 有效样本数: {len(entries)}")
        return entries

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_retries = 10
        current_idx = idx
        
        for _ in range(max_retries):
            try:
                sample = self.samples[current_idx]
                
                img_bgr = cv2.imread(str(sample.image_path))
                if img_bgr is None:
                    raise ValueError(f"图像读取失败: {sample.image_path}")
                    
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                label_data = sample.label_data
                focal_length = w * 1.0 
                camera_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                use_eye = label_data.get("_use_eye", "right")
                if use_eye == "right":
                    eye_loc_key = "Right_3D_Eye_Loc"
                    gaze_dir_key = "Right_Gaze_Dir"
                else:
                    eye_loc_key = "Left_3D_Eye_Loc"
                    gaze_dir_key = "Left_Gaze_Dir"
                
                warped_img, norm_gaze_vec = normalize_data(
                    img_rgb,
                    face_center=label_data[eye_loc_key], 
                    camera_matrix=camera_matrix,
                    gaze_target_3d=label_data[gaze_dir_key]
                )
                
                if norm_gaze_vec is None:
                    raise ValueError("Gaze vector 无效")

                norm_angles = vector_to_pitchyaw(norm_gaze_vec)
                gaze_angles = torch.tensor(norm_angles, dtype=torch.float32)

                image_pil = Image.fromarray(warped_img)
                if self.is_train and random.random() < 0.5:
                    image_pil = F.hflip(image_pil)
                    gaze_angles[0] = -gaze_angles[0]

                if self.transform:
                    image_tensor = self.transform(image_pil)
                else:
                    image_tensor = transforms.ToTensor()(image_pil)

                return {
                    "image": image_tensor,
                    "gaze_angles_deg": gaze_angles,
                    "id": sample.stem,
                }

            except Exception:
                current_idx = random.randint(0, len(self.samples) - 1)
        
        raise RuntimeError("连续 10 次加载样本失败，请检查数据集完整性！")

    @staticmethod
    def _parse_label_file(label_path: Path) -> Dict[str, np.ndarray]:
        """解析标签文件为字典，值为 numpy array"""
        parsed = {}
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line: continue
                key, raw = line.split(":", maxsplit=1)
                parsed[key.strip()] = _parse_numeric_array(raw)
        return parsed

def get_train_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
    ])

def get_val_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])