import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    """이미지 데이터셋 클래스"""
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        id_col: str = 'ID',
        target_col: str = 'target'
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.id_col = id_col
        self.target_col = target_col
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row[self.id_col])
        
        # 이미지 로드
        img = np.array(Image.open(img_path))
        
        # transform 적용
        if self.transform:
            img = self.transform(image=img)['image']
        
        # 타�� 값 (없을 수 있음)
        target = row.get(self.target_col, -1)
        
        return img, target

def get_transforms(img_size: int, is_train: bool = True) -> A.Compose:
    """이미지 변환 파이프라인 생성"""
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) 