from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import cv2
import mlflow
from data_utils.image_processor import ImagePreprocessor
from sklearn.datasets import load_iris, load_digits, make_classification
from torchvision import transforms
from torch.utils.data import DataLoader
from data_utils.image_dataset import ImageDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DatasetManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.data_info = {}
        
        # 이미지 크기 설정
        self.img_size = config.get('image_preprocessing', {}).get('target_size', 224)
        
        # transform 설정
        if config.get('image_preprocessing'):
            self.trn_transform = get_transforms(self.img_size, is_train=True)
            self.tst_transform = get_transforms(self.img_size, is_train=False)
    
    def load_dataset(self) -> Dict[str, Any]:
        """설정에 따라 데이터셋 로드"""
        if self.config.name == "custom":
            if hasattr(self.config, 'image') and self.config.image.data_dir:
                return self._load_image_dataset()
            elif hasattr(self.config, 'csv') and self.config.csv.train_file:
                return self._load_csv_dataset()
            else:
                raise ValueError("Neither CSV nor image dataset configuration found")
        elif self.config.name == "iris":
            return self._load_iris_dataset()
        elif self.config.name == "digits":
            return self._load_digits_dataset()
        elif self.config.name == "synthetic":
            return self._load_synthetic_dataset()
        else:
            raise ValueError(f"Unknown dataset: {self.config.name}")
    
    def _load_csv_dataset(self) -> Dict[str, Any]:
        """CSV 데이터셋 로드"""
        cfg = self.config.csv
        
        # 학습 데이터 로드
        train_df = pd.read_csv(cfg.train_file)
        X = train_df[cfg.feature_cols].values
        y = train_df[cfg.label_col].values
        
        # 테스트 데이터가 별도로 제공되는 경우
        test_data = None
        if hasattr(cfg, 'test_file') and cfg.test_file:
            # 테스트 데이터는 레이블 없이 feature만 로드
            test_df = pd.read_csv(cfg.test_file)
            test_data = {'X': test_df[cfg.feature_cols].values}
        else:
            # 테스트 세트 분리
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=self.config.split.random_state
            )
            X, y = X_train, y_train
            test_data = {'X': X_test, 'y': y_test}
        
        # Validation 분할 (학습 데이터에서만)
        train_data, val_data = self._split_validation(
            X, y,
            validation_size=self.config.split.validation_size,
            stratify=self.config.split.stratify,
            random_state=self.config.split.random_state
        )
        
        self.data_info.update({
            'n_features': len(cfg.feature_cols),
            'feature_names': cfg.feature_cols,
            'n_classes': len(np.unique(y))
        })
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'info': self.data_info
        }
    
    def _load_image_dataset(self) -> Dict[str, Any]:
        """이미지 데이터셋 로드"""
        cfg = self.config.image
        base_dir = Path(cfg.data_dir)
        
        # Dataset 정의
        trn_dataset = ImageDataset(
            csv_path=base_dir / 'train.csv',
            image_dir=base_dir / 'train',
            transform=self.trn_transform,
            id_col=cfg.id_col,
            target_col=cfg.target_col
        )
        
        tst_dataset = ImageDataset(
            csv_path=base_dir / 'sample_submission.csv',
            image_dir=base_dir / 'test',
            transform=self.tst_transform,
            id_col=cfg.id_col,
            target_col=cfg.target_col
        )
        
        if self.config.debug.enabled:
            print(f"\nDataset sizes:")
            print(f"Train: {len(trn_dataset)}")
            print(f"Test: {len(tst_dataset)}")
        
        # DataLoader 정의
        trn_loader = DataLoader(
            trn_dataset,
            batch_size=self.config.model.params.batch_size,
            shuffle=True,
            num_workers=self.config.environment.num_workers,
            pin_memory=True
        )
        
        tst_loader = DataLoader(
            tst_dataset,
            batch_size=self.config.model.params.batch_size,
            shuffle=False,
            num_workers=self.config.environment.num_workers,
            pin_memory=True
        )
        
        # 메타데이터 저장
        meta_df = pd.read_csv(base_dir / cfg.meta_file) if cfg.meta_file else None
        self.data_info.update({
            'n_train': len(trn_dataset),
            'n_test': len(tst_dataset),
            'n_classes': len(meta_df) if meta_df is not None else None,
            'class_names': meta_df['class_name'].tolist() if meta_df is not None else None
        })
        
        return {
            'train': {'loader': trn_loader, 'dataset': trn_dataset},
            'test': {'loader': tst_loader, 'dataset': tst_dataset},
            'info': self.data_info
        }
    
    def _split_validation(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_size: float,
        stratify: bool = True,
        random_state: int = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """학습 데이터에서 validation 세트 분할"""
        stratify_arg = y if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_size,
            stratify=stratify_arg,
            random_state=random_state
        )
        
        return (
            {'X': X_train, 'y': y_train},
            {'X': X_val, 'y': y_val}
        )
    
    def _log_dataset_info(self):
        """데이터셋 정보를 MLflow에 기록"""
        # 데이터셋 설정 기록
        mlflow.log_params({
            "dataset_name": self.config.name,
            "test_size": self.config.split.test_size,
            "validation_size": self.config.split.validation_size,
            "random_state": self.config.split.random_state,
            **self.data_info
        }) 
    
    def _load_iris_dataset(self) -> Dict[str, Any]:
        """Iris 데이터셋 로드"""
        data = load_iris()
        X, y = data.data, data.target
        
        # 데저 test set 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,  # 20%를 테스트 세트로
            stratify=y,
            random_state=self.config.split.random_state
        )
        
        # 남은 train set에서 validation 분리
        train_data, val_data = self._split_validation(
            X_train, y_train,
            validation_size=self.config.split.validation_size,
            stratify=self.config.split.stratify,
            random_state=self.config.split.random_state
        )
        
        # 테스트 데이터 준비
        test_data = {'X': X_test, 'y': y_test}
        
        self.data_info.update({
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'feature_names': data.feature_names,
            'target_names': data.target_names
        })
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'info': self.data_info
        }
    
    def _load_digits_dataset(self) -> Dict[str, Any]:
        """Digits 데이터셋 로드"""
        data = load_digits()
        X, y = data.data, data.target
        
        # 먼저 test set 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,  # 20%를 테스트 세트로
            stratify=y,
            random_state=self.config.split.random_state
        )
        
        # 남은 train set에서 validation 분리
        train_data, val_data = self._split_validation(
            X_train, y_train,
            validation_size=self.config.split.validation_size,
            stratify=self.config.split.stratify,
            random_state=self.config.split.random_state
        )
        
        # 테스트 데이터 준비
        test_data = {'X': X_test, 'y': y_test}
        
        self.data_info.update({
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'image_shape': (8, 8)
        })
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'info': self.data_info
        }
    
    def _load_synthetic_dataset(self) -> Dict[str, Any]:
        """합성 데이터셋 생성"""
        params = self.config.params
        X, y = make_classification(
            n_samples=params.n_samples,
            n_features=params.n_features,
            n_classes=params.n_classes,
            n_informative=params.n_informative,
            n_redundant=params.n_redundant,
            class_sep=params.class_sep,
            random_state=params.random_state
        )
        
        # 먼저 test set 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=self.config.split.random_state
        )
        
        # 남은 train set에서 validation 분리
        train_data, val_data = self._split_validation(
            X_train, y_train,
            validation_size=self.config.split.validation_size,
            stratify=self.config.split.stratify,
            random_state=self.config.split.random_state
        )
        
        # 테스트 데이터 준비
        test_data = {'X': X_test, 'y': y_test}
        
        self.data_info.update({
            'n_features': X.shape[1],
            'n_classes': params.n_classes,
            'n_samples': params.n_samples,
            'n_informative': params.n_informative,
            'n_redundant': params.n_redundant,
            'class_sep': params.class_sep
        })
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'info': self.data_info
        }

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