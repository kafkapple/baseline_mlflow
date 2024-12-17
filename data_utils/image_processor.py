from typing import Dict, Any, Union, List, Tuple
import numpy as np
import cv2
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

class ImagePreprocessor(BaseEstimator, TransformerMixin):
    """이미지 전처리 및 특징 추출 클래스"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 feature_extractor: str = 'basic',
                 device: str = None):
        """
        Args:
            target_size: 출력 이미지 크기 (height, width)
            feature_extractor: 특징 추출 방식 ('basic', 'hog', 'resnet')
            device: 'cuda' or 'cpu' (resnet 사용시)
        """
        self.target_size = target_size
        self.feature_extractor = feature_extractor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_extractor()
    
    def _setup_extractor(self):
        """특징 추출기 초기화"""
        if self.feature_extractor == 'resnet':
            # ResNet 모델 로드 (마지막 레이어 제외)
            model = models.resnet18(weights='DEFAULT')
            self.feature_model = nn.Sequential(*list(model.children())[:-1])
            self.feature_model.to(self.device)
            self.feature_model.eval()
            
            # transform을 개별 컴포넌트로 분리
            self.resize = T.Resize(self.target_size)
            self.to_tensor = T.ToTensor()
            self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    
    def _load_image(self, image_path: str) -> Image.Image:
        """이미지 파일 로드 및 PIL Image로 변환"""
        if isinstance(image_path, str):
            # 파일에서 로드
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
            
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return image
    
    def _preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """기본 이미지 전처리"""
        # PIL Image로 변환
        if not isinstance(image, Image.Image):
            image = self._load_image(image)
        
        # 크기 조정
        image = image.resize(self.target_size)
        
        # numpy 배열로 변환
        image = np.array(image).astype(np.float32) / 255.0
        
        return image
    
    def _extract_basic_features(self, image: np.ndarray) -> np.ndarray:
        """기본 특징 추출 (색상 히스토그램 + 통계)"""
        features = []
        
        # 색상 히스토그램
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [32], [0, 1])
            features.extend(hist.flatten())
        
        # 기본 통계
        features.extend([
            image.mean(),
            image.std(),
            image.var()
        ])
        
        return np.array(features)
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """HOG 특징 추출"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        return features.flatten()
    
    def _extract_resnet_features(self, image: Image.Image) -> np.ndarray:
        """ResNet 특징 추출"""
        # 이미지 전처리를 단계별로 수행
        img = self.resize(image)
        img_tensor = self.to_tensor(img)
        img_normalized = self.normalize(img_tensor)
        
        # 배치 차원 추가 및 디바이스 이동
        img_batch = img_normalized.unsqueeze(0).to(self.device)
        
        # 특징 추출
        with torch.no_grad():
            features = self.feature_model(img_batch)
            features = features.squeeze().cpu().numpy()
        
        return features
    
    def fit(self, X: Union[List[str], np.ndarray], y=None):
        """학습 (아무것도 하지 않음)"""
        return self
    
    def transform(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """이미지를 특징 벡터로 변환"""
        features = []
        
        for x in X:
            # 이미지 로드 및 PIL Image로 변환
            image = self._load_image(x)
            
            # 특징 추출
            if self.feature_extractor == 'resnet':
                feat = self._extract_resnet_features(image)
            else:
                # basic과 hog는 numpy 배열 필요
                image_array = self._preprocess_image(image)
                if self.feature_extractor == 'basic':
                    feat = self._extract_basic_features(image_array)
                elif self.feature_extractor == 'hog':
                    feat = self._extract_hog_features(image_array)
            
            features.append(feat)
        
        return np.array(features) 