from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from .base import BaseModel
import pandas as pd
from pathlib import Path

class TimmModel(BaseModel):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        self.params = {
            'model_name': 'resnet34',
            'pretrained': True,
            'num_classes': None,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'max_epochs': 100
        }
    
    def _create_model(self):
        """모델 생성"""
        self.model = timm.create_model(
            self.params['model_name'],
            pretrained=self.params['pretrained'],
            num_classes=self.params['num_classes']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.params['learning_rate'])
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.params['max_epochs']
        )
    
    def fit(self, data: Dict[str, Any]) -> None:
        loader = data['loader']
        if self.params['num_classes'] is None:
            self.params['num_classes'] = self.data_info['n_classes']
        
        if self.model is None:
            self._create_model()
        
        # 학습
        for batch in loader:
            images, targets = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
    
    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        loader = data['loader']
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def evaluate(self, data: Dict[str, Any], average: str = 'macro') -> Dict[str, float]:
        y_pred = self.predict(data)
        
        metrics = {
            'accuracy': accuracy_score(data['y'], y_pred),
            'precision': precision_score(data['y'], y_pred, average=average),
            'recall': recall_score(data['y'], y_pred, average=average),
            'f1': f1_score(data['y'], y_pred, average=average)
        }
        
        # Loss 계산
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(data['X']).to(self.device)
            y = torch.LongTensor(data['y']).to(self.device)
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            metrics['loss'] = loss.item()
        
        return metrics
    
    def save_predictions(self, X: np.ndarray, file_path: str) -> None:
        predictions = self.predict({"X": X})
        
        # 예측 확률 계산
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # 파일 경로 분석
        output_dir = Path(file_path).parent
        sample_submission_path = output_dir / 'sample_submission.csv'
        
        if sample_submission_path.exists():
            # sample_submission.csv 형식 유지
            submission_df = pd.read_csv(sample_submission_path)
            
            # 새로운 파일명 생성
            if file_path.endswith('sample_submission.csv'):
                output_path = output_dir / 'submission.csv'
            else:
                output_path = Path(file_path)
            
            # 예측값 업데이트하여 새 파일로 저장
            submission_df['target'] = predictions
            submission_df.to_csv(output_path, index=False)
            
            # 확률값은 별도 파일로 저장
            if probabilities is not None:
                probs_df = pd.DataFrame(
                    probabilities,
                    columns=[f'prob_class_{i}' for i in range(probabilities.shape[1])]
                )
                probs_df.index = submission_df['ID']
                probs_path = output_path.with_stem(output_path.stem + '_probabilities')
                probs_df.to_csv(probs_path, index=True)
        else:
            # 기본 형식으로 저장
            results = {
                'prediction': predictions,
                **{f'prob_class_{i}': probabilities[:, i] for i in range(probabilities.shape[1])}
            }
            pd.DataFrame(results).to_csv(file_path, index=False)
    
    def sample_params(self, trial: optuna.Trial, param_config: Dict) -> Dict[str, Any]:
        params = {}
        for name, config in param_config.items():
            if isinstance(config, (list, tuple)):
                params[name] = trial.suggest_categorical(name, config)
            elif isinstance(config, dict):
                if config['type'] == 'int':
                    params[name] = trial.suggest_int(
                        name, config['low'], config['high'],
                        step=config.get('step', 1)
                    )
                elif config['type'] == 'float':
                    params[name] = trial.suggest_float(
                        name, config['low'], config['high'],
                        log=config.get('log', False)
                    )
        self.params.update(params)
        return params 