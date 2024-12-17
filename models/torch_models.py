from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from .base import BaseModel
import pandas as pd
from tqdm import tqdm
from pathlib import Path

class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SimpleNNModel(BaseModel):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.params = {
            'input_size': None,
            'hidden_size': 64,
            'num_classes': None,
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 100
        }
    
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(data).to(self.device)
    
    def fit(self, data: Dict[str, Any]) -> None:
        X, y = data['X'], data['y']
        
        # 입력 크기와 클래스 수 설정
        if self.params['input_size'] is None:
            self.params['input_size'] = X.shape[1]
        if self.params['num_classes'] is None:
            self.params['num_classes'] = len(np.unique(y))
        
        if self.model is None:
            self.model = SimpleNN(
                input_size=self.params['input_size'],
                hidden_size=self.params['hidden_size'],
                num_classes=self.params['num_classes']
            ).to(self.device)
        
        # 데이터 준비
        X_tensor = self._to_tensor(X)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'],
            shuffle=True
        )
        
        # 학습 설정
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['learning_rate']
        )
        
        # 학습 진행바 설정
        pbar = tqdm(range(self.params['max_epochs']), desc="Training")
        
        # 학습
        self.model.train()
        for epoch in pbar:
            total_loss = 0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 현재 epoch에서의 validation 성능 계산
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.params['max_epochs'] - 1:
                self.model.eval()
                with torch.no_grad():
                    val_metrics = self.evaluate(data)
                    self.model.train()
                
                # 진행바 업데이트
                pbar.set_postfix({
                    'loss': f"{total_loss/len(dataloader):.4f}",
                    'val_acc': f"{val_metrics['accuracy']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}"
                })
            else:
                # loss만 표시
                pbar.set_postfix({'loss': f"{total_loss/len(dataloader):.4f}"})
        
        # 최종 성능 출력
        final_metrics = self.evaluate(data)
        print("\nTraining completed:")
        print(f"Final loss: {final_metrics['loss']:.4f}")
        print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final F1 score: {final_metrics['f1']:.4f}")
    
    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(data['X'])
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        return predictions
    
    def evaluate(self, data: Dict[str, Any], average: str = 'macro') -> Dict[str, float]:
        y_pred = self.predict(data)
        
        metrics = {
            'accuracy': accuracy_score(data['y'], y_pred),
            'precision': precision_score(data['y'], y_pred, average=average),
            'recall': recall_score(data['y'], y_pred, average=average),
            'f1': f1_score(data['y'], y_pred, average=average)
        }
        
        # Loss 계산
        with torch.no_grad():
            X_tensor = self._to_tensor(data['X'])
            y_tensor = torch.LongTensor(data['y']).to(self.device)
            outputs = self.model(X_tensor)
            loss = nn.CrossEntropyLoss()(outputs, y_tensor)
            metrics['loss'] = loss.item()
        
        return metrics
    
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
    
    def save_predictions(self, X: np.ndarray, file_path: str) -> None:
        """예측 결과를 CSV 파일로 저장"""
        predictions = self.predict({"X": X})
        
        # 예측 확률 계산
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # 파일 경로 분석
        output_dir = Path(file_path).parent
        sample_submission_path = output_dir / 'sample_submission.csv'
        
        if sample_submission_path.exists():
            if self.debug:
                print(f"\nFound sample_submission.csv at: {sample_submission_path}")
            
            # sample_submission.csv 로드
            submission_df = pd.read_csv(sample_submission_path)
            
            if 'ID' in submission_df.columns and 'target' in submission_df.columns:
                if self.debug:
                    print("Updating sample_submission.csv format with predictions")
                    print(f"Number of predictions: {len(predictions)}")
                    print(f"Number of samples in submission file: {len(submission_df)}")
                
                # ID 순서와 예측값 매칭 확인
                if len(predictions) == len(submission_df):
                    # 예측값 업데이트
                    submission_df['target'] = predictions
                    
                    # 새로운 파일명 생성 (submission.csv 또는 지정된 경로)
                    if file_path.endswith('sample_submission.csv'):
                        output_path = output_dir / 'submission.csv'
                    else:
                        output_path = Path(file_path)
                    
                    # 저장
                    submission_df.to_csv(output_path, index=False)
                    if self.debug:
                        print(f"Saved predictions to: {output_path}")
                        print(f"First few predictions:\n{submission_df.head()}")
                else:
                    print(f"Warning: Number of predictions ({len(predictions)}) "
                          f"doesn't match submission file ({len(submission_df)})")
                    # 기본 형식으로 저장
                    self._save_default_format(predictions, probabilities, file_path)
            else:
                print("Warning: sample_submission.csv doesn't have expected columns (ID, target)")
                self._save_default_format(predictions, probabilities, file_path)
        else:
            if self.debug:
                print(f"No sample_submission.csv found at: {sample_submission_path}")
                print("Saving in default format")
            self._save_default_format(predictions, probabilities, file_path)

    def _save_default_format(self, predictions: np.ndarray, probabilities: np.ndarray, file_path: str):
        """기본 예측 결과 저장 형식"""
        results = {
            'prediction': predictions,
            **{f'prob_class_{i}': probabilities[:, i] for i in range(probabilities.shape[1])}
        }
        pd.DataFrame(results).to_csv(file_path, index=False)
        if self.debug:
            print(f"Saved predictions in default format to: {file_path}")