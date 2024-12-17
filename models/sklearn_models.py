from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from .base import BaseModel
import pandas as pd
from pathlib import Path

class SklearnModelBase(BaseModel):
    """scikit-learn 모델을 위한 기본 클래스"""
    def __init__(self):
        self.metric_average = 'macro'  # 기본값
    
    def fit(self, data: Dict[str, Any]) -> None:
        self.model.fit(data['X'], data['y'])
    
    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        return self.model.predict(data['X'])
    
    def evaluate(self, data: Dict[str, Any], average: str = None) -> Dict[str, float]:
        """
        모델 평가
        Args:
            data: 평가 데이터
            average: 메트릭 평균 방식 ('macro', 'weighted', 'micro')
        """
        y_pred = self.predict(data)
        avg = average or self.metric_average
        
        return {
            'accuracy': accuracy_score(data['y'], y_pred),
            'precision': precision_score(data['y'], y_pred, average=avg),
            'recall': recall_score(data['y'], y_pred, average=avg),
            'f1': f1_score(data['y'], y_pred, average=avg)
        }
    
    def save_predictions(self, X: np.ndarray, file_path: str) -> None:
        """예측 결과를 CSV 파일로 저장"""
        predictions = self.predict({"X": X})
        probabilities = self.model.predict_proba(X)
        
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
            
            # 확률값은 별도 파일로 저장 (선택적)
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

class RandomForestModel(SklearnModelBase):
    def __init__(self):
        self.model = None
    
    def fit(self, data: Dict[str, Any]) -> None:
        if self.model is None:
            self.model = RandomForestClassifier()
        super().fit(data)
    
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
        self.model = RandomForestClassifier(**params)
        return params

class LogisticRegressionModel(SklearnModelBase):
    def __init__(self):
        self.model = None
    
    def fit(self, data: Dict[str, Any]) -> None:
        if self.model is None:
            self.model = LogisticRegression()
        super().fit(data)
    
    def sample_params(self, trial: optuna.Trial, param_config: Dict) -> Dict[str, Any]:
        params = {}
        for name, config in param_config.items():
            if isinstance(config, (list, tuple)):
                params[name] = trial.suggest_categorical(name, config)
            elif isinstance(config, dict):
                if config['type'] == 'float':
                    params[name] = trial.suggest_float(
                        name, config['low'], config['high'],
                        log=config.get('log', False)
                    )
        self.model = LogisticRegression(**params)
        return params 