from abc import ABC, abstractmethod
from typing import Dict, Any
import optuna
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        self.metric_average = 'macro'  # 기본값 설정
    
    @abstractmethod
    def fit(self, data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        pass
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any], average: str = 'macro') -> Dict[str, float]:
        pass
    
    @abstractmethod
    def sample_params(self, trial: Any, param_config: Dict) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def save_predictions(self, X: np.ndarray, file_path: str) -> None:
        """예측 결과를 파일로 저장"""
        pass

def get_model(model_name: str, debug: bool = False) -> BaseModel:
    """모델 팩토리 함수"""
    if model_name == "random_forest":
        from .sklearn_models import RandomForestModel
        return RandomForestModel()
    elif model_name == "logistic_regression":
        from .sklearn_models import LogisticRegressionModel
        return LogisticRegressionModel()
    elif model_name == "simple_nn":
        from .torch_models import SimpleNNModel
        return SimpleNNModel(debug=debug)
    elif model_name == "timm_resnet":
        from .timm_models import TimmModel
        return TimmModel(debug=debug)
    else:
        raise ValueError(f"Unknown model: {model_name}") 