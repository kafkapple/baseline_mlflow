from typing import Callable, Dict, Any, Tuple
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler

class HyperparameterSearch:
    def __init__(
        self,
        study_name: str,
        direction: str = "maximize",
        storage: str = None
    ):
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.study = self._create_study()
    
    def _create_study(self) -> optuna.Study:
        """Optuna study 생성"""
        return optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=True
        )
    
    def optimize(
        self,
        objective: Callable,
        n_trials: int = 100,
        method: str = "tpe"
    ) -> Tuple[Dict[str, Any], float]:
        """하이퍼파라미터 최적화 실행"""
        # 샘플러 설정
        if method == "tpe":
            sampler = TPESampler()
        elif method == "random":
            sampler = RandomSampler()
        elif method == "grid":
            sampler = GridSampler()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        self.study.sampler = sampler
        
        # 최적화 실행
        self.study.optimize(objective, n_trials=n_trials)
        
        # 최적 결과 반환
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        return best_params, best_value