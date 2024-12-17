from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from mlflow_utils.tracking import MLflowTracker
from optuna_utils.search import HyperparameterSearch
from models.base import get_model
from models.torch_models import SimpleNNModel
import mlflow
from data_utils.loader import load_data  # 데이터 로딩 함수 import
from typing import Any, Dict

def sweep_hyperparameters(cfg: DictConfig, tracker: MLflowTracker):
    """하이퍼파라미터 탐색 실행"""
    # 데이터 로드
    data = load_data(cfg.experiment.dataset, debug=cfg.debug.enabled)
    
    # 최적화 설정
    metric_name = cfg.search.objective.metric
    direction = cfg.search.objective.direction
    n_trials = cfg.search.n_trials
    
    print(f"\nStarting hyperparameter optimization:")
    print(f"Target metric: {metric_name} ({direction})")
    print(f"Number of trials: {n_trials}")
    print(f"Search space:")
    for param_name, param_config in cfg.model.sweep.items():
        print(f"  {param_name}: {param_config}")
    
    # 하이퍼파라미터 탐색 설정
    search = HyperparameterSearch(
        study_name=f"{cfg.model.name}_optimization",
        direction=direction,
        storage=f"sqlite:///{Path(__file__).parent/'optuna.db'}"
    )
    
    # 현재까지의 최고 성능
    current_best_score = float('-inf') if direction == 'maximize' else float('inf')
    
    # 부모 run 이름 설정 (모델명_데이터셋명)
    parent_run_name = f"{cfg.model.name}_{cfg.experiment.dataset.name}"
    
    # 부모 run 시작
    with tracker.start_run(run_name=parent_run_name) as parent_run:
        # 실험 설정 로깅
        mlflow.log_params({
            "model_name": cfg.model.name,
            "dataset_name": cfg.experiment.dataset.name,
            "n_trials": n_trials,
            "search_method": cfg.search.method,
            "optimization_metric": metric_name,
            "optimization_direction": direction
        })
        
        # 설정 파일 저장
        config_str = OmegaConf.to_yaml(cfg)
        config_path = Path("config.yaml")
        config_path.write_text(config_str)
        mlflow.log_artifact(str(config_path))
        config_path.unlink()
        
        def objective(trial):
            nonlocal current_best_score, metric_name
            
            # Trial 번호로 nested run 이름 생성
            trial_number = trial.number + 1
            trial_run_name = f"trial_{trial_number:03d}"
            
            print(f"\n{'='*50}")
            print(f"Trial {trial_number}/{n_trials}")
            
            # Nested run 시작
            with tracker.start_run(
                run_name=trial_run_name,
                nested=True
            ) as run:
                # 모델 생성 및 하이퍼파라미터 샘플링
                model = get_model(cfg.model.name, debug=cfg.debug.enabled)
                params = model.sample_params(trial, cfg.model.sweep)
                
                # 하이퍼파라미터 출력
                print("\nHyperparameters:")
                for name, value in params.items():
                    print(f"  {name}: {value}")
                
                # 학습 및 평가
                model.fit(data['train'])
                
                # validation 데이터가 있는 경우에만 평가
                if 'validation' in data:
                    val_data = data['validation']
                else:
                    # validation이 없으면 train 데이터로 평가
                    val_data = data['train']
                
                metrics = model.evaluate(
                    val_data,
                    average=cfg.experiment.metrics.average
                )
                
                # MLflow에 결과 기록
                mlflow.log_params({
                    "trial_number": trial_number,
                    **params
                })
                mlflow.log_metrics(metrics)
                
                # 현재 성능 출력
                score = metrics[metric_name]
                print(f"\nMetrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                # 최고 성능 업데이트
                is_better = (
                    score > current_best_score if direction == 'maximize'
                    else score < current_best_score
                )
                if is_better:
                    current_best_score = score
                    print(f"\n★ New best {metric_name}: {score:.4f}")
                    
                    mlflow.log_param("is_best", True)
                    
                    # 최고 성능 모델 저장
                    if isinstance(model, SimpleNNModel):
                        mlflow.pytorch.log_model(model.model, "best_model")
                    else:  # scikit-learn 모델
                        mlflow.sklearn.log_model(model.model, "best_model")
                    
                    # 예측 결과도 저장
                    if data['test'] is not None:
                        predictions_file = f"best_predictions.csv"
                        test_data = data['test']
                        # 이미지 데이터셋인 경우
                        if 'loader' in test_data:
                            test_X = test_data['dataset'].get_features()
                        else:
                            test_X = test_data['X']
                        model.save_predictions(test_X, predictions_file)
                        mlflow.log_artifact(predictions_file)
                        Path(predictions_file).unlink()
                
                print(f"{'='*50}\n")
                return score
        
        # 최적화 실행
        best_params, best_score = search.optimize(
            objective=objective,
            n_trials=n_trials,
            method=cfg.search.method
        )
        
        # 최종 결과를 부모 run에 기록
        mlflow.log_params({
            "best_" + k: v for k, v in best_params.items()
        })
        mlflow.log_metric(f"best_{metric_name}", best_score)
    
    # 최종 결과 출력
    print("\nOptimization finished")
    print(f"Best {metric_name}: {best_score:.4f}")
    print("Best parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")