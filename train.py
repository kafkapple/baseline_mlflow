import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
import mlflow
from pathlib import Path

from sklearn.model_selection import ParameterSampler
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import time
import sys
import os
import shutil  # 추가
from mlflow_utils.server import MLflowServer
from mlflow_utils.tracking import MLflowTracker
from mlflow_utils.environment import MLflowEnvironment
from utils.debug import DebugLogger
from sweep import sweep_hyperparameters  # sweep.py에서 가져오기

debug = DebugLogger(enabled=True)  # 디버그 출력 활성화

@hydra.main(
    config_path="conf",
    config_name="config",
    version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    """메인 실행 함수"""
    # 프로젝트 루트 디렉토리 설정
    project_root = Path(__file__).parent.absolute()
    
    # 설정에 경로 주입
    OmegaConf.update(cfg, "paths.root_dir", str(project_root))
    
    # 디버그 정보 출력
    debug = DebugLogger(enabled=cfg.debug.enabled)
    debug.print_system_info()
    debug.print_config(cfg)
    
    # MLflow 환경 설정
    env = MLflowEnvironment(project_root)  # Path 객체 전달
    if cfg.mlflow.reset_db:
        env.reset()
    env.setup()
    
    # 경로 정보 출력
    debug.print_paths({
        'Project Root': project_root,
        'MLflow DB': env.db_path,
        'Artifacts': env.artifacts_dir,
        'MLruns': env.mlruns_dir
    })
    
    # MLflow 서버 및 트래커 설정
    server = MLflowServer(
        tracking_uri=f"sqlite:///{env.db_path}",
        artifact_location=str(env.artifacts_dir),
        host=cfg.mlflow.host,
        port=cfg.mlflow.port
    )
    
    # start_server 설정 반영
    if cfg.mlflow.start_server:
        server.start(wait_for_ui=cfg.mlflow.wait_for_ui)
        server.print_server_info(debug)
    
    tracker = MLflowTracker(
        experiment_name=cfg.mlflow.experiment_name,
        auto_log_config=cfg.mlflow.auto_log if cfg.mlflow.auto_log.enabled else None
    )
    
    try:
        if cfg.mode == "train":
            train_model(cfg, tracker)
        elif cfg.mode == "sweep":
            sweep_hyperparameters(cfg, tracker)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")
    finally:
        if not cfg.mlflow.keep_server:
            server.stop()
            print("\nMLflow server stopped.")
        else:
            print(f"\nMLflow server is still running at {server.ui_url}")
            print("Use Ctrl+C to stop the program.")

def train_model(cfg: DictConfig, tracker: MLflowTracker):
    """일반 학습 실행"""
    from sweep import load_data
    from models.base import get_model
    
    try:
        data = load_data(cfg.experiment.dataset)
        model = get_model(cfg.model.name)
        
        # 데이터셋 정보 출력 및 로깅
        print("\nDataset Info:")
        dataset_info = {}
        for key, value in data['info'].items():
            print(f"{key}: {value}")
            # 리스트나 numpy 배열은 문자열로 변환
            if isinstance(value, (list, np.ndarray)):
                dataset_info[f"dataset_{key}"] = str(value)
            else:
                dataset_info[f"dataset_{key}"] = value
        
        # 메이터셋 크기 정보 추가
        dataset_info.update({
            "dataset_train_size": data['train']['X'].shape[0],
            "dataset_validation_size": data['validation']['X'].shape[0]
        })
        if data['test'] is not None:
            dataset_info["dataset_test_size"] = data['test']['X'].shape[0]
        
        print("\nDataset Sizes:")
        print(f"Train: {dataset_info['dataset_train_size']}")
        print(f"Validation: {dataset_info['dataset_validation_size']}")
        if 'dataset_test_size' in dataset_info:
            print(f"Test: {dataset_info['dataset_test_size']}")
        
        # 메인 run 시작
        with tracker.start_run(run_name=f"{cfg.model.name}_main") as parent_run:
            # 메인 run에 데이터셋 정보 로깅
            mlflow.log_params(dataset_info)
            
            # 학습용 nested run
            with tracker.start_run(
                run_name=f"{cfg.model.name}_training",
                nested=True
            ) as run:
                # 실험 설정 로깅
                mlflow.log_params({
                    "dataset_name": cfg.experiment.dataset.name,
                    "model_name": cfg.model.name,
                    "validation_size": cfg.experiment.dataset.split.validation_size,
                    "random_state": cfg.experiment.dataset.split.random_state,
                    "metric_average": cfg.experiment.metrics.average,
                    **dataset_info  # 데이터셋 상세 정보도 포함
                })
                
                # 학습
                print("\nTraining model...")
                model.fit(data['train'])
                
                # Validation 평가
                metrics = model.evaluate(
                    data['validation'],
                    average=cfg.experiment.metrics.average
                )
                
                # 결과 기록
                mlflow.log_metrics(metrics)
                
                # 결과 출력
                print("\nValidation Results:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                # 테스트 예측 (있는 경우)
                if data['test'] is not None:
                    print("\nGenerating test predictions...")
                    predictions_file = f"{cfg.model.name}_predictions.csv"
                    model.save_predictions(data['test']['X'], predictions_file)
                    mlflow.log_artifact(predictions_file)
                    print(f"Saved predictions to {predictions_file}")
    finally:
        tracker.cleanup()

if __name__ == "__main__":
    try:
        main()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
        print("프로그램을 종료합니다.")
        sys.exit(0) 