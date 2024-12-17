from pathlib import Path
import sys
import mlflow
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns

# test_mlflow 패키지를 import할 수 있도록 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_utils.dataset import DatasetManager
from mlflow_utils.tracking import MLflowTracker
from models.base import get_model

def run_csv_example():
    """CSV 데이터셋 예제 실행"""
    print("\n=== Running CSV Dataset Example ===")
    
    # 설정 생성
    config = OmegaConf.create({
        'name': 'csv',
        'csv': {
            'train_file': str(project_root / 'data/csv/train.csv'),
            'test_file': str(project_root / 'data/csv/test.csv'),
            'label_col': 'label',
            'feature_cols': ['feature1', 'feature2', 'feature3']
        },
        'split': {
            'validation_size': 0.2,
            'random_state': 42,
            'stratify': True
        }
    })
    
    # 데이터 로드
    dataset_manager = DatasetManager(config)
    data = dataset_manager.load_dataset()
    
    # ��이터 정보 출력
    print("\nDataset Info:")
    print(f"Train shape: {data['train']['X'].shape}")
    print(f"Validation shape: {data['validation']['X'].shape}")
    if data['test'] is not None:
        print(f"Test shape: {data['test']['X'].shape}")
    
    # 데이터 시각화
    plt.figure(figsize=(12, 4))
    for i, feature in enumerate(config.csv.feature_cols):
        plt.subplot(1, 3, i+1)
        sns.histplot(data['train']['X'][:, i], bins=30)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig(project_root / 'examples/csv_features_dist.png')
    plt.close()

def run_image_example():
    """이미지 데이터셋 예제 실행"""
    print("\n=== Running Image Dataset Example ===")
    
    # 설정 생성
    config = OmegaConf.create({
        'name': 'image_classification',
        'image': {
            'data_dir': str(project_root / 'data/images'),
            'metadata_file': str(project_root / 'data/metadata/image_metadata.csv'),
            'image_col': 'filename',
            'label_col': 'class'
        },
        'split': {
            'validation_size': 0.2,
            'random_state': 42,
            'stratify': True
        }
    })
    
    # 데이터 로드
    dataset_manager = DatasetManager(config)
    data = dataset_manager.load_dataset()
    
    # 데이터 정보 출력
    print("\nDataset Info:")
    print(f"Train shape: {data['train']['X'].shape}")
    print(f"Validation shape: {data['validation']['X'].shape}")
    if data['test'] is not None:
        print(f"Test shape: {data['test']['X'].shape}")
    
    # 샘플 이미지 시각화
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(data['train']['X'][i])
        plt.title(f"Class: {data['train']['y'][i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(project_root / 'examples/image_samples.png')
    plt.close()

def run_mlflow_example():
    """MLflow 로깅 예제 실행"""
    print("\n=== Running MLflow Logging Example ===")
    
    # MLflow 트래커 설정
    tracker = MLflowTracker(
        experiment_name="example_experiment",
        auto_log_config={
            'enabled': True,
            'sklearn': {'enabled': True},
            'pytorch': {'enabled': True}
        }
    )
    
    # 간단한 실험 실행
    with tracker.start_run(run_name="example_run"):
        # 데이터 로드
        config = OmegaConf.create({
            'name': 'csv',
            'csv': {
                'train_file': str(project_root / 'data/csv/train.csv'),
                'test_file': str(project_root / 'data/csv/test.csv'),
                'label_col': 'label',
                'feature_cols': ['feature1', 'feature2', 'feature3']
            },
            'split': {
                'validation_size': 0.2,
                'random_state': 42,
                'stratify': True
            }
        })
        
        dataset_manager = DatasetManager(config)
        data = dataset_manager.load_dataset()
        
        # 모델 학습
        model = get_model('logistic_regression')
        model.fit(data['train'])
        
        # 평가
        metrics = model.evaluate(data['validation'])
        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    # 예제 데이터가 있는지 확인
    if not (project_root / 'data/csv/train.csv').exists():
        print("Example datasets not found. Please run create_example_datasets.py first.")
        sys.exit(1)
    
    try:
        run_csv_example()
        run_image_example()
        run_mlflow_example()
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"\nError running examples: {e}")
        raise 