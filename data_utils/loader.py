from typing import Dict, Any
from omegaconf import DictConfig
from .dataset import DatasetManager

def load_data(data_config: DictConfig, debug: bool = False) -> Dict[str, Any]:
    """데이터 로드 및 전처리
    
    Args:
        data_config: 데이터 설정
        debug: 디버그 모드 여부
    
    Returns:
        Dict containing:
            - train: {'loader': train_loader, 'dataset': train_dataset} or {'X': X, 'y': y}
            - validation: {'loader': val_loader, 'dataset': val_dataset} or {'X': X, 'y': y}
            - test: {'loader': test_loader, 'dataset': test_dataset} or {'X': X, 'y': y}
            - info: Dataset metadata
    """
    if debug:
        print("\nLoading dataset...")
    
    dataset_manager = DatasetManager(data_config)
    data = dataset_manager.load_dataset()
    
    if debug:
        print(f"Dataset loaded: {data_config.name}")
        # 이미지 데이터셋인 경우
        if 'loader' in data['train']:
            print(f"Train size: {len(data['train']['dataset'])}")
            if 'validation' in data:
                print(f"Validation size: {len(data['validation']['dataset'])}")
            print(f"Test size: {len(data['test']['dataset'])}")
        # 일반 데이터셋인 경우
        else:
            print(f"Train size: {len(data['train']['X'])}")
            if 'validation' in data:
                print(f"Validation size: {len(data['validation']['X'])}")
            print(f"Test size: {len(data['test']['X'])}")
        
        if 'info' in data:
            print("\nDataset info:")
            for k, v in data['info'].items():
                print(f"  {k}: {v}")
    
    return data 