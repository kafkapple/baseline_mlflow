import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import json

def create_directory_structure():
    """디렉토리 구조 생성"""
    base_dir = Path(__file__).parent.parent
    dirs = [
        base_dir / 'data/train',
        base_dir / 'data/test'
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {d}")

def create_image_dataset(n_classes=16, n_samples_per_class=100, n_test=500):
    """이미지 데이터셋 생성"""
    base_dir = Path(__file__).parent.parent / 'data'
    
    def create_dummy_image(size=(64, 64), class_idx=0):
        img = np.ones((*size, 3), dtype=np.uint8) * 128
        color = np.array([(class_idx * 47) % 255, 
                         (class_idx * 97) % 255, 
                         (class_idx * 157) % 255])
        cv2.circle(img, (size[0]//2, size[1]//2), size[0]//3, color.tolist(), -1)
        return img
    
    # 학습 이미지 생성
    train_data = []
    for class_idx in range(n_classes):
        for sample_idx in range(n_samples_per_class):
            img_filename = f"train_{class_idx:02d}_{sample_idx:04d}.jpg"
            img = create_dummy_image(class_idx=class_idx)
            cv2.imwrite(str(base_dir / 'train' / img_filename), img)
            train_data.append({
                'ID': img_filename,
                'target': class_idx
            })
    
    # 테스트 이미지 생성
    test_data = []
    for i in range(n_test):
        img_filename = f"test_{i:04d}.jpg"
        img = create_dummy_image(class_idx=i % n_classes)
        cv2.imwrite(str(base_dir / 'test' / img_filename), img)
        test_data.append({
            'ID': img_filename,
            'target': 0  # sample_submission용 초기값
        })
    
    # CSV 파일 생성
    pd.DataFrame(train_data).to_csv(base_dir / 'train.csv', index=False)
    pd.DataFrame(test_data).to_csv(base_dir / 'sample_submission.csv', index=False)
    
    # 메타데이터 생성
    meta_data = [
        {'target': i, 'class_name': f'class_{i}'} 
        for i in range(n_classes)
    ]
    pd.DataFrame(meta_data).to_csv(base_dir / 'meta.csv', index=False)

if __name__ == "__main__":
    print("Creating example datasets...")
    create_directory_structure()
    create_image_dataset(
        n_classes=16,          # 클래스 수
        n_samples_per_class=100,  # 클래스당 학습 샘플 수
        n_test=500             # 테스트 샘플 수
    )
    print("\nDone! Example datasets have been created successfully.") 