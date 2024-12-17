import pandas as pd
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from data_utils.image_processor import ImagePreprocessor
# 데이터 경로 설정

data_dir = project_root / "data"
image_dir = data_dir / "images"

# 데이터 로드
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")

# 이미지 전처리기 초기화
preprocessor = ImagePreprocessor(
    target_size=(224, 224),
    feature_extractor='resnet'  # 'basic', 'hog', 'resnet' 중 선택
)

# 학습 데이터 특징 추출
train_images = [str(image_dir / 'train' / filename) for filename in train_df['ID']]
X_train = preprocessor.fit_transform(train_images)

# 테스트 데이터 특징 추출
test_images = [str(image_dir / 'test' / filename) for filename in test_df['ID']]
X_test = preprocessor.transform(test_images)

print(f"Train features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}") 