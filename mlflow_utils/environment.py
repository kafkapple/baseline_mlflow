from pathlib import Path
import shutil
import os
import mlflow
from typing import Union

class MLflowEnvironment:
    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir).absolute()
        self.mlruns_dir = self.root_dir / "mlruns"
        self.artifacts_dir = self.mlruns_dir / "artifacts"
        self.db_path = self.root_dir / "mlruns.db"
        self.optuna_db_path = self.root_dir / "optuna.db"
    
    def reset(self):
        """환경 초기화"""
        paths_to_clean = [
            self.mlruns_dir,
            self.artifacts_dir,
            self.db_path,
            self.optuna_db_path
        ]
        
        for path in paths_to_clean:
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                print(f"Deleted: {path}")
    
    def setup(self):
        """환경 설정"""
        self.mlruns_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # POSIX 형식으로 경로 ��일
        tracking_uri = f"sqlite:///{self.db_path.as_posix()}"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_TRACKING_DIR"] = self.mlruns_dir.as_posix()
        os.environ["MLFLOW_ARTIFACT_ROOT"] = self.artifacts_dir.as_posix()
        
        # MLflow 설정
        mlflow.set_tracking_uri(tracking_uri) 