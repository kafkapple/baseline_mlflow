from pathlib import Path
import subprocess
import psutil
import time
import mlflow
import sys
import os

class MLflowServer:
    def __init__(self, tracking_uri: str, artifact_location: str, host: str = "127.0.0.1", port: int = 5000):
        self.tracking_uri = tracking_uri
        self.artifact_location = str(Path(artifact_location).absolute().as_posix())
        self.host = host
        self.port = port
        self.process = None
        self._setup_paths()
    
    def _setup_paths(self):
        """경로 설정"""
        self.artifact_path = Path(self.artifact_location)
        self.db_path = Path(self.tracking_uri.replace("sqlite:///", "")).absolute()
        self.artifact_path.mkdir(parents=True, exist_ok=True)
    
    def start(self, wait_for_ui: int = 15):
        """MLflow 서버 시작"""
        self.process = self._start_server()
        if wait_for_ui > 0:
            self._wait_for_server(timeout=wait_for_ui)
        return self
    
    def stop(self):
        """MLflow 서버 종료"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
    
    def _start_server(self):
        """MLflow 서버 시작"""
        python_path = sys.executable
        scripts_dir = Path(python_path).parent / "Scripts"
        
        # OS별 mlflow 실행 파일 경로
        if os.name == 'nt':  # Windows
            mlflow_path = scripts_dir / "mlflow.exe"
        else:  # Linux/Mac
            mlflow_path = scripts_dir / "mlflow"
            if not mlflow_path.exists():
                mlflow_path = Path("/usr/local/bin/mlflow")  # 일반적인 Linux 설치 경로
        
        if not mlflow_path.exists():
            raise FileNotFoundError(f"MLflow executable not found at {mlflow_path}")
        
        cmd = [
            str(mlflow_path), "server",
            "--backend-store-uri", f"sqlite:///{self.db_path.as_posix()}",
            "--default-artifact-root", str(self.artifact_path.as_posix()),
            "--host", self.host,
            "--port", str(self.port)
        ]
        
        env = os.environ.copy()
        env["MLFLOW_TRACKING_DIR"] = str(self.artifact_path.parent.as_posix())
        
        return subprocess.Popen(cmd, env=env)
    
    def _wait_for_server(self, timeout: int):
        """서버 준비 대기"""
        max_retries = 30
        retry_interval = 1
        
        for i in range(max_retries):
            try:
                import requests
                response = requests.get(f"http://127.0.0.1:{self.port}/health")
                if response.status_code == 200:
                    print(f"MLflow server is running at http://127.0.0.1:{self.port}")
                    return True
            except:
                if i < max_retries - 1:
                    time.sleep(retry_interval)
                    continue
        
        print("Warning: Could not verify MLflow server status")
        return False
    
    @property
    def ui_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def print_server_info(self, debug_logger=None):
        """서버 정보 출력"""
        info = f"""
{'='*50}
MLflow Server Information:
UI URL: {self.ui_url}
Tracking URI: {self.tracking_uri}
Artifact Location: {self.artifact_location}
{'='*50}
"""
        if debug_logger:
            debug_logger.print(info)
        else:
            print(info)