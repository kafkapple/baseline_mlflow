from typing import Optional, Dict, Any, ContextManager, Generator
from contextlib import contextmanager
import mlflow

class MLflowTracker:
    def __init__(self, experiment_name: str, auto_log_config: Dict = None):
        self.experiment_name = experiment_name
        self.auto_log_config = auto_log_config or {}
        self.active_runs = []
        self.autolog_enabled = False
        self.parent_run_id = None  # 부모 run ID 저장용
        
        self._reset_state()
        self._setup_experiment()
        self._setup_autolog()
    
    def _reset_state(self):
        """MLflow 상태 초기화"""
        # 모든 autolog 비활성화
        self.disable_autolog()
        
        # 모든 active run 종료
        while mlflow.active_run():
            mlflow.end_run()
        self.active_runs = []
    
    def _setup_experiment(self):
        """실험 설정"""
        # 실험 생성 또는 가져오기
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
    
    def _setup_autolog(self):
        """autolog 설정"""
        if not self.auto_log_config:
            return
            
        if self.auto_log_config.get('enabled', False):
            self.autolog_enabled = True
            sklearn_config = self.auto_log_config.get('sklearn', {})
            if sklearn_config.get('enabled', False):
                mlflow.sklearn.autolog(
                    log_input_examples=sklearn_config.get('log_input_examples', True),
                    log_model_signatures=sklearn_config.get('log_model_signatures', True),
                    log_models=sklearn_config.get('log_models', True),
                    log_datasets=sklearn_config.get('log_datasets', True)
                )
        
        pytorch_config = self.auto_log_config.get('pytorch', {})
        if pytorch_config.get('enabled', False):
            mlflow.pytorch.autolog(
                log_every_n_epoch=pytorch_config.get('log_every_n_epoch', 1),
                log_every_n_step=pytorch_config.get('log_every_n_step', 50),
                log_models=pytorch_config.get('log_models', True),
                disable=False,
                exclusive=True,
                disable_for_unsupported_versions=False,
                silent=False
            )
    
    def get_or_create_parent_run(self, run_name: str) -> str:
        """동일한 이름의 parent run이 있으면 가져오고, 없으면 생성"""
        # 현재 실험의 모든 run 검색
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}' and tags.mlflow.parentRunId = ''",
            max_results=1
        )
        
        if runs:
            # 기존 parent run 사용
            self.parent_run_id = runs[0].info.run_id
            print(f"Found existing parent run: {run_name} (ID: {self.parent_run_id})")
        else:
            # 새로운 parent run 생성
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
                self.parent_run_id = run.info.run_id
                print(f"Created new parent run: {run_name} (ID: {self.parent_run_id})")
        
        return self.parent_run_id
    
    @contextmanager
    def start_run(self, run_name: str, nested: bool = False) -> Generator[mlflow.ActiveRun, None, None]:
        """새로운 run 시작 (context manager)"""
        try:
            if nested and not self.active_runs:
                # nested run인데 parent가 없는 경우, 기존 parent run 검색 또는 생성
                parent_run_id = self.get_or_create_parent_run(run_name.split('_trial_')[0])
                with mlflow.start_run(
                    experiment_id=self.experiment_id,
                    run_id=parent_run_id
                ) as parent_run:
                    self.active_runs.append(parent_run)
                    with mlflow.start_run(
                        experiment_id=self.experiment_id,
                        run_name=run_name,
                        nested=True
                    ) as run:
                        self.active_runs.append(run)
                        yield run
            else:
                # 일반적인 경우
                run = mlflow.start_run(
                    experiment_id=self.experiment_id,
                    run_name=run_name,
                    nested=nested
                )
                self.active_runs.append(run)
                yield run
        finally:
            if self.active_runs:
                mlflow.end_run()
                self.active_runs.pop()
    
    def cleanup(self):
        """모든 리소스 정리"""
        self._reset_state()
    
    def disable_autolog(self):
        """모든 autolog 비활성화"""
        mlflow.sklearn.autolog(disable=True)
        mlflow.pytorch.autolog(disable=True)