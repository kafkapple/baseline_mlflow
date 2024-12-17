from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
import sys

class DebugLogger:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
    
    def print(self, *args, **kwargs):
        if self.enabled:
            print(*args, **kwargs)
    
    def print_section(self, title: str):
        if self.enabled:
            print(f"\n{'='*20} {title} {'='*20}")
    
    def print_config(self, cfg: DictConfig, title: str = "Configuration"):
        if self.enabled:
            self.print_section(title)
            #print(OmegaConf.to_yaml(cfg))
    
    def print_system_info(self):
        if self.enabled:
            self.print_section("System Information")
            print(f"Python version: {sys.version}")
            print(f"Working directory: {Path.cwd()}")
            print(f"Script location: {Path(__file__).parent.parent}")

    def print_paths(self, paths: Dict[str, Path]):
        if self.enabled:
            self.print_section("Path Information")
            for name, path in paths.items():
                print(f"{name}: {path.absolute()}")
                print(f"Exists: {path.exists()}") 