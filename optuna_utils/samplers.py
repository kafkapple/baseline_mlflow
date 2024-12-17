from typing import Dict, Any
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler

class SamplerFactory:
    @staticmethod
    def create_sampler(method: str, **kwargs) -> optuna.samplers.BaseSampler:
        if method == "tpe":
            return TPESampler(
                seed=kwargs.get('seed', 42),
                n_startup_trials=kwargs.get('n_startup_trials', 10),
                consider_prior=True,
                consider_magic_clip=True,
                consider_endpoints=True
            )
        elif method == "random":
            return RandomSampler(seed=kwargs.get('seed', 42))
        elif method == "grid":
            if 'search_space' not in kwargs:
                raise ValueError("Grid search requires search_space parameter")
            return GridSampler(kwargs['search_space'])
        else:
            raise ValueError(f"Unknown sampling method: {method}") 
        
