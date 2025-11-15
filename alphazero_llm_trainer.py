import verifiers as vf
from alphazero_llm_trainer.core.environment import AlphaZeroLLMEnvironment


def load_environment(**kwargs) -> vf.Environment:
    return AlphaZeroLLMEnvironment(**kwargs)

__all__ = ["load_environment"]
