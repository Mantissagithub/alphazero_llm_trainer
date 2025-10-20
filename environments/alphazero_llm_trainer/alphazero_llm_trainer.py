import verifiers as vf
from core.environment import AlphaZeroLLMEnvironment


def load_environment(**kwargs) -> vf.Environment:
    return AlphaZeroLLMEnvironment(**kwargs)
