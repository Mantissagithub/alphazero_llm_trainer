# # Lazy imports for heavy dependencies
# def __getattr__(name):
#     if name == 'HardRewardEstimator':
#         from .hre import HardRewardEstimator
#         return HardRewardEstimator
#     elif name == 'PerplexityRewardEstimator':
#         from .pre import PerplexityRewardEstimator
#         return PerplexityRewardEstimator
#     elif name == 'CombinedRewardSystem':
#         from .combined import CombinedRewardSystem
#         return CombinedRewardSystem
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from .hre import HardRewardEstimator
from .pre import PerplexityRewardEstimator
from .combined import CombinedRewardSystem

__all__ = ['HardRewardEstimator', 'PerplexityRewardEstimator', 'CombinedRewardSystem']
