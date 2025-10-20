from typing import Dict, Optional
from .hre import HardRewardEstimator
from .pre import PerplexityRewardEstimator
from config import get_training_config


class CombinedRewardSystem:
    def __init__(
        self,
        hre: HardRewardEstimator,
        pre: PerplexityRewardEstimator,
        hre_weight: Optional[float] = None,
        pre_weight: Optional[float] = None
    ):
        self.hre = hre
        self.pre = pre

        config = get_training_config()
        self.hre_weight = hre_weight or config["rewards"]["hre_weight"]
        self.pre_weight = pre_weight or config["rewards"]["pre_weight"]

        total_weight = self.hre_weight + self.pre_weight
        self.hre_weight /= total_weight
        self.pre_weight /= total_weight

    def calculate_reward(
        self,
        question: str,
        response: str,
        path_length: Optional[int] = None,
        binary_code: Optional[str] = None
    ) -> Dict[str, float]:

        hre_reward = self.hre.calculate_reward(question, response, terminal_only=True)
        pre_reward = self.pre.calculate_reward(question, response, normalize=True)

        intermediate_pre = self.pre.calculate_reward(question, response, normalize=True) if response else 0.0

        length_penalty = 0.0
        if path_length is not None:
            length_penalty = -0.01 * path_length

        direction_bonus = 0.0
        if binary_code:
            right_count = binary_code.count('1')
            wrong_count = binary_code.count('0')
            direction_bonus = 0.1 * right_count - 0.05 * wrong_count

        combined_reward = (
            self.hre_weight * hre_reward +
            self.pre_weight * pre_reward +
            length_penalty +
            direction_bonus
        )

        return {
            "total": combined_reward,
            "hre": hre_reward,
            "pre": pre_reward,
            "intermediate_pre": intermediate_pre,
            "length_penalty": length_penalty,
            "direction_bonus": direction_bonus,
            "breakdown": {
                "hre_contribution": self.hre_weight * hre_reward,
                "pre_contribution": self.pre_weight * pre_reward
            }
        }

    def calculate_incremental_reward(
        self,
        question: str,
        previous_response: str,
        current_response: str
    ) -> float:

        prev_pre = self.pre.calculate_reward(question, previous_response, normalize=True)
        curr_pre = self.pre.calculate_reward(question, current_response, normalize=True)

        improvement = curr_pre - prev_pre
        return improvement

    def set_weights(self, hre_weight: float, pre_weight: float):
        total = hre_weight + pre_weight
        self.hre_weight = hre_weight / total
        self.pre_weight = pre_weight / total

    def calculate_accumulated_reward(self, path_nodes: list, question: str) -> Dict:
        config = get_training_config()
        discount = config["rewards"].get("pre_discount_factor", 0.95)

        total_pre = 0.0
        for i, node in enumerate(path_nodes):
            step_pre = self.pre.calculate_reward(question, node.state, normalize=True)
            total_pre += (discount ** i) * step_pre

        final_hre = self.hre.calculate_reward(question, path_nodes[-1].state, terminal_only=True)

        return {
            "total": self.hre_weight * final_hre + self.pre_weight * total_pre,
            "hre": final_hre,
            "accumulated_pre": total_pre
        }

