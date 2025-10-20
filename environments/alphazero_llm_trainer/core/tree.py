import math
from typing import Optional, List


class TreeNode:
    def __init__(
        self,
        state: str,
        parent: Optional['TreeNode'] = None,
        direction: Optional[str] = None,
        binary_code: str = ""
    ):
        self.state = state
        self.parent = parent
        self.direction = direction
        self.binary_code = binary_code
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.reward = 0.0
        self.policy_prior = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        self.teacher_model_used = None
        self.student_similarity = 0.0
        self.accumulated_pre = 0.0  # Track accumulated perplexity reward
        self.step_pre = 0.0  # PRE at this specific step

    def uct_score(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def puct_score(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return self.policy_prior * exploration_constant

        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits + self.policy_prior

        exploitation = self.value / self.visits
        exploration = exploration_constant * self.policy_prior * (
            math.sqrt(self.parent.visits) / (1 + self.visits)
        )

        return exploitation + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return hasattr(self, 'terminal') and self.terminal

    def add_child(self, child: 'TreeNode'):
        self.children.append(child)

    def select_best_child(self, mode: str = "uct") -> 'TreeNode':
        if not self.children:
            return self

        if mode == "uct":
            return max(self.children, key=lambda c: c.uct_score())
        elif mode == "puct":
            return max(self.children, key=lambda c: c.puct_score())
        elif mode == "visits":
            return max(self.children, key=lambda c: c.visits)
        elif mode == "value":
            return max(self.children, key=lambda c: c.value / max(c.visits, 1))
        else:
            return max(self.children, key=lambda c: c.uct_score())

    def get_path_from_root(self) -> List['TreeNode']:
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_subtree_size(self) -> int:
        size = 1
        for child in self.children:
            size += child.get_subtree_size()
        return size

    def get_binary_code_path(self) -> str:
        return self.binary_code

    def __repr__(self) -> str:
        return f"TreeNode(state={self.state[:50]}..., visits={self.visits}, value={self.value:.3f}, depth={self.depth})"
