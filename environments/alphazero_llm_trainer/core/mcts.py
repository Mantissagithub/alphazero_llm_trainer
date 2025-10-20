import numpy as np
from typing import Optional, List, Dict
from .tree import TreeNode
from models import TeacherEnsemble, StudentModel, TerminalChecker
from rewards import CombinedRewardSystem
from prompts import generate_tokens_prompt
from config import get_training_config


class MCTSSystem:
    def __init__(
        self,
        teacher_ensemble: TeacherEnsemble,
        student_model: StudentModel,
        terminal_checker: TerminalChecker,
        reward_system: CombinedRewardSystem,
        exploration_constant: Optional[float] = None,
        num_iterations: Optional[int] = None,
        max_depth: Optional[int] = None
    ):
        self.teacher_ensemble = teacher_ensemble
        self.student_model = student_model
        self.terminal_checker = terminal_checker
        self.reward_system = reward_system

        config = get_training_config()
        self.exploration_constant = exploration_constant or config["mcts"]["exploration_constant"]
        self.num_iterations = num_iterations or config["mcts"]["num_iterations"]
        self.max_depth = max_depth or config["mcts"]["max_tree_depth"]
        self.right_tokens = config["mcts"]["right_tokens"]
        self.wrong_tokens = config["mcts"]["wrong_tokens"]
        self.epsilon = 0.3
        self.use_student_for_expansion = True

    def search(self, question: str, reference_answer: str) -> Dict:
        root = TreeNode(state="", parent=None, binary_code="")
        root.visits = 1

        best_terminal_node = None
        best_reward = -float('inf')

        for iteration in range(self.num_iterations):
            node = self._select(root)

            if node.depth < self.max_depth and not node.is_terminal():
                children = self._expand(node, question)
                if children:
                    node = np.random.choice(children)

            reward_dict = self._simulate(node, question, reference_answer)
            reward = reward_dict["total"]

            if node.is_terminal() and reward > best_reward:
                best_reward = reward
                best_terminal_node = node

            self._backpropagate(node, reward)

        if best_terminal_node:
            return {
                "best_response": best_terminal_node.state,
                "reward": best_reward,
                "visits": best_terminal_node.visits,
                "depth": best_terminal_node.depth,
                "binary_code": best_terminal_node.binary_code,
                "tree_size": root.get_subtree_size()
            }
        else:
            best_child = root.select_best_child(mode="visits")
            return {
                "best_response": best_child.state,
                "reward": best_child.value / max(best_child.visits, 1),
                "visits": best_child.visits,
                "depth": best_child.depth,
                "binary_code": best_child.binary_code,
                "tree_size": root.get_subtree_size()
            }

    def _select(self, node: TreeNode) -> TreeNode:
        while not node.is_leaf():
            node = node.select_best_child(mode="uct")
        return node

    def _expand(self, node: TreeNode, question: str) -> List[TreeNode]:
        children = []
        prompt = f"{question}\n{node.state}"

        student_token = self.student_model.generate(prompt, max_new_tokens=10, temperature=0.7)

        right_prompt = generate_tokens_prompt(question, node.state, "right", 5)
        wrong_prompt = generate_tokens_prompt(question, node.state, "wrong", 5)

        right_outputs = self.teacher_ensemble.generate_batch(right_prompt, num_generations=5, max_tokens=50)
        wrong_outputs = self.teacher_ensemble.generate_batch(wrong_prompt, num_generations=5, max_tokens=50)

        all_outputs = right_outputs + wrong_outputs

        from utils import calculate_embedding_similarity
        similarities = []
        for output in all_outputs:
            sim = calculate_embedding_similarity(student_token, output, self.teacher_ensemble.client)
            similarities.append(sim)

            child = TreeNode(
                state=(node.state + " " + output).strip(),
                parent=node,
                direction="right" if sim > 0.7 else "wrong",
                binary_code=node.binary_code + ("1" if sim > 0.7 else "0")
            )
            child.student_similarity = sim
            node.add_child(child)
            children.append(child)

        if np.random.random() < self.epsilon:
            selected_child = np.random.choice(children)
        else:
            best_idx = np.argmax(similarities)
            selected_child = children[best_idx]

        return [selected_child]

    def _simulate(self, node: TreeNode, question: str, reference_answer: str) -> Dict:
        is_terminal = self.terminal_checker.is_terminal(question, node.state)
        node.terminal = is_terminal

        step_pre = self.reward_system.pre.calculate_reward(question, node.state, normalize=True)
        node.step_pre = step_pre

        accumulated_pre = step_pre
        if node.parent and hasattr(node.parent, 'accumulated_pre'):
            accumulated_pre += node.parent.accumulated_pre

        node.accumulated_pre = accumulated_pre

        if is_terminal:
            hre_reward = self.reward_system.hre.calculate_reward(question, node.state, terminal_only=True)
        else:
            hre_reward = 0.0

        reward_dict = {
            "total": 0.4 * hre_reward + 0.6 * accumulated_pre,
            "hre": hre_reward,
            "pre": step_pre,
            "accumulated_pre": accumulated_pre,
            "length_penalty": -0.01 * node.depth,
            "direction_bonus": 0.1 * node.binary_code.count('1') - 0.05 * node.binary_code.count('0')
        }

        reward_dict["total"] += reward_dict["length_penalty"] + reward_dict["direction_bonus"]

        return reward_dict

    def _backpropagate(self, node: TreeNode, reward: float):
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward

            if current.teacher_model_used:
                self.teacher_ensemble.update_model_weight(
                    current.teacher_model_used, reward, current.direction
                )

            current = current.parent

    def collect_trajectories(self, question, reference_answer):
        root = TreeNode(state="", parent=None, binary_code="")
        root.visits = 1
        trajectories = []

        for _ in range(self.num_iterations):
            node = self._select(root)
            state_before = node.state

            if node.depth < self.max_depth and not node.is_terminal():
                children = self._expand(node, question)
                if children:
                    node = np.random.choice(children)

            reward_dict = self._simulate(node, question, reference_answer)
            reward = reward_dict["total"]

            trajectories.append({
                "state": state_before,
                "next_state": node.state,
                "reward": reward,
                "question": question
            })

            self._backpropagate(node, reward)

        return trajectories

    def get_policy_distribution(self, node: TreeNode) -> np.ndarray:
        if not node.children:
            return np.array([])

        visits = np.array([child.visits for child in node.children])
        total_visits = visits.sum()

        if total_visits == 0:
            return np.ones(len(visits)) / len(visits)

        return visits / total_visits
