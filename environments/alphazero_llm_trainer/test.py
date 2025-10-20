import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def test_configs():
    print("=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)

    from config import get_model_config, get_training_config, get_teacher_models

    model_cfg = get_model_config()
    training_cfg = get_training_config()
    free_models = get_teacher_models("free")
    production_models = get_teacher_models("production")

    print(f"\n✅ Model Config Loaded")
    print(f"   Student model: {model_cfg['student_model']['name']}")
    print(f"   Max seq length: {model_cfg['student_model']['max_seq_length']}")
    print(f"   LoRA rank: {model_cfg['student_model']['lora_config']['r']}")

    print(f"\n✅ Training Config Loaded")
    print(f"   MCTS iterations: {training_cfg['mcts']['num_iterations']}")
    print(f"   Exploration constant: {training_cfg['mcts']['exploration_constant']}")
    print(f"   HRE weight: {training_cfg['rewards']['hre_weight']}")
    print(f"   PRE weight: {training_cfg['rewards']['pre_weight']}")

    print(f"\n✅ Teacher Models Loaded")
    print(f"   Free tier models: {len(free_models)}")
    print(f"   Production tier models: {len(production_models)}")
    print(f"   First free model: {free_models[0]['name']}")


def test_environment_loading():
    print("\n" + "=" * 60)
    print("Testing Verifiers Environment Loading")
    print("=" * 60)

    try:
        import verifiers as vf
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        env = vf.load_environment("alphazero-llm-trainer", tier="free", use_student_model=False)
        print("\n✅ Environment loaded via Verifiers")
        print(f"   Type: {type(env).__name__}")
        print(f"   Base class: {type(env).__bases__[0].__name__}")

        print(f"\n✅ Dataset loaded")
        print(f"   Type: {type(env.dataset).__name__}")
        print(f"   Number of samples: {len(env.dataset)}")

        if len(env.dataset) > 0:
            first_example = env.dataset[0]
            print(f"\n✅ First example structure:")
            print(f"   Keys: {list(first_example.keys())}")
            print(f"   Question: {first_example['question'][:80]}...")
            print(f"   Answer type: {type(first_example['answer'])}")

        print(f"\n✅ Rubric loaded")
        print(f"   Type: {type(env.rubric).__name__}")

        # Correct attribute names from Verifiers documentation
        if hasattr(env.rubric, 'reward_funcs'):
            print(f"   Number of functions: {len(env.rubric.reward_funcs)}")

        if hasattr(env.rubric, 'reward_weights'):
            print(f"   Weights: {env.rubric.reward_weights}")

        if hasattr(env.rubric, 'parallelize_scoring'):
            print(f"   Parallel scoring: {env.rubric.parallelize_scoring}")

        # Test a single rollout
        if os.environ.get("OPENROUTER_API_KEY"):
            print(f"\n✅ Testing single rollout...")
            sample = env.dataset[0]

            try:
                result = env.rollout_sync(
                    client=client,
                    model="google/gemini-2.0-flash-exp:free",
                    prompt=sample['question'] if 'question' in sample else sample['prompt'],
                    answer=sample.get('answer', ''),
                    info=sample.get('info', {}),
                    sampling_args={"max_tokens": 512, "temperature": 0.7}
                )

                print(f"   Completion received: {len(result[0])} messages")
                print(f"   State keys: {list(result[1].keys())}")

                if 'rewards' in result[1]:
                    print(f"   Total reward: {result[1]['rewards'].get('total', 'N/A')}")

            except Exception as e:
                print(f"   ⚠️ Rollout test failed: {e}")
        else:
            print(f"\n⚠️  Skipping rollout test (no API key)")

    except Exception as e:
        print(f"\n❌ Error loading environment: {e}")
        import traceback
        traceback.print_exc()


def test_direct_import():
    print("\n" + "=" * 60)
    print("Testing Direct Environment Import")
    print("=" * 60)

    try:
        from core.environment import AlphaZeroLLMEnvironment

        env = AlphaZeroLLMEnvironment(tier="free", use_student_model=False)
        print("\n✅ Environment created directly")
        print(f"   Type: {type(env).__name__}")
        print(f"   Has dataset: {hasattr(env, 'dataset')}")
        print(f"   Has rubric: {hasattr(env, 'rubric')}")
        print(f"   Dataset size: {len(env.dataset)}")

    except Exception as e:
        print(f"\n❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()


def test_dataset_format():
    print("\n" + "=" * 60)
    print("Testing Dataset Format")
    print("=" * 60)

    try:
        from utils import load_gsm8k_as_hf_dataset

        dataset = load_gsm8k_as_hf_dataset("train", num_samples=5)
        print("\n✅ HF Dataset loaded")
        print(f"   Type: {type(dataset).__name__}")
        print(f"   Size: {len(dataset)}")
        print(f"   Column names: {dataset.column_names}")

        example = dataset[0]
        print(f"\n✅ Example structure:")
        print(f"   Question: {example['question'][:80]}...")
        print(f"   Answer preview: {str(example['answer'])[:80]}...")
        print(f"   Info keys: {list(example['info'].keys())}")

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


def test_rubric_functions():
    print("\n" + "=" * 60)
    print("Testing Rubric Functions")
    print("=" * 60)

    try:
        from rewards.rubric_functions import hre_reward_function, pre_reward_function

        print("\n✅ Rubric functions imported")
        print(f"   hre_reward_function: {hre_reward_function.__name__}")
        print(f"   pre_reward_function: {pre_reward_function.__name__}")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("\n⚠️  Skipping reward calculation (no API key)")
            return

        prompt = [{"role": "user", "content": "What is 2 + 2?"}]
        completion = [{"role": "assistant", "content": "2 + 2 = 4\n#### 4"}]
        answer = "#### 4"
        info = {"reference": "#### 4"}

        print("\n✅ Testing HRE reward (may take a moment)...")
        try:
            hre_score = hre_reward_function(prompt, completion, answer, info)
            print(f"   HRE Score: {hre_score}")
        except Exception as e:
            print(f"   ⚠️  HRE calculation skipped: {str(e)[:100]}")

        print("\n✅ Testing PRE reward (may take a moment)...")
        try:
            pre_score = pre_reward_function(prompt, completion, {})
            print(f"   PRE Score: {pre_score}")
        except Exception as e:
            print(f"   ⚠️  PRE calculation skipped: {str(e)[:100]}")

    except Exception as e:
        print(f"\n❌ Error testing rubric functions: {e}")
        import traceback
        traceback.print_exc()



def test_prompts():
    print("\n" + "=" * 60)
    print("Testing Prompt Generation")
    print("=" * 60)

    from prompts import generate_tokens_prompt, check_terminal_state_prompt, extract_answer_prompt

    question = "What is 5 * 8?"
    current_state = "Let me calculate"

    right_prompt = generate_tokens_prompt(question, current_state, "right", 5)
    wrong_prompt = generate_tokens_prompt(question, current_state, "wrong", 5)
    terminal_prompt = check_terminal_state_prompt(question, "5 * 8 = 40")
    extract_prompt = extract_answer_prompt(question, "The answer is 40")

    print("\n✅ Right direction prompt generated")
    print(f"   Length: {len(right_prompt)} chars")

    print("\n✅ Wrong direction prompt generated")
    print(f"   Length: {len(wrong_prompt)} chars")

    print("\n✅ Terminal check prompt generated")
    print(f"   Length: {len(terminal_prompt)} chars")

    print("\n✅ Answer extraction prompt generated")
    print(f"   Length: {len(extract_prompt)} chars")


def test_utils():
    print("\n" + "=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)

    from utils import extract_numerical_answer, normalize_answer, compare_answers

    test_cases = [
        "The answer is 42",
        "5 * 8 = 40",
        "First we get 10, then multiply by 2 to get 20",
        "#### 100",
        "No answer here"
    ]

    print("\n✅ Testing answer extraction:")
    for text in test_cases:
        answer = extract_numerical_answer(text)
        print(f"   '{text[:40]}...' -> {answer}")

    print("\n✅ Testing answer comparison:")
    print(f"   42 == 42.0 -> {compare_answers(42, 42.0)}")
    print(f"   '42' == 42 -> {compare_answers('42', 42)}")
    print(f"   40 == 42 -> {compare_answers(40, 42)}")


def test_tree_node():
    print("\n" + "=" * 60)
    print("Testing Tree Node")
    print("=" * 60)

    from core import TreeNode

    root = TreeNode(state="", parent=None, binary_code="")
    root.visits = 10
    root.value = 5.0

    child1 = TreeNode(state="Step 1", parent=root, direction="right", binary_code="1")
    child1.visits = 3
    child1.value = 2.0

    child2 = TreeNode(state="Step 2", parent=root, direction="wrong", binary_code="0")
    child2.visits = 2
    child2.value = 0.5

    root.add_child(child1)
    root.add_child(child2)

    print("\n✅ Tree structure created")
    print(f"   Root visits: {root.visits}")
    print(f"   Root value: {root.value}")
    print(f"   Number of children: {len(root.children)}")

    print("\n✅ UCT scores:")
    print(f"   Child 1 (right): {child1.uct_score():.3f}")
    print(f"   Child 2 (wrong): {child2.uct_score():.3f}")

    best_child = root.select_best_child(mode="uct")
    print(f"\n✅ Best child selected: {best_child.direction} direction")


def test_api_connection():
    print("\n" + "=" * 60)
    print("Testing API Connection")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("\n❌ OPENROUTER_API_KEY not found in environment")
        return

    print(f"\n✅ API key found (length: {len(api_key)})")
    print(f"   API Key: {api_key[:4]}...{api_key[-4:]}")

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        response = client.chat.completions.create(
            model="nvidia/nemotron-nano-9b-v2:free",
            messages=[{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
            max_tokens=10
        )

        print(f"   ✅ API connection successful")
        print(f"   Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"   ❌ API connection failed: {e}")


def test_no_step_method():
    print("\n" + "=" * 60)
    print("Testing SingleTurnEnv Architecture")
    print("=" * 60)

    try:
        from core.environment import AlphaZeroLLMEnvironment

        env = AlphaZeroLLMEnvironment(tier="free", use_student_model=False)

        print("\n✅ Verifying SingleTurnEnv architecture:")
        print(f"   Has step() method: {hasattr(env, 'step')}")
        print(f"   Has verify() method: {hasattr(env, 'verify')}")
        print(f"   Has dataset attribute: {hasattr(env, 'dataset')}")
        print(f"   Has rubric attribute: {hasattr(env, 'rubric')}")

        if not hasattr(env, 'step'):
            print("\n   ✅ Correct: No step() method (uses Rubric instead)")
        else:
            print("\n   ⚠️  Warning: step() method exists (should not)")

        if hasattr(env, 'dataset') and hasattr(env, 'rubric'):
            print("   ✅ Correct: Has dataset and rubric (SingleTurnEnv pattern)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_mcts_components():
    print("\n" + "=" * 60)
    print("Testing MCTS Components (External)")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\n⚠️  Skipping MCTS test (no API key)")
        return

    try:
        from openai import OpenAI
        from models import TeacherEnsemble, TerminalChecker
        from core.mcts import MCTSSystem
        from rewards import HardRewardEstimator
        from utils import normalize_answer

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        print("\n✅ Components initialized:")
        teacher = TeacherEnsemble(client, tier="free")
        print(f"   Teacher ensemble: {len(teacher.active_models)} models")

        terminal_checker = TerminalChecker(client)
        print(f"   Terminal checker: {type(terminal_checker).__name__}")

        correct_answer = normalize_answer("#### 4")
        hre = HardRewardEstimator(terminal_checker, correct_answer)
        print(f"   HRE reward estimator: {type(hre).__name__}")

        mcts = MCTSSystem(
            teacher_ensemble=teacher,
            student_model=None,
            terminal_checker=terminal_checker,
            reward_system=hre
        )
        print(f"   MCTS system: {type(mcts).__name__}")
        print("\n✅ MCTS components ready (external to environment)")

    except Exception as e:
        print(f"\n❌ Error initializing MCTS: {e}")
        import traceback
        traceback.print_exc()


def run_all_tests():
    print("\n" + "=" * 60)
    print("ALPHAZERO LLM TRAINER - VERIFIERS INTEGRATION TESTS")
    print("=" * 60)

    test_configs()
    test_dataset_format()
    test_direct_import()
    test_environment_loading()
    test_rubric_functions()
    test_no_step_method()
    test_prompts()
    test_utils()
    test_tree_node()
    test_api_connection()
    test_mcts_components()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

