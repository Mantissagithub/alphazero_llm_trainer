def check_terminal_state_prompt(question: str, current_response: str) -> str:
    return f"""question: {question}

current response: {current_response}

has this response reached a complete answer? respond with only 'yes' or 'no'.

answer:"""


def extract_answer_prompt(question: str, response: str) -> str:
    return f"""question: {question}

response: {response}

extract only the final numerical answer from the response. if there are multiple numbers, return only the final answer. if there is no clear answer, return 'none'.

examples:
- "the answer is 42" -> 42
- "5 * 8 = 40" -> 40
- "first we get 10, then multiply by 2 to get 20" -> 20
- "i don't know" -> none

final answer:"""


def get_baseline_perplexity_prompt(question: str, current_state: str) -> str:
    return f"""question: {question}

current partial answer: {current_state}

continue this response naturally.

continuation:"""
