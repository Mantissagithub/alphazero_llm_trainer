def generate_right_token_prompt(question: str, current_state: str, num_tokens: int = 5) -> str:
    if current_state.strip():
        return f"""question: {question}

current response: {current_state}

continue the response in the right direction. generate the next {num_tokens} tokens that move towards the correct answer.

next tokens:"""
    else:
        return f"""question: {question}

generate the first {num_tokens} tokens to start solving this problem in the right direction.

response:"""


def generate_wrong_token_prompt(question: str, current_state: str, num_tokens: int = 5) -> str:
    if current_state.strip():
        return f"""question: {question}

current response: {current_state}

continue the response in the wrong direction. generate the next {num_tokens} tokens that lead away from the correct answer or introduce errors.

next tokens:"""
    else:
        return f"""question: {question}

generate the first {num_tokens} tokens that would start solving this problem incorrectly or inefficiently.

response:"""


def generate_tokens_prompt(question: str, current_state: str, direction: str, num_tokens: int = 5) -> str:
    if direction.lower() == "right":
        return generate_right_token_prompt(question, current_state, num_tokens)
    elif direction.lower() == "wrong":
        return generate_wrong_token_prompt(question, current_state, num_tokens)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'right' or 'wrong'")
