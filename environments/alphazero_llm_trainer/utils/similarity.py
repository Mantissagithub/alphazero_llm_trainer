import numpy as np
from openai import OpenAI
from typing import List, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch
from prompts.terminal_check import (
    check_terminal_state_prompt,
    extract_answer_prompt,
    get_baseline_perplexity_prompt
)


# Global embedding model for local similarity calculations (lazy loaded)
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading local embedding model (all-MiniLM-L6-v2)...")
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                _embedding_model = _embedding_model.cuda()
            print("✓ Local embedding model loaded")
        except Exception as e:
            print(f"Warning: Could not load sentence-transformers: {e}")
            _embedding_model = False  # Mark as failed to avoid repeated attempts
    return _embedding_model if _embedding_model is not False else None


def calculate_local_similarity(text1: str, text2: str) -> float:
    model = _get_embedding_model()
    if model is None:
        print("Warning: Local embedding model unavailable, using fallback similarity")
        return 0.5

    try:
        emb1 = model.encode([text1], convert_to_tensor=True)
        emb2 = model.encode([text2], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
        # Normalize from [-1, 1] to [0, 1]
        normalized_sim = (similarity + 1) / 2
        return float(normalized_sim)
    except Exception as e:
        print(f"Error calculating local similarity: {e}")
        return 0.5


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def calculate_embedding_similarity(
    text1: str,
    text2: str,
    client: OpenAI,
    model: str = "openai/text-embedding-3-small"
) -> float:
    try:
        try:
            response1 = client.embeddings.create(input=text1, model=model)
            embedding1 = response1.data[0].embedding
        except Exception as e:
            print(f"warning: failed to get embedding for text1: {e}")
            return 0.5

        try:
            response2 = client.embeddings.create(input=text2, model=model)
            embedding2 = response2.data[0].embedding
        except Exception as e:
            print(f"warning: failed to get embedding for text2: {e}")
            return 0.5

        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        cosine_sim = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        normalized_sim = (cosine_sim + 1) / 2

        return float(normalized_sim)

    except Exception as e:
        print(f"error calculating similarity: {e}")
        return 0.5


def batch_cosine_similarity(
    query_vector: np.ndarray,
    vectors: List[np.ndarray]
) -> np.ndarray:
    query_vector = np.array(query_vector).reshape(1, -1)
    vectors_matrix = np.array(vectors)
    similarities = cosine_similarity(query_vector, vectors_matrix)[0]
    return similarities


def get_embeddings_batch(
    texts: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-small"
) -> List[List[float]]:

    response = client.embeddings.create(input=texts, model=model)
    embeddings = [item.embedding for item in response.data]
    return embeddings


def find_most_similar(
    query: str,
    candidates: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-small"
) -> tuple:

    all_texts = [query] + candidates
    embeddings = get_embeddings_batch(all_texts, client, model)

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    similarities = batch_cosine_similarity(query_embedding, candidate_embeddings)
    best_idx = np.argmax(similarities)

    return best_idx, similarities[best_idx], candidates[best_idx]


class TerminalChecker:
    def __init__(self, client: OpenAI, model_name: Optional[str] = None):
        self.client = client
        if model_name is None:
            from config import get_model_config
            config = get_model_config()
            self.model_name = config['terminal_checker']['name']
        else:
            self.model_name = model_name

    def is_terminal(self, question: str, response: str) -> bool:
        prompt = check_terminal_state_prompt(question, response)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            answer = result.choices[0].message.content.strip().lower()
            return answer == "yes"
        except:
            return len(response.strip()) > 100

    def extract_answer(self, question: str, response: str) -> Optional[str]:
        prompt = extract_answer_prompt(question, response)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0
            )
            return result.choices[0].message.content.strip()
        except:
            return None

    def get_baseline_completion(self, question: str, current_state: str, max_tokens: int = 50) -> str:
        # Prompt is now imported at top
        prompt = get_baseline_perplexity_prompt(question, current_state)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return result.choices[0].message.content.strip()
        except:
            return ""
