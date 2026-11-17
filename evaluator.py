"""Prompt Evaluation Runner: loads dataset, runs prompt templates, and computes aggregate accuracy."""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List
from metrics import evaluate_prediction


def default_model_runner(prompt: str) -> str:
    """Simulated LLM response generator for standalone offline benchmarking."""
    p = prompt.lower()
    if "capital of france" in p:
        return "Paris"
    if "25 * 16" in p or "25 x 16" in p:
        return "400"
    if "http status code" in p and "not found" in p:
        return "404 Not Found"
    if "primary language" in p and "android" in p:
        return "Kotlin"
    if "photosynthesis" in p:
        return "Plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
    if "time complexity" in p and "binary search" in p:
        return "O(log n)"
    return "Unknown answer"


class PromptEvaluator:
    def __init__(
        self,
        prompt_template: str = "Answer concisely: {question}",
        model_fn: Callable[[str], str] = default_model_runner,
        pass_threshold: float = 0.75
    ):
        self.prompt_template = prompt_template
        self.model_fn = model_fn
        self.pass_threshold = pass_threshold

    def load_dataset(self, dataset_path: str) -> List[Dict[str, str]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset must be a non-empty JSON list of {id, question, expected_output}.")
        return data

    def run(self, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        results = []
        total_em = 0.0
        total_f1 = 0.0
        total_sim = 0.0
        total_comp = 0.0
        passed_count = 0

        for idx, item in enumerate(dataset, start=1):
            qid = item.get("id", f"Q{idx}")
            question = item["question"]
            expected = item["expected_output"]

            rendered_prompt = self.prompt_template.format(question=question)
            predicted = self.model_fn(rendered_prompt)
            scores = evaluate_prediction(predicted, expected, threshold=self.pass_threshold)

            total_em += scores["exact_match"]
            total_f1 += scores["token_f1"]
            total_sim += scores["similarity"]
            total_comp += scores["composite_score"]
            if scores["passed"]:
                passed_count += 1

            results.append({
                "id": qid,
                "question": question,
                "expected_output": expected,
                "predicted_output": predicted,
                "scores": scores
            })

        n = len(dataset)
        summary = {
            "total_samples": n,
            "passed_samples": passed_count,
            "accuracy_pct": round((passed_count / n) * 100, 2),
            "avg_exact_match": round(total_em / n, 4),
            "avg_token_f1": round(total_f1 / n, 4),
            "avg_similarity": round(total_sim / n, 4),
            "avg_composite_score": round(total_comp / n, 4)
        }

        return {
            "prompt_template": self.prompt_template,
            "summary": summary,
            "results": results
        }
