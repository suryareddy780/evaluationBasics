import unittest
from metrics import exact_match_score, token_f1_score, similarity_score, evaluate_prediction
from evaluator import PromptEvaluator


class TestPromptEvaluationSystem(unittest.TestCase):
    def test_exact_match_normalization(self):
        self.assertEqual(exact_match_score("Paris.", "paris"), 1.0)
        self.assertEqual(exact_match_score("London", "Paris"), 0.0)

    def test_token_f1_partial_overlap(self):
        score = token_f1_score("404 Not Found error", "404 Not Found")
        self.assertGreater(score, 0.8)

    def test_similarity_score_bounds(self):
        sim = similarity_score("O(log n)", "O(log n)")
        self.assertEqual(sim, 1.0)

    def test_evaluate_prediction_pass_and_fail(self):
        good = evaluate_prediction("Kotlin", "Kotlin")
        self.assertTrue(good["passed"])
        self.assertEqual(good["composite_score"], 1.0)

        bad = evaluate_prediction("Java", "Python")
        self.assertFalse(bad["passed"])

    def test_full_dataset_evaluation_run(self):
        evaluator = PromptEvaluator()
        dataset = evaluator.load_dataset("data/eval_dataset.json")
        report = evaluator.run(dataset)
        self.assertEqual(report["summary"]["total_samples"], 6)
        self.assertEqual(report["summary"]["accuracy_pct"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
