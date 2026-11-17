import argparse
import json
from pathlib import Path
from evaluator import PromptEvaluator


def main():
    parser = argparse.ArgumentParser(description="Prompt Evaluation System with Scoring Metrics")
    parser.add_argument("--dataset", type=str, default="data/eval_dataset.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--template", type=str, default="Answer concisely: {question}", help="Prompt template string")
    parser.add_argument("--report", type=str, default="eval_report.json", help="Output JSON report path")
    args = parser.parse_args()

    evaluator = PromptEvaluator(prompt_template=args.template)
    dataset = evaluator.load_dataset(args.dataset)
    report = evaluator.run(dataset)
    summary = report["summary"]

    print("=" * 72)
    print("  PROMPT EVALUATION SYSTEM - SCORING & ACCURACY REPORT")
    print(f"  Dataset: {args.dataset} ({summary['total_samples']} questions)")
    print(f"  Template: {args.template}")
    print("=" * 72)

    for row in report["results"]:
        sc = row["scores"]
        badge = "PASS" if sc["passed"] else "FAIL"
        print(f"[{badge}] {row['id']}: {row['question']}")
        print(f"       Expected : {row['expected_output']}")
        print(f"       Predicted: {row['predicted_output']}")
        print(f"       Metrics  : EM={sc['exact_match']:.2f} | F1={sc['token_f1']:.2f} | Sim={sc['similarity']:.2f} | Composite={sc['composite_score']:.2f}")
        print("-" * 72)

    print("AGGREGATE BENCHMARK SUMMARY:")
    print(f"  Overall Accuracy   : {summary['accuracy_pct']}% ({summary['passed_samples']}/{summary['total_samples']} passed)")
    print(f"  Avg Exact Match    : {summary['avg_exact_match']:.4f}")
    print(f"  Avg Token F1 Score : {summary['avg_token_f1']:.4f}")
    print(f"  Avg Similarity     : {summary['avg_similarity']:.4f}")
    print(f"  Composite Score    : {summary['avg_composite_score']:.4f}")
    print("=" * 72)

    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved detailed evaluation report to: {args.report}")


if __name__ == "__main__":
    main()
