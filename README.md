# Prompt Evaluation System with Scoring Metrics

A modular Python evaluation harness for benchmarking LLM prompts against curated **Input Datasets** and **Expected Outputs** using **Exact Match (EM)**, **Token-Level F1**, **Sequence Similarity**, and **Overall Accuracy %**.

## Project Structure
```text
prompt-evaluation-system/
├── app.py                  # CLI evaluation runner & report exporter
├── evaluator.py            # Dataset loader, prompt renderer, and batch scorer
├── metrics.py              # Exact Match, Token F1, Similarity, and Composite metrics
├── data/
│   └── eval_dataset.json   # Input questions and expected ground-truth outputs
├── tests/
│   └── test_evaluator.py   # Automated unit tests for scoring & accuracy metrics
├── requirements.txt        # Zero external dependencies (Python 3.8+ stdlib)
├── .gitignore
└── README.md
```

## Quick Start
```bash
# Run unit tests
python -m unittest discover -s tests -v

# Execute prompt evaluation benchmark
python app.py --dataset data/eval_dataset.json --report eval_report.json
```
