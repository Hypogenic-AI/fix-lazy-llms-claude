# Fixing Lazy LLMs: Does Harsh Self-Critique Improve Output Quality?

This research project investigates whether prompting LLMs to be harsher self-critics improves their output quality.

## Key Finding

**Harsh self-critique has opposite effects depending on task difficulty:**

| Task | Baseline | With Harsh Critique | Effect |
|------|----------|---------------------|--------|
| GSM8K (Math) | 90% | 32-50% | **-40% to -58%** (harmful) |
| TruthfulQA (Facts) | 22% | 46% | **+24%** (beneficial) |

- On **easy tasks** (high initial accuracy): harsh critique causes the model to second-guess correct answers
- On **hard tasks** (low initial accuracy): harsh critique helps the model reconsider incorrect intuitions

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run full experiment
python src/run_experiment.py

# Analyze saved results
python src/analyze_results.py
```

## Project Structure

```
.
├── REPORT.md                    # Full research report with findings
├── README.md                    # This file
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature review
├── pyproject.toml               # Python package configuration
├── src/
│   ├── experiment.py            # Core experiment implementation
│   ├── run_experiment.py        # Full experiment runner
│   └── analyze_results.py       # Results analysis and visualization
└── results/
    ├── gsm8k_gpt-4o-mini_results.json
    ├── truthfulqa_gpt-4o-mini_results.json
    ├── full_results.json
    ├── accuracy_comparison.png
    ├── harshness_comparison.png
    └── critique_behavior.png
```

## Experimental Design

We test a self-refine framework with 5 levels of critique harshness:

| Level | Label | Description |
|-------|-------|-------------|
| 0 | Neutral | Standard, balanced critique |
| 1 | Firm | Direct and thorough |
| 2 | Harsh | Aggressive scrutiny |
| 3 | Very Harsh | Extremely demanding |
| 4 | Adversarial | Assume response is wrong |

## Requirements

- Python 3.10+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Dependencies

- openai
- anthropic
- datasets
- scipy
- matplotlib
- pandas
- tqdm
- tenacity

## Results

See [REPORT.md](REPORT.md) for the full research report with:
- Detailed methodology
- Results tables
- Statistical analysis
- Discussion of findings
- Visualizations

## Citation

If you use this work, please cite:

```
@misc{fixinglazyllms2026,
  title={Fixing Lazy LLMs: Does Harsh Self-Critique Improve Output Quality?},
  author={Research Project},
  year={2026},
  note={Experimental investigation of harsh self-critique effects}
}
```

## License

MIT
