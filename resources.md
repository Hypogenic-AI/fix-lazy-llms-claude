# Resources Catalog

This document catalogs all resources gathered for the "Fixing Lazy LLMs" research project.

## Summary

| Resource Type | Count | Location |
|--------------|-------|----------|
| Papers (PDFs) | 23 | `papers/` |
| Datasets | 4 | `datasets/` |
| Code Repositories | 3 | `code/` |

---

## Papers

**Total papers downloaded: 23**

| Title | Authors | Year | File | Relevance |
|-------|---------|------|------|-----------|
| Mind Your Tone | Dobariya & Kumar | 2025 | papers/2510.04950_prompt_politeness.pdf | **HIGH** - Rude prompts improve performance |
| Self-Refine | Madaan et al. | 2023 | papers/2303.17651_self_refine.pdf | **HIGH** - Core self-critique method |
| Yes-Men to Truth-Tellers | Chen et al. | 2024 | papers/2409.01658_sycophancy_pinpoint_tuning.pdf | **HIGH** - Sycophancy mechanism |
| Should We Respect LLMs | Yin et al. | 2024 | papers/2402.14531_respect_llms_politeness.pdf | **HIGH** - Cross-lingual tone effects |
| Does Tone Change Answer | Cai et al. | 2025 | papers/2512.12812_tone_effects.pdf | Medium - Multi-model comparison |
| CriticBench | Lin et al. | 2024 | papers/2402.14809_criticbench.pdf | Medium - Evaluation benchmark |
| ELEPHANT | Cheng et al. | 2025 | papers/2505.13995_elephant_sycophancy.pdf | Medium - Social sycophancy |
| SycEval | Fanous et al. | 2025 | papers/2502.08177_syceval.pdf | Medium - Sycophancy rates |
| Distilled Self-Critique | Gallego | 2023 | papers/2312.01957_distilled_self_critique.pdf | Medium - Self-critique theory |
| Linear Probe Penalties | Papadatos & Freedman | 2024 | papers/2412.00967_linear_probe_sycophancy.pdf | Medium - Sycophancy mitigation |
| Training with Language Feedback | Scheurer et al. | 2022 | papers/2204.14146_language_feedback.pdf | Medium - Feedback training |
| Self-Improve from Feedback | Hu et al. | 2024 | papers/2406.07168_self_improve_feedback.pdf | Medium - SRT method |
| RRHF | Yuan et al. | 2023 | papers/2304.05302_rrhf.pdf | Low - Alignment method |
| Goal-oriented Prompting | Li et al. | 2024 | papers/2401.14043_goal_oriented_prompting.pdf | Low - Survey |
| Code Prompting | Puerto et al. | 2024 | papers/2401.10065_code_prompting.pdf | Low - Code prompts |
| Self-Critique Summarization | Hu et al. | 2025 | papers/2512.05387_self_critique_summarization.pdf | Low - Summarization |
| Self-Critique Explanations | Wang & Atanasova | 2025 | papers/2505.22823_self_critique_explanations.pdf | Low - Explanations |
| Intrinsic Self-Critique | Bohnet et al. | 2025 | papers/2512.24103_intrinsic_self_critique.pdf | Low - Planning |
| Dancing with Critiques | Li et al. | 2025 | papers/2503.17363_dancing_critiques.pdf | Low - Stepwise critique |
| Double-Checker | Xu et al. | 2025 | papers/2506.21285_double_checker.pdf | Low - Fine-tuning |
| LLM as Judge | Wang et al. | 2025 | papers/2502.06193_llm_as_judge.pdf | Low - Evaluation |
| Sycophancy Trust | Sun & Wang | 2025 | papers/2502.10844_sycophancy_trust.pdf | Low - User study |
| Persona Prompts | Zhang et al. | 2025 | papers/2507.22171_persona_prompts.pdf | Low - Jailbreaks |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 4**

| Name | Source | Size | Task | Location | License |
|------|--------|------|------|----------|---------|
| TruthfulQA | HuggingFace | 817 val | Factual MCQ | datasets/truthfulqa/ | Apache 2.0 |
| GSM8K | HuggingFace | 7.5K train, 1.3K test | Math reasoning | datasets/gsm8k/ | MIT |
| MMLU | HuggingFace | 14K test, 100K aux | Multitask MCQ | datasets/mmlu/ | MIT |
| OASST Sample | HuggingFace | 5K | Dialogue | datasets/oasst_sample/ | Apache 2.0 |

See `datasets/README.md` for download instructions and detailed descriptions.

---

## Code Repositories

**Total repositories cloned: 3**

| Name | URL | Purpose | Location | Stars |
|------|-----|---------|----------|-------|
| Self-Refine | github.com/madaan/self-refine | Self-feedback refinement | code/self-refine/ | ~1.2K |
| Sycophancy-Pinpoint | github.com/yellowtownhz/sycophancy-interpretability | Sycophancy analysis/tuning | code/sycophancy-pinpoint/ | New |
| RepEng | github.com/vgel/repeng | Representation engineering | code/representation-engineering/ | ~1.5K |

See `code/README.md` for usage instructions.

---

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service unavailable** - proceeded with arXiv API search
2. **Search queries**:
   - "LLM prompt engineering output quality"
   - "LLM self-critique iterative refinement"
   - "prompt tone effect LLM behavior"
   - "LLM sycophancy problem"
   - "self-refine language model feedback"
3. **Sources searched**: arXiv (primary), HuggingFace (datasets), GitHub (code)

### Selection Criteria
- **Papers**: Direct relevance to hypothesis (tone effects, self-critique, sycophancy)
- **Datasets**: Standard benchmarks used in related papers
- **Code**: Official implementations of key methods

### Challenges Encountered
1. **Contradictory findings**: Papers disagree on whether rude prompts help or hurt
2. **Model-specific effects**: Results vary significantly by LLM generation
3. **No direct "harsh critic" studies**: Gap in literature for our hypothesis

### Gaps and Workarounds
- **No explicit "lazy LLM" dataset**: Using math reasoning (GSM8K) as proxy since LLMs show "everything looks good" behavior
- **No combined tone + self-critique study**: This is a research opportunity
- **Limited open evaluation code**: Will need to implement evaluation pipeline

---

## Recommendations for Experiment Design

### Primary Dataset
**GSM8K** - Math reasoning task where:
- LLMs struggle to identify errors in self-critique
- Self-Refine showed minimal gains (0.2%)
- Good test case for "lazy" evaluation behavior

### Baseline Methods
1. Standard single-shot generation
2. Chain-of-thought prompting
3. Self-Refine with neutral feedback prompt
4. Self-Refine with harsh/critical feedback prompt (**novel condition**)

### Evaluation Metrics
1. **Accuracy** on test set
2. **Self-critique precision** - does model identify actual errors?
3. **Response quality** judged by human or GPT-4
4. **Effort proxies** - response length, reasoning depth

### Code to Adapt/Reuse
1. **Self-Refine framework** - adapt for harsh critic experiments
2. **Sycophancy evaluation** - measure if harsh self-critique reduces sycophancy
3. **RepEng** - potentially steer models toward critical behavior

---

## Environment Setup

A fresh Python virtual environment has been created at `.venv/` with:
- Python 3.12
- pypdf (for PDF reading)
- requests, arxiv (for paper downloads)
- datasets, huggingface_hub (for data access)

Activate with:
```bash
source .venv/bin/activate
```

Install additional packages:
```bash
uv pip install <package-name>
```
