# Downloaded Papers

This directory contains academic papers relevant to the "Fixing Lazy LLMs" research project.

## Papers by Category

### Prompt Tone/Politeness Effects

| # | Title | Authors | Year | File | Key Finding |
|---|-------|---------|------|------|-------------|
| 1 | Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy | Dobariya & Kumar | 2025 | `2510.04950_prompt_politeness.pdf` | Rude prompts outperform polite (80.8% → 84.8% on GPT-4o) |
| 2 | Does Tone Change the Answer? | Cai et al. | 2025 | `2512.12812_tone_effects.pdf` | Cross-model comparison of tone effects |
| 3 | Should We Respect LLMs? | Yin et al. | 2024 | `2402.14531_respect_llms_politeness.pdf` | Impolite hurts, but overly polite doesn't help |

### Self-Critique and Refinement

| # | Title | Authors | Year | File | Key Finding |
|---|-------|---------|------|------|-------------|
| 4 | Self-Refine: Iterative Refinement with Self-Feedback | Madaan et al. | 2023 | `2303.17651_self_refine.pdf` | ~20% avg improvement via self-feedback |
| 5 | Distilled Self-Critique of LLMs | Gallego | 2023 | `2312.01957_distilled_self_critique.pdf` | Bayesian interpretation of RLAIF |
| 6 | CriticBench | Lin et al. | 2024 | `2402.14809_criticbench.pdf` | Benchmark for critique-correct reasoning |
| 7 | Self-Critique for Faithful Summarization | Hu et al. | 2025 | `2512.05387_self_critique_summarization.pdf` | Self-critique reduces hallucinations |
| 8 | Self-Critique for NL Explanations | Wang & Atanasova | 2025 | `2505.22823_self_critique_explanations.pdf` | Improves explanation faithfulness |
| 9 | Intrinsic Self-Critique for Planning | Bohnet et al. | 2025 | `2512.24103_intrinsic_self_critique.pdf` | Self-critique without external verifier |
| 10 | Dancing with Critiques | Li et al. | 2025 | `2503.17363_dancing_critiques.pdf` | Stepwise natural language self-critique |
| 11 | Double-Checker | Xu et al. | 2025 | `2506.21285_double_checker.pdf` | Self-critical fine-tuning for reasoning |

### Sycophancy

| # | Title | Authors | Year | File | Key Finding |
|---|-------|---------|------|------|-------------|
| 12 | From Yes-Men to Truth-Tellers | Chen et al. | 2024 | `2409.01658_sycophancy_pinpoint_tuning.pdf` | Only ~4% of heads control sycophancy |
| 13 | ELEPHANT: Social Sycophancy | Cheng et al. | 2025 | `2505.13995_elephant_sycophancy.pdf` | Defines social sycophancy as face-preservation |
| 14 | SycEval | Fanous et al. | 2025 | `2502.08177_syceval.pdf` | 58% sycophantic behavior across models |
| 15 | Be Friendly, Not Friends | Sun & Wang | 2025 | `2502.10844_sycophancy_trust.pdf` | Sycophancy-friendliness-trust dynamics |
| 16 | Linear Probe Penalties | Papadatos & Freedman | 2024 | `2412.00967_linear_probe_sycophancy.pdf` | Penalize sycophancy markers in reward model |

### Training with Feedback

| # | Title | Authors | Year | File | Key Finding |
|---|-------|---------|------|------|-------------|
| 17 | Training LMs with Language Feedback | Scheurer et al. | 2022 | `2204.14146_language_feedback.pdf` | NL feedback > comparison feedback |
| 18 | Teaching LMs to Self-Improve | Hu et al. | 2024 | `2406.07168_self_improve_feedback.pdf` | Self-Refinement Tuning (SRT) |
| 19 | RRHF | Yuan et al. | 2023 | `2304.05302_rrhf.pdf` | Rank responses for alignment |

### Prompt Engineering

| # | Title | Authors | Year | File | Key Finding |
|---|-------|---------|------|------|-------------|
| 20 | Goal-oriented Prompt Engineering | Li et al. | 2024 | `2401.14043_goal_oriented_prompting.pdf` | Survey of prompt engineering methods |
| 21 | Code Prompting | Puerto et al. | 2024 | `2401.10065_code_prompting.pdf` | Code prompts elicit conditional reasoning |

### Evaluation

| # | Title | Authors | Year | File | Key Finding |
|---|-------|---------|------|------|-------------|
| 22 | Can LLMs Replace Human Evaluators? | Wang et al. | 2025 | `2502.06193_llm_as_judge.pdf` | LLM-as-judge for SE tasks |

### Persona Effects

| # | Title | Authors | Year | File | Key Finding |
|---|-------|---------|------|------|-------------|
| 23 | Enhancing Jailbreaks via Persona Prompts | Zhang et al. | 2025 | `2507.22171_persona_prompts.pdf` | Persona prompts affect LLM behavior |

---

## Most Relevant for Research Hypothesis

1. **2510.04950_prompt_politeness.pdf** - Direct evidence rude prompts improve performance
2. **2303.17651_self_refine.pdf** - Core self-critique methodology
3. **2409.01658_sycophancy_pinpoint_tuning.pdf** - Understanding LLM "yes-men" behavior
4. **2402.14531_respect_llms_politeness.pdf** - Nuanced view of tone effects

---

## Reading Priority

### Must Read (Deep)
1. Self-Refine (2303.17651)
2. Mind Your Tone (2510.04950)
3. Yes-Men to Truth-Tellers (2409.01658)

### Should Read (Skim)
4. Should We Respect LLMs (2402.14531)
5. CriticBench (2402.14809)
6. SycEval (2502.08177)

### Can Reference
- Remaining papers for methodology and related work sections
