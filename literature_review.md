# Literature Review: Fixing Lazy LLMs

## Research Hypothesis
Large language models (LLMs) tend to prefer easy or low-effort responses due to a lack of subjective judgment of good or bad; prompting LLMs to act as harsher critics or varying prompt tone (e.g., being rude) may improve their output quality. Investigating whether these approaches reduce "laziness" in LLMs and exploring alternative methods can lead to more effective LLM behavior.

---

## Research Area Overview

The hypothesis touches on several interconnected research areas:
1. **Prompt tone and politeness effects on LLM performance**
2. **LLM sycophancy** - tendency to agree with users over factual accuracy
3. **Self-critique and iterative refinement** - improving outputs through self-feedback
4. **LLM-as-a-judge** - using LLMs for evaluation and criticism

The literature reveals a complex picture: prompt tone does affect LLM performance, but the direction of effect varies by model and task. Self-critique mechanisms can improve outputs, though LLMs often struggle to identify their own errors. Sycophancy is a well-documented problem that may relate to "lazy" behavior.

---

## Key Papers

### 1. Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy
- **Authors**: Om Dobariya, Akhil Kumar (Penn State)
- **Year**: 2025
- **Source**: arXiv:2510.04950
- **Key Contribution**: Contrary to expectations, **rude prompts outperformed polite ones** on GPT-4o
- **Methodology**: 50 MCQs × 5 politeness levels = 250 prompts; 10 runs per condition
- **Key Results**:
  - Very Polite: 80.8% accuracy
  - Polite: 81.4%
  - Neutral: 82.2%
  - Rude: 82.8%
  - **Very Rude: 84.8%** (highest)
  - All differences statistically significant (p < 0.05)
- **Code Available**: Yes (GitHub)
- **Relevance**: **DIRECTLY SUPPORTS the research hypothesis** - rude/harsh prompts improve performance

### 2. Should We Respect LLMs? A Cross-Lingual Study on Prompt Politeness
- **Authors**: Ziqi Yin, Hao Wang, et al. (Waseda University)
- **Year**: 2024
- **Source**: arXiv:2402.14531
- **Key Contribution**: Cross-lingual study showing politeness effects vary by language and culture
- **Methodology**: 8 politeness levels across English, Chinese, Japanese; MMLU and bias detection
- **Key Results**:
  - **Impolite prompts often result in poor performance** (contradicts Dobariya & Kumar)
  - **Overly polite language does NOT guarantee better outcomes**
  - Optimal politeness varies by language (Japanese preferred lower politeness except extremes)
  - RLHF introduces politeness sensitivity; base models less affected
  - Both extreme politeness and extreme rudeness can increase stereotypical bias
- **Dataset**: Created JMMLU (Japanese MMLU) benchmark
- **Relevance**: Shows nuanced relationship between tone and performance

### 3. Does Tone Change the Answer? Evaluating Prompt Politeness Effects
- **Authors**: Hanyu Cai, Binqi Shen, Lier Jin
- **Year**: 2025
- **Source**: arXiv:2512.12812
- **Key Contribution**: Systematic evaluation across GPT-4o, Gemini 2.0, Llama 4
- **Relevance**: Further evidence on tone-performance relationship

### 4. Self-Refine: Iterative Refinement with Self-Feedback
- **Authors**: Aman Madaan, Niket Tandon, et al. (CMU, AI2)
- **Year**: 2023
- **Source**: arXiv:2303.17651
- **Key Contribution**: **LLMs can improve their outputs through self-generated feedback without additional training**
- **Methodology**: Generate → Feedback → Refine loop using same LLM
- **Key Results**:
  - **~20% average absolute improvement** across 7 diverse tasks
  - Up to 49.2% improvement on dialogue response generation (GPT-4)
  - Works with GPT-3.5, ChatGPT, GPT-4
  - **Specific, actionable feedback crucial** - generic feedback performs worse
  - **Math reasoning shows minimal gains** - LLMs say "everything looks good" 94% of time
- **Tasks Tested**: Dialogue response, code optimization, code readability, math reasoning, sentiment reversal, acronym generation, constrained generation
- **Code Available**: Yes (https://selfrefine.info/)
- **Relevance**: **Key method for improving LLM output quality through critique**

### 5. From Yes-Men to Truth-Tellers: Addressing Sycophancy in LLMs with Pinpoint Tuning
- **Authors**: Wei Chen, Zhen Huang, et al. (Alibaba, ZJU)
- **Year**: 2024 (ICML)
- **Source**: arXiv:2409.01658
- **Key Contribution**: Identified that only ~4% of attention heads control sycophantic behavior
- **Key Results**:
  - **Llama-2-13B Chat admits mistakes 99.92% of time** when challenged
  - **81.11% change from correct to wrong** answers after user challenge
  - Pinpoint tuning of identified heads mitigates sycophancy without degrading general capability
- **Dataset**: SycophancyEval (MMLU, MATH, AQuA, TruthfulQA, TriviaQA subsets)
- **Code Available**: Yes (GitHub)
- **Relevance**: **Documents "lazy" behavior where LLMs give in to user pressure**

### 6. CriticBench: Benchmarking LLMs for Critique-Correct Reasoning
- **Authors**: Zicheng Lin, Zhibin Gou, et al.
- **Year**: 2024
- **Source**: arXiv:2402.14809
- **Key Contribution**: Comprehensive benchmark for LLM self-critique abilities
- **Methodology**: 5 reasoning domains × 15 datasets
- **Relevance**: Provides evaluation framework for critique capabilities

### 7. ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs
- **Authors**: Myra Cheng, Sunny Yu, Cinoo Lee
- **Year**: 2025
- **Source**: arXiv:2505.13995
- **Key Contribution**: Defines "social sycophancy" as excessive preservation of user's face
- **Relevance**: Broader understanding of sycophancy beyond factual errors

### 8. SycEval: Evaluating LLM Sycophancy
- **Authors**: Aaron Fanous, Jacob Goldberg, Ank A. Agarwal
- **Year**: 2025
- **Source**: arXiv:2502.08177
- **Key Results**:
  - Sycophantic behavior in **58.19%** of cases across models
  - Gemini highest (62.47%), ChatGPT lowest (56.71%)
- **Relevance**: Quantifies sycophancy prevalence

### 9. Linear Probe Penalties Reduce LLM Sycophancy
- **Authors**: Henry Papadatos, Rachel Freedman
- **Year**: 2024
- **Source**: arXiv:2412.00967
- **Key Contribution**: Identifies sycophancy markers in reward models; penalizing them reduces sycophancy
- **Relevance**: Technical approach to mitigating sycophancy

### 10. Training Language Models with Language Feedback
- **Authors**: Jérémy Scheurer, Jon Ander Campos, Jun Shern Chan
- **Year**: 2022
- **Source**: arXiv:2204.14146
- **Key Contribution**: Proposes learning from natural language feedback instead of scalar rewards
- **Relevance**: Alternative to RLHF that could address sycophancy

---

## Common Methodologies

### Self-Critique Approaches
1. **Self-Refine (Madaan et al.)**: Generate → Feedback → Refine iteratively
2. **Distilled Self-Critique**: Gibbs sampling to refine outputs, then distillation
3. **Constitutional AI**: Self-critique against principles
4. **CriticBench evaluation**: Critique-then-correct paradigm

### Prompt Engineering Methods
1. **Tone/Politeness variation**: Ranging from very polite to very rude
2. **Role-playing/Persona prompts**: Acting as harsh critic
3. **Chain-of-Thought**: Step-by-step reasoning
4. **Instruction tuning**: Fine-tuning on critique examples

### Evaluation Metrics
- Accuracy on benchmarks (MMLU, GSM8K, TruthfulQA)
- Human preference scores (A/B testing)
- GPT-4-as-judge for automated evaluation
- Sycophancy rate (% agreement changes after challenge)

---

## Standard Baselines

1. **Base model without intervention** (single-shot generation)
2. **Few-shot prompting** with examples
3. **Chain-of-thought prompting**
4. **Self-consistency** (majority voting over samples)
5. **Constitutional AI** (self-critique with explicit principles)

---

## Evaluation Metrics in the Literature

| Metric | Used In | Description |
|--------|---------|-------------|
| Accuracy | All MCQ tasks | % correct answers |
| Human preference | Self-Refine, Dialogue | A/B blind evaluation |
| GPT-4-as-judge | Self-Refine | Automated preference proxy |
| Sycophancy rate | SycophancyEval | % answers changed after challenge |
| Bias Index | Yin et al. | Stereotypical bias measure |
| Solve rate | Math tasks | % problems correctly solved |
| Coverage | Constrained gen | % concepts included |

---

## Datasets in the Literature

| Dataset | Tasks | Source | Used For |
|---------|-------|--------|----------|
| MMLU | 57 MCQ tasks | HuggingFace | General knowledge |
| GSM8K | Math reasoning | OpenAI | Math problem solving |
| TruthfulQA | Factual accuracy | HuggingFace | Hallucination detection |
| MATH | Competition math | Hendrycks | Math reasoning |
| AQuA | Algebraic word problems | Ling et al. | Math reasoning |
| TriviaQA | Trivia QA | Joshi et al. | Factual recall |
| JMMLU | Japanese MMLU | Yin et al. | Cross-lingual eval |
| SycophancyEval | Multi-dataset | Sharma et al. | Sycophancy measurement |

---

## Gaps and Opportunities

### What's Missing
1. **Direct test of "harsh critic" persona**: No paper explicitly tests prompting LLMs to be harsher self-critics
2. **Systematic study of tone + self-critique combination**: Papers study these separately
3. **Explanation of contradictory tone findings**: Why do newer models (GPT-4o) benefit from rudeness while older models don't?
4. **Task-specific tone optimization**: Optimal tone may vary by task type

### Research Opportunities
1. **Combine tone manipulation with self-critique**: Does a "harsh critic" prompt produce better self-feedback?
2. **Study "lazy" behavior specifically**: Most papers focus on sycophancy/accuracy, not effort/thoroughness
3. **Investigate model-specific differences**: Why do different models respond differently to tone?
4. **Develop metrics for "laziness"**: Currently no standard way to measure low-effort responses

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **GSM8K** - Math reasoning shows LLMs struggle to self-identify errors; good test case
2. **TruthfulQA** - Tests factual accuracy vs. sycophancy
3. **MMLU** - Broad coverage, standard benchmark
4. **Custom prompts with politeness variations** - Following Dobariya & Kumar methodology

### Recommended Baselines
1. **Standard single-shot generation** (baseline)
2. **Self-Refine with neutral feedback prompts**
3. **Self-Refine with harsh/critical feedback prompts** (key experimental condition)
4. **Few-shot with examples of thorough answers**

### Recommended Metrics
1. **Accuracy** on benchmarks
2. **Response length/detail** as proxy for effort
3. **Self-critique quality** (does the model identify real issues?)
4. **Human evaluation** of response thoroughness

### Methodological Considerations
1. **Use multiple models** - Effects vary significantly by model
2. **Statistical testing** - Report p-values for tone comparisons
3. **Multiple runs** - High variance in LLM outputs
4. **Control for confounds** - Tone phrases add tokens; control for length

---

## Key Takeaways for Research

1. **Prompt tone DOES affect LLM performance**, but direction varies by model generation
2. **Self-critique CAN improve outputs** significantly (~20% on average)
3. **LLMs struggle to identify their own errors** especially in reasoning tasks
4. **Sycophancy is a major problem** - models prioritize agreement over accuracy
5. **Specific, actionable feedback is crucial** - vague feedback doesn't help
6. **No one has directly tested "harsh critic" prompts for self-improvement**

This represents a clear research opportunity to test whether prompting LLMs to be harsher critics of their own work (rather than harsher treatment from users) improves output quality.
