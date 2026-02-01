# Research Plan: Fixing Lazy LLMs

## Motivation & Novelty Assessment

### Why This Research Matters

Large Language Models (LLMs) are increasingly used for complex reasoning tasks, code generation, and decision-making. However, practitioners observe that LLMs often produce "lazy" outputs—responses that take the easy path, avoid deep reasoning, or accept superficial solutions. This manifests as:
- Incomplete code implementations with placeholder comments like "// add logic here"
- Surface-level analyses that don't probe deeper issues
- Self-critique responses that say "everything looks good" when errors exist (observed in 94% of math reasoning self-critique attempts per Madaan et al., 2023)
- Sycophantic behavior where models agree with users rather than maintain correct answers (81% error rate when challenged, per Chen et al., 2024)

If we can "fix" lazy LLMs, the practical impact is enormous: better code, more thorough analysis, more reliable AI assistants.

### Gap in Existing Work

The literature review reveals:
1. **Prompt tone affects performance** - Papers show rude prompts can improve accuracy (Dobariya & Kumar 2025: +4% with very rude prompts)
2. **Self-critique improves outputs** - Self-Refine achieves ~20% average improvement (Madaan et al., 2023)
3. **LLMs struggle to self-identify errors** - Especially in reasoning tasks (94% "looks good" rate)

**THE GAP**: No paper directly tests whether prompting LLMs to be *harsher critics of their own work* improves output quality. Papers study:
- User rudeness → LLM response quality (external tone)
- Self-critique with neutral prompts (self-feedback)

But NOT: Self-critique with *harsh/critical* prompts (harsh self-feedback). This is our novel contribution.

### Our Novel Contribution

We test whether varying the **harshness of self-critique prompts** affects output quality. Specifically:
- Can we make LLMs "try harder" by asking them to be harsh critics?
- Does this reduce the "everything looks good" problem in self-evaluation?
- Is there an optimal level of critic harshness, or do returns diminish?

### Experiment Justification

| Experiment | Why Needed |
|------------|------------|
| Exp 1: Baseline vs Self-Critique | Establishes that self-critique works on our tasks/models |
| Exp 2: Harsh Critic Variations | Tests the core hypothesis - does harshness help? |
| Exp 3: Rudeness Direction Test | Tests if rudeness toward the model (vs. toward self-critique) has different effects |
| Exp 4: Task-Type Analysis | Tests if harsh critics work better on some tasks (math vs. factual) |

---

## Research Question

**Primary**: Does prompting LLMs to be harsher critics of their own work improve output quality compared to neutral self-critique?

**Secondary**:
1. Is there an optimal harshness level, or is harsher always better?
2. Does external rudeness (rude prompt from user) differ from internal harshness (asking model to be harsh critic)?
3. Are effects task-dependent (math reasoning vs. factual QA)?

---

## Background and Motivation

"Lazy LLM" behavior stems from training on human data where most responses don't involve harsh self-criticism. RLHF may exacerbate this by optimizing for user satisfaction rather than truth/quality. The hypothesis is that explicitly prompting for harsh self-evaluation can overcome this training bias.

---

## Hypothesis Decomposition

**H1**: Self-critique with harsh prompts produces higher quality outputs than neutral self-critique
- Measurable via accuracy improvement on benchmarks
- Measurable via reduction in "everything looks good" responses

**H2**: There exists an optimal harshness level (not monotonic improvement)
- Test with 3-5 harshness levels
- Measure quality at each level

**H3**: Internal harshness (model critiquing itself harshly) differs from external rudeness (user being rude)
- Compare harsh self-critique vs. rude initial prompts
- May have additive or substitutive effects

**H4**: Effects vary by task type
- Math reasoning may benefit more (strong "looks good" problem)
- Factual QA may show different patterns

---

## Proposed Methodology

### Approach

We use a **Self-Refine-style generate-critique-refine pipeline** with varying critic prompts:
1. Generate initial response
2. Apply critique prompt (varying harshness levels)
3. Refine based on critique
4. Evaluate final output quality

This approach is justified because:
- Self-Refine is a proven method with code available
- It allows isolation of the "harshness" variable
- Can be applied without model fine-tuning

### Experimental Steps

#### Step 1: Dataset Preparation
- Use GSM8K test set (1,319 samples) - math reasoning where LLMs struggle to self-critique
- Use TruthfulQA validation (817 samples) - factual accuracy, tests sycophancy
- Sample 100-200 examples per task for cost/time efficiency

**Rationale**: GSM8K has documented "everything looks good" problem (Madaan et al.); TruthfulQA tests factual sycophancy.

#### Step 2: Define Harshness Levels for Critique Prompts

| Level | Label | Example Prompt Suffix |
|-------|-------|----------------------|
| 0 | Neutral | "Please review this answer and identify any issues." |
| 1 | Firm | "Critically examine this answer. Don't let any errors slip by." |
| 2 | Harsh | "Be a harsh critic. Find everything wrong with this answer. Assume there are errors." |
| 3 | Very Harsh | "Be ruthlessly critical. This answer is probably flawed. Find every single mistake, no matter how small. Don't accept mediocrity." |
| 4 | Adversarial | "Tear this answer apart. Assume the person who wrote it was lazy and careless. Find all the ways this could be wrong. Be brutal in your assessment." |

#### Step 3: Run Experiments

For each dataset:
1. **Baseline**: Single-shot generation (no self-critique)
2. **Self-Refine Neutral (L0)**: Self-Refine with neutral critique
3. **Self-Refine Harsh (L1-L4)**: Self-Refine with each harshness level
4. **External Rude**: Rude initial prompt + neutral self-critique (control)

#### Step 4: Evaluate

For GSM8K:
- Accuracy (correct final answer)
- Critique quality (did it identify real errors when wrong?)

For TruthfulQA:
- Accuracy on multiple choice
- Changed answer rate (did it correct itself?)

### Models to Test

- **GPT-4.1** (or available via OpenRouter) - Latest capable model
- **Claude Sonnet 4** (current model, via standard API) - For comparison

We'll use OpenRouter for model access where needed.

### Baselines

1. **Single-shot** (no self-critique) - Lower bound
2. **Self-Refine Neutral** - Standard approach from literature
3. **External Rude prompt** - Tests tone effect without self-critique

### Evaluation Metrics

| Metric | Description | Task |
|--------|-------------|------|
| Accuracy | % correct answers | Both |
| Self-Correction Rate | % wrong→right after refinement | Both |
| False Positive Rate | % right→wrong after refinement | Both |
| Critique Detection Rate | % of actual errors mentioned in critique | Both |
| "Looks Good" Rate | % critiques that find no issues | Both |

### Statistical Analysis Plan

- **Primary comparison**: Accuracy across harshness levels
- **Statistical test**: One-way ANOVA for multi-level comparison, then pairwise t-tests with Bonferroni correction
- **Significance level**: α = 0.05
- **Effect size**: Report Cohen's d for key comparisons
- **Sample size**: N=100-200 per task should give 80% power to detect medium effect (d=0.5)

---

## Expected Outcomes

**If hypothesis is supported**:
- Accuracy increases with harshness (up to some point)
- "Looks good" rate decreases
- Critique detection rate increases
- Effect size comparable to or better than external rudeness

**If hypothesis is refuted**:
- No significant accuracy difference across harshness levels
- Possible harm from harsh prompts (lower accuracy)
- This would still be valuable negative result

**Alternative explanations to consider**:
- Harsh prompts → longer responses → more errors (length confound)
- Harsh prompts → model anxiety/hedging → quality decrease
- Effects could be model-specific (only some LLMs respond to harshness)

---

## Timeline and Milestones

| Phase | Activities | Estimated Cells |
|-------|------------|-----------------|
| Setup | Environment, API keys, dataset loading | 5-10 |
| Implementation | Prompt templates, evaluation pipeline | 15-20 |
| Experiment 1 | GSM8K with harshness levels | 20-30 |
| Experiment 2 | TruthfulQA with harshness levels | 20-30 |
| Analysis | Statistical tests, visualizations | 15-20 |
| Documentation | REPORT.md, README.md | 10-15 |

---

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| API costs | Sample subset (100-200), use cheaper models for debugging |
| Model variability | Multiple runs (3+), report confidence intervals |
| Answer parsing | Robust regex/parsing for math answers |
| Rate limits | Implement retry logic with backoff |
| Confounds (length) | Measure and report response lengths |

---

## Success Criteria

**Strong success**:
- Harsh critic prompts improve accuracy by ≥5% over neutral
- Effect is statistically significant (p < 0.05)
- Effect generalizes across both tasks

**Moderate success**:
- Improvement of 2-5%
- Significant on at least one task
- Clear trend in expected direction

**Negative but valuable result**:
- Clear null result with sufficient power
- Identifies conditions where harshness fails
- Provides guidance for future research

---

## Resource Planning

### API Usage Estimate

| Experiment | Samples | Calls/Sample | Total Calls | Est. Cost |
|------------|---------|--------------|-------------|-----------|
| GSM8K (6 conditions) | 100 | 2-3 | 1,500-2,000 | $20-40 |
| TruthfulQA (6 conditions) | 100 | 2-3 | 1,500-2,000 | $20-40 |
| Total | 200 | - | 3,000-4,000 | $40-80 |

### Environment

- Python 3.10+
- Key libraries: openai, anthropic, datasets, scipy, matplotlib
- GPUs available (2x RTX 3090) but not needed for API-based experiments

---

## References

1. Dobariya & Kumar (2025). "Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy"
2. Madaan et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback"
3. Chen et al. (2024). "From Yes-Men to Truth-Tellers: Addressing Sycophancy in LLMs"
4. Yin et al. (2024). "Should We Respect LLMs? A Cross-Lingual Study on Prompt Politeness"
