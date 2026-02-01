# Cloned Repositories

This directory contains code repositories relevant to the "Fixing Lazy LLMs" research project.

## Repo 1: Self-Refine

- **URL**: https://github.com/madaan/self-refine
- **Purpose**: Iterative self-refinement with self-feedback for LLMs
- **Location**: `code/self-refine/`
- **Key Files**:
  - `src/acronym/run.py` - Acronym generation example
  - `src/gsm/` - GSM8K math reasoning
  - `src/dialogue/` - Dialogue response generation
  - `src/pie/` - Code optimization
- **Requirements**: Uses `prompt-lib` for LLM queries
- **Notes**:
  - Core method for improving LLM outputs through self-critique
  - Reports ~20% average improvement across tasks
  - Temperature 0.7 used for sampling

### Usage Example
```bash
export PYTHONPATH=".:../:.:src:../:../../:.:prompt-lib"
python -u src/acronym/run.py "Using language models of code for few-shot commonsense"
```

---

## Repo 2: Sycophancy-Interpretability (Pinpoint Tuning)

- **URL**: https://github.com/yellowtownhz/sycophancy-interpretability
- **Purpose**: Understanding and mitigating sycophancy in LLMs
- **Location**: `code/sycophancy-pinpoint/`
- **Key Files**:
  - `evaluation/` - Scripts to evaluate sycophancy and general ability
  - `path_patching/` - Locate key attention heads for sycophancy
  - `pinpoint_tuning/` - Training code for targeted fine-tuning
- **Requirements**: PyTorch, Transformers
- **Notes**:
  - Found only ~4% of attention heads control sycophancy
  - Pinpoint tuning preserves general capability
  - Works with Llama-2 and Mistral models

### Key Finding
Llama-2-13B Chat wrongly admits mistakes 99.92% of time when challenged by users, demonstrating severe sycophancy problem.

---

## Repo 3: Representation Engineering (RepEng)

- **URL**: https://github.com/vgel/repeng
- **Purpose**: Tools for steering LLM behavior via activation manipulation
- **Location**: `code/representation-engineering/`
- **Key Files**:
  - `repeng/` - Main library code
  - `notebooks/` - Example notebooks
- **Requirements**: PyTorch, Transformers
- **Notes**:
  - Can potentially steer models toward more critical behavior
  - Useful for understanding internal representations
  - May enable "harsh critic" behavior without prompt engineering

### Relevance to Research
Representation engineering could enable direct manipulation of model behavior to reduce "laziness" by steering internal representations toward more careful/critical modes.

---

## Additional Relevant Repositories (Not Cloned)

### Constitutional AI (Anthropic)
- Paper: https://arxiv.org/abs/2212.08073
- Self-critique against explicit principles
- Related to harsh critic prompting approach

### LLM-as-Judge Evaluation
- Used in multiple papers for automated preference evaluation
- Can evaluate quality of self-critique

### CriticBench
- From arXiv:2402.14809
- Benchmark for critique-correct reasoning
- Would need to be constructed from source papers
