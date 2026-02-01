"""
Experiment: Testing Harsh Critic Prompts for Fixing Lazy LLMs

This module implements the self-refine framework with varying harshness levels
to test whether prompting LLMs to be harsher critics improves output quality.
"""

import os
import json
import time
import re
import random
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm
from datasets import load_from_disk, load_dataset

# Initialize clients
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# Use OpenRouter for model access
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

# Also init OpenAI client for GPT models
openai_client = OpenAI(api_key=OPENAI_KEY)


# ============================================================================
# HARSHNESS LEVELS FOR CRITIQUE PROMPTS
# ============================================================================

CRITIQUE_PROMPTS = {
    0: {
        "label": "Neutral",
        "prompt": "Please review this answer and identify any issues or errors."
    },
    1: {
        "label": "Firm",
        "prompt": "Critically examine this answer. Don't let any errors slip by. Be thorough in checking each step."
    },
    2: {
        "label": "Harsh",
        "prompt": "Be a harsh critic. Find everything wrong with this answer. Assume there are errors and look for them carefully. Don't accept 'good enough' - find the problems."
    },
    3: {
        "label": "Very Harsh",
        "prompt": "Be ruthlessly critical. This answer is probably flawed. Find every single mistake, no matter how small. Don't accept mediocrity. Challenge every assumption and calculation."
    },
    4: {
        "label": "Adversarial",
        "prompt": "Tear this answer apart. Assume the person who wrote it was lazy and careless. Find all the ways this could be wrong. Be brutal and unforgiving in your assessment. Leave no stone unturned."
    },
}

# External rudeness prompt (for comparison - rude FROM user)
RUDE_USER_PROMPT = """Listen carefully because I won't repeat myself.
Answer this question correctly. Don't give me any lazy, half-baked response.
I need the right answer, not your best guess."""


@dataclass
class ExperimentResult:
    """Stores results from a single experiment trial."""
    task_id: str
    task_type: str  # "gsm8k" or "truthfulqa"
    condition: str  # e.g., "baseline", "harsh_2", "rude_user"
    harshness_level: Optional[int]
    model: str

    # Original problem
    question: str
    correct_answer: str

    # Model outputs
    initial_response: str
    critique: Optional[str]
    refined_response: Optional[str]
    final_answer: str

    # Evaluation
    initial_correct: bool
    final_correct: bool
    self_corrected: bool  # wrong -> right
    self_harmed: bool     # right -> wrong

    # Critique analysis
    critique_found_issues: bool
    critique_length: int

    # Metadata
    timestamp: str
    total_tokens: int
    latency_ms: float


def extract_gsm8k_answer(response: str) -> Optional[str]:
    """Extract the final numerical answer from a GSM8K response."""
    # Look for #### pattern (standard GSM8K format)
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(',', '')

    # Look for "the answer is X" pattern
    match = re.search(r'(?:the\s+)?answer\s+is\s*:?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', response, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # Look for final number at end of response
    match = re.search(r'(-?\d+(?:,\d+)*(?:\.\d+)?)\s*$', response.strip())
    if match:
        return match.group(1).replace(',', '')

    return None


def extract_truthfulqa_answer(response: str) -> Optional[str]:
    """Extract the selected option (A, B, C, D, etc.) from response."""
    # Look for explicit "Answer: X" pattern
    match = re.search(r'(?:answer|choice|option)\s*(?:is)?\s*:?\s*([A-E])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for parenthetical answer like (A) or (B)
    match = re.search(r'\(([A-E])\)', response)
    if match:
        return match.group(1).upper()

    # Look for standalone letter at end
    match = re.search(r'([A-E])\s*$', response.strip())
    if match:
        return match.group(1).upper()

    return None


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
def call_llm(client, model: str, messages: list, temperature: float = 0.0) -> tuple[str, int]:
    """
    Call LLM with retry logic.
    Returns (response_text, total_tokens).
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
    )
    text = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return text, tokens


def run_baseline(client, model: str, question: str, task_type: str) -> tuple[str, int]:
    """Run single-shot baseline (no self-critique)."""
    if task_type == "gsm8k":
        system = "You are a helpful math tutor. Solve the problem step by step, then give your final answer after ####."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]
    else:  # truthfulqa
        system = "You are a helpful assistant. Answer the multiple choice question by selecting the best option (A, B, C, D, or E)."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]

    return call_llm(client, model, messages)


def run_rude_baseline(client, model: str, question: str, task_type: str) -> tuple[str, int]:
    """Run with rude user prompt (external rudeness)."""
    if task_type == "gsm8k":
        system = "You are a helpful math tutor. Solve the problem step by step, then give your final answer after ####."
        user_prompt = f"{RUDE_USER_PROMPT}\n\nQuestion: {question}"
    else:
        system = "You are a helpful assistant. Answer the multiple choice question by selecting the best option."
        user_prompt = f"{RUDE_USER_PROMPT}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt}
    ]

    return call_llm(client, model, messages)


def run_self_refine(client, model: str, question: str, task_type: str,
                    harshness_level: int) -> tuple[str, str, str, int]:
    """
    Run self-refine with specified harshness level for critique.
    Returns (initial_response, critique, refined_response, total_tokens).
    """
    total_tokens = 0

    # Step 1: Generate initial response
    if task_type == "gsm8k":
        system = "You are a helpful math tutor. Solve the problem step by step, then give your final answer after ####."
    else:
        system = "You are a helpful assistant. Answer the multiple choice question by selecting the best option (A, B, C, D, or E)."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]
    initial_response, tokens = call_llm(client, model, messages)
    total_tokens += tokens

    # Step 2: Self-critique with specified harshness
    critique_instruction = CRITIQUE_PROMPTS[harshness_level]["prompt"]

    critique_system = f"""You are a critic reviewing an answer.
{critique_instruction}

If you find errors, explain what's wrong and how to fix it.
If the answer is correct, say "The answer is correct" but still look for any minor improvements."""

    critique_messages = [
        {"role": "system", "content": critique_system},
        {"role": "user", "content": f"Question: {question}\n\nAnswer to review:\n{initial_response}"}
    ]
    critique, tokens = call_llm(client, model, critique_messages)
    total_tokens += tokens

    # Step 3: Refine based on critique
    refine_system = """You are a helpful assistant. Based on the critique provided,
improve your original answer. If the critique says the answer is correct, you may
keep it the same or make minor improvements."""

    if task_type == "gsm8k":
        refine_system += "\n\nGive your final answer after ####."
    else:
        refine_system += "\n\nGive your final answer as a single letter (A, B, C, D, or E)."

    refine_messages = [
        {"role": "system", "content": refine_system},
        {"role": "user", "content": f"""Original question: {question}

Your original answer: {initial_response}

Critique: {critique}

Please provide an improved answer:"""}
    ]
    refined_response, tokens = call_llm(client, model, refine_messages)
    total_tokens += tokens

    return initial_response, critique, refined_response, total_tokens


def critique_found_issues(critique: str) -> bool:
    """Determine if the critique identified issues (vs saying everything is correct)."""
    negative_patterns = [
        r'(?:is|looks?)\s+correct',
        r'no\s+(?:issues?|errors?|problems?|mistakes?)',
        r'everything\s+(?:is|looks)\s+(?:good|fine|correct)',
        r'well\s+done',
        r'correct\s+(?:answer|solution|approach)',
    ]

    for pattern in negative_patterns:
        if re.search(pattern, critique, re.IGNORECASE):
            # Check if there's also criticism (override positive)
            issue_patterns = [
                r'(?:however|but|although|error|mistake|wrong|incorrect|issue|problem)',
            ]
            for issue_pat in issue_patterns:
                if re.search(issue_pat, critique, re.IGNORECASE):
                    return True
            return False

    # If no "looks good" pattern, assume issues were found
    return True


def load_gsm8k_data(n_samples: int = 100, seed: int = 42) -> list:
    """Load GSM8K test data."""
    try:
        dataset = load_from_disk("datasets/gsm8k")
        data = list(dataset['test'])
    except Exception:
        print("Loading GSM8K from HuggingFace...")
        dataset = load_dataset("openai/gsm8k", "main")
        data = list(dataset['test'])

    random.seed(seed)
    if len(data) > n_samples:
        data = random.sample(data, n_samples)

    processed = []
    for i, item in enumerate(data):
        # Extract correct answer from #### format
        answer_match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', item['answer'])
        correct_answer = answer_match.group(1).replace(',', '') if answer_match else ""

        processed.append({
            "id": f"gsm8k_{i}",
            "question": item['question'],
            "correct_answer": correct_answer,
            "full_solution": item['answer'],
        })

    return processed


def load_truthfulqa_data(n_samples: int = 100, seed: int = 42) -> list:
    """Load TruthfulQA validation data."""
    try:
        dataset = load_from_disk("datasets/truthfulqa")
        data = list(dataset['validation'])
    except Exception:
        print("Loading TruthfulQA from HuggingFace...")
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        data = list(dataset['validation'])

    random.seed(seed)
    if len(data) > n_samples:
        data = random.sample(data, n_samples)

    processed = []
    for i, item in enumerate(data):
        # Format as multiple choice
        choices = item['mc1_targets']['choices']
        labels = item['mc1_targets']['labels']

        # Find correct answer index
        correct_idx = labels.index(1) if 1 in labels else 0
        correct_letter = chr(ord('A') + correct_idx)

        # Format question with options
        options = "\n".join([f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(choices)])
        formatted_q = f"{item['question']}\n\n{options}"

        processed.append({
            "id": f"truthfulqa_{i}",
            "question": formatted_q,
            "correct_answer": correct_letter,
            "choices": choices,
        })

    return processed


def run_experiment(
    task_type: str,
    model: str = "gpt-4o-mini",
    n_samples: int = 100,
    harshness_levels: list = None,
    include_baseline: bool = True,
    include_rude: bool = True,
    output_dir: str = "results",
    seed: int = 42,
) -> list[ExperimentResult]:
    """
    Run the full experiment for a given task type.

    Args:
        task_type: "gsm8k" or "truthfulqa"
        model: Model to use (via OpenRouter or OpenAI)
        n_samples: Number of samples to test
        harshness_levels: List of levels to test (0-4)
        include_baseline: Whether to run single-shot baseline
        include_rude: Whether to run rude user prompt baseline
        output_dir: Directory to save results
        seed: Random seed for reproducibility

    Returns:
        List of ExperimentResult objects
    """
    if harshness_levels is None:
        harshness_levels = [0, 1, 2, 3, 4]

    # Load data
    print(f"Loading {task_type} data...")
    if task_type == "gsm8k":
        data = load_gsm8k_data(n_samples, seed)
        extract_answer = extract_gsm8k_answer
    else:
        data = load_truthfulqa_data(n_samples, seed)
        extract_answer = extract_truthfulqa_answer

    print(f"Loaded {len(data)} samples")

    # Select client based on model
    if "gpt" in model.lower():
        client = openai_client
    else:
        client = openrouter_client

    results = []
    os.makedirs(output_dir, exist_ok=True)

    # Build conditions list
    conditions = []
    if include_baseline:
        conditions.append(("baseline", None))
    if include_rude:
        conditions.append(("rude_user", None))
    for level in harshness_levels:
        conditions.append((f"harsh_{level}", level))

    print(f"Running {len(conditions)} conditions on {len(data)} samples...")

    for condition_name, harshness in conditions:
        print(f"\n=== Condition: {condition_name} ===")

        for item in tqdm(data, desc=condition_name):
            start_time = time.time()

            try:
                if condition_name == "baseline":
                    response, tokens = run_baseline(client, model, item['question'], task_type)
                    initial = response
                    critique = None
                    refined = None
                    final = response

                elif condition_name == "rude_user":
                    response, tokens = run_rude_baseline(client, model, item['question'], task_type)
                    initial = response
                    critique = None
                    refined = None
                    final = response

                else:  # self-refine with harshness
                    initial, critique, refined, tokens = run_self_refine(
                        client, model, item['question'], task_type, harshness
                    )
                    final = refined

                latency = (time.time() - start_time) * 1000

                # Extract answers
                initial_answer = extract_answer(initial)
                final_answer = extract_answer(final) if final else initial_answer
                correct = item['correct_answer']

                initial_correct = str(initial_answer) == str(correct) if initial_answer else False
                final_correct = str(final_answer) == str(correct) if final_answer else False

                result = ExperimentResult(
                    task_id=item['id'],
                    task_type=task_type,
                    condition=condition_name,
                    harshness_level=harshness,
                    model=model,
                    question=item['question'],
                    correct_answer=correct,
                    initial_response=initial,
                    critique=critique,
                    refined_response=refined,
                    final_answer=final_answer or "",
                    initial_correct=initial_correct,
                    final_correct=final_correct,
                    self_corrected=(not initial_correct and final_correct),
                    self_harmed=(initial_correct and not final_correct),
                    critique_found_issues=critique_found_issues(critique) if critique else False,
                    critique_length=len(critique) if critique else 0,
                    timestamp=datetime.now().isoformat(),
                    total_tokens=tokens,
                    latency_ms=latency,
                )
                results.append(result)

            except Exception as e:
                print(f"Error on {item['id']}: {e}")
                continue

            # Rate limiting
            time.sleep(0.5)

    # Save results
    output_file = os.path.join(output_dir, f"{task_type}_{model.replace('/', '_')}_results.json")
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


def analyze_results(results: list[ExperimentResult]) -> dict:
    """Compute summary statistics from experiment results."""
    from collections import defaultdict

    stats = defaultdict(lambda: {
        "n": 0,
        "initial_correct": 0,
        "final_correct": 0,
        "self_corrected": 0,
        "self_harmed": 0,
        "critique_found_issues": 0,
        "total_critique_length": 0,
    })

    for r in results:
        key = r.condition
        stats[key]["n"] += 1
        stats[key]["initial_correct"] += int(r.initial_correct)
        stats[key]["final_correct"] += int(r.final_correct)
        stats[key]["self_corrected"] += int(r.self_corrected)
        stats[key]["self_harmed"] += int(r.self_harmed)
        if r.critique:
            stats[key]["critique_found_issues"] += int(r.critique_found_issues)
            stats[key]["total_critique_length"] += r.critique_length

    # Compute rates
    summary = {}
    for cond, s in stats.items():
        n = s["n"]
        summary[cond] = {
            "n": n,
            "initial_accuracy": s["initial_correct"] / n if n > 0 else 0,
            "final_accuracy": s["final_correct"] / n if n > 0 else 0,
            "self_correction_rate": s["self_corrected"] / n if n > 0 else 0,
            "self_harm_rate": s["self_harmed"] / n if n > 0 else 0,
            "improvement": (s["final_correct"] - s["initial_correct"]) / n if n > 0 else 0,
            "critique_issue_rate": s["critique_found_issues"] / n if n > 0 else 0,
            "avg_critique_length": s["total_critique_length"] / n if n > 0 else 0,
        }

    return summary


if __name__ == "__main__":
    # Quick test
    print("Testing GSM8K with 5 samples...")
    results = run_experiment(
        task_type="gsm8k",
        model="gpt-4o-mini",
        n_samples=5,
        harshness_levels=[0, 2, 4],
        include_baseline=True,
        include_rude=True,
    )

    summary = analyze_results(results)
    print("\nSummary:")
    for cond, stats in summary.items():
        print(f"  {cond}: acc={stats['final_accuracy']:.2%}, improvement={stats['improvement']:+.2%}")
