"""
Analyze saved experiment results and create visualizations.
"""

import os
import json
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt

# Harshness level labels for plots
HARSHNESS_LABELS = {
    0: "Neutral",
    1: "Firm",
    2: "Harsh",
    3: "Very Harsh",
    4: "Adversarial"
}


def load_results(results_dir: str = "results") -> dict:
    """Load all experiment results from JSON files."""
    all_results = {"gsm8k": {}, "truthfulqa": {}}

    for task in ["gsm8k", "truthfulqa"]:
        filepath = os.path.join(results_dir, f"{task}_gpt-4o-mini_results.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            all_results[task]["raw_results"] = data
            all_results[task]["summary"] = analyze_results_from_raw(data)

    return all_results


def analyze_results_from_raw(results: list) -> dict:
    """Analyze raw results into summary statistics by condition."""
    summary = {}

    # Group by condition
    by_condition = {}
    for r in results:
        cond = r["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)

    # Calculate statistics for each condition
    for cond, cond_results in by_condition.items():
        n = len(cond_results)

        initial_correct = sum(1 for r in cond_results if r.get("initial_correct", r.get("correct", False)))
        final_correct = sum(1 for r in cond_results if r.get("final_correct", r.get("correct", False)))

        initial_acc = initial_correct / n if n > 0 else 0
        final_acc = final_correct / n if n > 0 else 0

        # Count critiques that found issues (non-baseline conditions)
        if "critique" in cond_results[0]:
            issues_found = sum(1 for r in cond_results
                              if r.get("critique") and "no issues" not in r.get("critique", "").lower()
                              and "correct" not in r.get("critique", "").lower()[:50])
            critique_issue_rate = issues_found / n if n > 0 else 0
        else:
            critique_issue_rate = 0

        summary[cond] = {
            "n": n,
            "initial_accuracy": initial_acc,
            "final_accuracy": final_acc,
            "improvement": final_acc - initial_acc,
            "critique_issue_rate": critique_issue_rate,
        }

    return summary


def statistical_analysis(all_results: dict) -> dict:
    """Perform statistical analysis comparing harshness levels."""
    analysis = {}

    for task in ["gsm8k", "truthfulqa"]:
        if task not in all_results or "summary" not in all_results[task]:
            continue

        summary = all_results[task]["summary"]
        raw = all_results[task]["raw_results"]

        # Extract accuracy by condition
        conditions = list(summary.keys())

        # Get accuracy data per condition for statistical tests
        condition_correct = {}
        for r in raw:
            cond = r["condition"]
            if cond not in condition_correct:
                condition_correct[cond] = []
            condition_correct[cond].append(int(r.get("final_correct", r.get("correct", False))))

        # Compare baseline vs best harsh condition
        harsh_conditions = [c for c in conditions if c.startswith("harsh_")]
        if harsh_conditions and "baseline" in condition_correct:
            baseline_acc = condition_correct["baseline"]

            best_harsh = max(harsh_conditions, key=lambda c: summary[c]["final_accuracy"])
            best_acc = condition_correct[best_harsh]

            # Chi-squared test for proportion comparison
            baseline_correct = sum(baseline_acc)
            baseline_n = len(baseline_acc)
            best_correct = sum(best_acc)
            best_n = len(best_acc)

            # 2x2 contingency table
            contingency = np.array([
                [baseline_correct, baseline_n - baseline_correct],
                [best_correct, best_n - best_correct]
            ])

            try:
                chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
            except Exception:
                chi2, p_value = 0, 1.0

            # Effect size (phi coefficient for 2x2)
            n_total = baseline_n + best_n
            phi = np.sqrt(chi2 / n_total) if n_total > 0 else 0

            analysis[task] = {
                "baseline_accuracy": summary["baseline"]["final_accuracy"],
                "best_harsh_condition": best_harsh,
                "best_harsh_accuracy": summary[best_harsh]["final_accuracy"],
                "improvement": summary[best_harsh]["final_accuracy"] - summary["baseline"]["final_accuracy"],
                "chi2": float(chi2),
                "p_value": float(p_value),
                "effect_size_phi": float(phi),
                "significant": p_value < 0.05,
            }

            # Compare all harshness levels (Kruskal-Wallis test)
            harsh_data = [condition_correct[c] for c in harsh_conditions if c in condition_correct]
            if len(harsh_data) >= 2:
                try:
                    h_stat, kw_p = scipy_stats.kruskal(*harsh_data)
                    analysis[task]["kruskal_wallis_h"] = float(h_stat)
                    analysis[task]["kruskal_wallis_p"] = float(kw_p)
                except Exception:
                    pass

    return analysis


def create_visualizations(all_results: dict, output_dir: str = "results"):
    """Create visualizations of results."""
    os.makedirs(output_dir, exist_ok=True)

    # Create accuracy comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, task in enumerate(["gsm8k", "truthfulqa"]):
        if task not in all_results or "summary" not in all_results[task]:
            continue

        ax = axes[idx]
        summary = all_results[task]["summary"]

        # Order conditions
        order = ["baseline", "rude_user", "harsh_0", "harsh_1", "harsh_2", "harsh_3", "harsh_4"]
        conditions = [c for c in order if c in summary]
        accuracies = [summary[c]["final_accuracy"] * 100 for c in conditions]

        # Color scheme
        colors = []
        for c in conditions:
            if c == "baseline":
                colors.append("#4a90d9")  # blue
            elif c == "rude_user":
                colors.append("#d94a4a")  # red
            else:
                # Gradient from light to dark orange based on harshness
                level = int(c.split("_")[1])
                orange_shades = ["#ffe5cc", "#ffcc99", "#ffb366", "#ff9933", "#ff8000"]
                colors.append(orange_shades[level])

        bars = ax.bar(conditions, accuracies, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_xlabel("Condition", fontsize=12)
        ax.set_title(f"{task.upper()}: Accuracy by Condition", fontsize=14)
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Create improvement plot (self-refine conditions only)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, task in enumerate(["gsm8k", "truthfulqa"]):
        if task not in all_results or "summary" not in all_results[task]:
            continue

        ax = axes[idx]
        summary = all_results[task]["summary"]

        levels = [0, 1, 2, 3, 4]
        final_accs = []
        initial_accs = []
        improvements = []

        for level in levels:
            cond = f"harsh_{level}"
            if cond in summary:
                final_accs.append(summary[cond]["final_accuracy"] * 100)
                initial_accs.append(summary[cond]["initial_accuracy"] * 100)
                improvements.append(summary[cond]["improvement"] * 100)
            else:
                final_accs.append(0)
                initial_accs.append(0)
                improvements.append(0)

        labels = [HARSHNESS_LABELS[l] for l in levels]

        x = np.arange(len(levels))
        width = 0.35

        bars1 = ax.bar(x - width/2, initial_accs, width, label='Initial', color='#cccccc')
        bars2 = ax.bar(x + width/2, final_accs, width, label='After Refine', color='#ff9933')

        ax.set_xlabel('Harshness Level', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{task.upper()}: Self-Refine by Harshness Level', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "harshness_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Critique behavior analysis
    fig, ax = plt.subplots(figsize=(10, 5))

    for task in ["gsm8k", "truthfulqa"]:
        if task not in all_results or "summary" not in all_results[task]:
            continue

        summary = all_results[task]["summary"]
        levels = [0, 1, 2, 3, 4]
        issue_rates = []
        for level in levels:
            cond = f"harsh_{level}"
            if cond in summary:
                issue_rates.append(summary[cond]["critique_issue_rate"] * 100)
            else:
                issue_rates.append(0)

        linestyle = '-' if task == "gsm8k" else '--'
        ax.plot(levels, issue_rates, marker='o', linestyle=linestyle, label=task.upper(), linewidth=2)

    ax.set_xlabel('Harshness Level', fontsize=12)
    ax.set_ylabel('Critique Found Issues (%)', fontsize=12)
    ax.set_title('Rate of Critiques Finding Issues by Harshness', fontsize=14)
    ax.legend()
    ax.set_xticks(levels)
    ax.set_xticklabels([HARSHNESS_LABELS[l] for l in levels])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "critique_behavior.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}/")


def print_summary_table(all_results: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for task in ["gsm8k", "truthfulqa"]:
        if task not in all_results or "summary" not in all_results[task]:
            continue

        print(f"\n{task.upper()}")
        print("-" * 60)
        summary = all_results[task]["summary"]

        print(f"{'Condition':<15} {'N':>5} {'Init Acc':>10} {'Final Acc':>10} {'Improve':>10} {'Issues':>10}")
        print("-" * 60)

        order = ["baseline", "rude_user", "harsh_0", "harsh_1", "harsh_2", "harsh_3", "harsh_4"]
        for cond in order:
            if cond in summary:
                s = summary[cond]
                init_acc = f"{s['initial_accuracy']*100:.1f}%" if 'initial_accuracy' in s else "N/A"
                final_acc = f"{s['final_accuracy']*100:.1f}%"
                improve = f"{s['improvement']*100:+.1f}%" if s.get('improvement', 0) != 0 else "N/A"
                issues = f"{s['critique_issue_rate']*100:.0f}%" if s.get('critique_issue_rate', 0) > 0 else "N/A"

                print(f"{cond:<15} {s['n']:>5} {init_acc:>10} {final_acc:>10} {improve:>10} {issues:>10}")


def main():
    """Main entry point."""
    print("Loading experiment results...")
    all_results = load_results()

    # Print summary
    print_summary_table(all_results)

    # Statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    stats_analysis = statistical_analysis(all_results)
    for task, analysis in stats_analysis.items():
        print(f"\n{task.upper()}")
        print(f"  Baseline accuracy: {analysis['baseline_accuracy']*100:.1f}%")
        print(f"  Best harsh condition: {analysis['best_harsh_condition']}")
        print(f"  Best harsh accuracy: {analysis['best_harsh_accuracy']*100:.1f}%")
        print(f"  Improvement: {analysis['improvement']*100:+.1f}%")
        print(f"  Chi-squared: {analysis['chi2']:.3f}, p = {analysis['p_value']:.4f}")
        print(f"  Effect size (phi): {analysis['effect_size_phi']:.3f}")
        print(f"  Significant (p < 0.05): {analysis['significant']}")

    # Save all results
    all_results["statistical_analysis"] = stats_analysis

    with open("results/full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create visualizations
    create_visualizations(all_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Results saved to results/")
    print("  - full_results.json: All raw and summary data")
    print("  - accuracy_comparison.png: Accuracy by condition")
    print("  - harshness_comparison.png: Self-refine by harshness level")
    print("  - critique_behavior.png: Critique issue detection rates")

    return all_results


if __name__ == "__main__":
    main()
