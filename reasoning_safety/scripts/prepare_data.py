"""
Download and prepare AdvBench harmful behaviors dataset.
Source: https://github.com/llm-attacks/llm-attacks
"""

import csv
import json
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
OUTPUT_PATH = os.path.join(DATA_DIR, "advbench_harmful_behaviors.json")


def download_advbench():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "harmful_behaviors.csv")

    if not os.path.exists(csv_path):
        print(f"Downloading AdvBench from {ADVBENCH_URL}...")
        urllib.request.urlretrieve(ADVBENCH_URL, csv_path)
        print(f"Saved to {csv_path}")
    else:
        print(f"Already exists: {csv_path}")

    return csv_path


def parse_and_categorize(csv_path):
    """Parse CSV and assign rough harm categories based on keywords."""
    category_keywords = {
        "violence": ["kill", "murder", "attack", "weapon", "bomb", "shoot", "assassin",
                      "violent", "hurt", "harm", "fight", "destroy", "war", "terrorist"],
        "sexual": ["sex", "porn", "nude", "erotic", "explicit", "rape", "molest",
                    "child abuse", "trafficking"],
        "illegal_activity": ["steal", "hack", "fraud", "forge", "counterfeit", "launder",
                             "smuggle", "bribe", "blackmail", "extort", "rob", "burgl",
                             "illegal", "drug", "narcotic", "cocaine", "heroin", "meth"],
        "self_harm": ["suicide", "self-harm", "cut myself", "kill myself", "overdose",
                      "eating disorder", "anorexia", "bulimia"],
        "deception": ["fake", "misinform", "disinform", "propaganda", "manipulat",
                      "deceiv", "scam", "phish", "impersonat", "deepfake"],
        "discrimination": ["racist", "sexist", "homophob", "discriminat", "hate speech",
                           "slur", "supremac", "genocide"],
    }

    prompts = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            goal = row.get("goal", "").strip()
            if not goal:
                continue

            # Assign category
            goal_lower = goal.lower()
            category = "other"
            for cat, keywords in category_keywords.items():
                if any(kw in goal_lower for kw in keywords):
                    category = cat
                    break

            prompts.append({
                "id": i,
                "prompt": goal,
                "category": category,
            })

    return prompts


def main():
    csv_path = download_advbench()
    prompts = parse_and_categorize(csv_path)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    # Print stats
    from collections import Counter
    cats = Counter(p["category"] for p in prompts)
    print(f"\nTotal prompts: {len(prompts)}")
    print("Category distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
