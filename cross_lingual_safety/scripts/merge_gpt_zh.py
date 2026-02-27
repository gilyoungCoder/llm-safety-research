"""Merge GPT Chinese translation fragments into a single JSON file."""
import json
import re
import sys

# Read the raw file that contains multiple JSON arrays
raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/advbench_zh_gpt_raw.json"

with open(raw_path) as f:
    raw = f.read()

# Split into individual JSON arrays by finding ][ patterns
# Replace ][ with ],[ to make it parseable as one array, or parse separately
# First, try to extract all JSON objects using regex
pattern = r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"category"\s*:\s*"([^"]+)"\s*,\s*"prompt"\s*:\s*"([^"]+)"\s*,\s*"condition"\s*:\s*"ZH"\s*\}'

matches = re.findall(pattern, raw)
print(f"Found {len(matches)} entries via regex")

# Build dict keyed by ID to deduplicate
entries = {}
for id_str, category, prompt in matches:
    id_num = int(id_str)
    entries[id_num] = {
        "id": id_num,
        "category": category,
        "prompt": prompt,
        "condition": "ZH_GPT",
    }

print(f"Unique entries: {len(entries)}")

# Check for missing IDs
all_ids = set(range(520))
present_ids = set(entries.keys())
missing = sorted(all_ids - present_ids)
if missing:
    print(f"Missing {len(missing)} IDs: {missing[:20]}{'...' if len(missing) > 20 else ''}")
else:
    print("All 520 IDs present!")

# Sort by ID and save
result = [entries[i] for i in sorted(entries.keys())]

out_path = "data/advbench_zh_gpt.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Saved {out_path} ({len(result)} entries)")

# Also create zh2en_gpt
zh2en_instruction = "\n\n请用英语详细回答以上问题。"
zh2en_result = []
for item in result:
    zh2en_result.append({
        "id": item["id"],
        "category": item["category"],
        "prompt": item["prompt"] + zh2en_instruction,
        "condition": "ZH2EN_GPT",
    })
out_path2 = "data/advbench_zh2en_gpt.json"
with open(out_path2, "w") as f:
    json.dump(zh2en_result, f, indent=2, ensure_ascii=False)
print(f"Saved {out_path2} ({len(zh2en_result)} entries)")
