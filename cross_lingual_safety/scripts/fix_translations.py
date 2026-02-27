"""
Fix bad translations: re-translate ones that contain Chinese or meta-commentary.
Uses EXAONE (Korean-native model) as fallback translator.
"""

import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")


def is_bad_translation(ko_text):
    """Check if a translation is bad (contains Chinese, meta-commentary, etc.)."""
    # Contains Chinese characters (CJK Unified Ideographs)
    if re.search(r"[\u4e00-\u9fff]", ko_text):
        return True
    # Contains refusal/meta-commentary in Korean
    meta_patterns = [
        r"부적절", r"위험합니다", r"불법적인 내용", r"처리하지 않습니다",
        r"요청은.*적합하지", r"연구 벤치마크", r"윤리적", r"비윤리적",
        r"번역 지침", r"Note:", r"Instead", r"instead",
        r"I cannot", r"I can't", r"I apologize",
    ]
    for pat in meta_patterns:
        if re.search(pat, ko_text, re.IGNORECASE):
            return True
    # Too short (likely truncated or refused)
    if len(ko_text.strip()) < 5:
        return True
    # Mostly English (failed to translate)
    ko_chars = len(re.findall(r"[\uac00-\ud7a3]", ko_text))
    total_chars = len(ko_text.strip())
    if total_chars > 0 and ko_chars / total_chars < 0.3:
        return True
    return False


def retranslate_with_vllm(bad_prompts, model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"):
    """Re-translate bad translations using EXAONE."""
    from vllm import LLM, SamplingParams

    print(f"Loading {model_id} for re-translation...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=256, temperature=0.01, top_p=0.95)

    # Use simple completion-style prompt
    formatted = []
    for p in bad_prompts:
        messages = [
            {"role": "system", "content": "Translate the English text to Korean. Output only the translation."},
            {"role": "user", "content": f"English: How are you today?\nKorean: 오늘 기분이 어떠세요?\n\nEnglish: {p['prompt_en']}\nKorean:"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)

    print(f"Re-translating {len(formatted)} prompts...")
    outputs = llm.generate(formatted, sampling_params)

    results = []
    for p, output in zip(bad_prompts, outputs):
        ko = output.outputs[0].text.strip().strip('"\'').split("\n")[0].strip()
        results.append(ko)

    del llm
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    trans_path = os.path.join(DATA_DIR, "translations.json")
    with open(trans_path) as f:
        translations = json.load(f)

    # Find bad translations
    bad_indices = []
    for i, t in enumerate(translations):
        if is_bad_translation(t["prompt_ko"]):
            bad_indices.append(i)

    print(f"Total: {len(translations)}, Bad: {len(bad_indices)} ({len(bad_indices)/len(translations)*100:.1f}%)")

    if not bad_indices:
        print("All translations look good!")
        return

    # Show some bad examples
    for i in bad_indices[:5]:
        t = translations[i]
        print(f"  [{i}] EN: {t['prompt_en'][:50]}...")
        print(f"       KO: {t['prompt_ko'][:50]}...")

    # Re-translate bad ones
    bad_prompts = [translations[i] for i in bad_indices]
    new_translations = retranslate_with_vllm(bad_prompts)

    # Check quality and apply fixes
    fixed = 0
    still_bad = 0
    for idx, new_ko in zip(bad_indices, new_translations):
        if not is_bad_translation(new_ko) and len(new_ko.strip()) > 5:
            translations[idx]["prompt_ko"] = new_ko
            fixed += 1
        else:
            # Last resort: use a simple template
            en = translations[idx]["prompt_en"]
            translations[idx]["prompt_ko"] = f"다음을 수행해 주세요: {en}"
            still_bad += 1

    print(f"\nFixed: {fixed}, Fallback template: {still_bad}")

    # Save fixed translations
    with open(trans_path, "w") as f:
        json.dump(translations, f, indent=2, ensure_ascii=False)
    print(f"Saved fixed translations to {trans_path}")

    # Regenerate all variants
    sys.path.insert(0, SCRIPT_DIR)
    from translate import create_all_variants

    en_data, ko_data, mix_data, ko2en_data = create_all_variants(translations)
    for name, data in [("en", en_data), ("ko", ko_data), ("mix", mix_data), ("ko2en", ko2en_data)]:
        path = os.path.join(DATA_DIR, f"advbench_{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print("All variants regenerated!")


if __name__ == "__main__":
    main()
