"""
Generate Chinese adaptive mixed version of AdvBench prompts.
Pattern: Chinese verbs/key words + English nouns/structure (mirrors the Korean adaptive mixed approach).

Usage:
  python3 generate_zh_adaptive_mixed.py
"""

import json
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")


def create_zh_adaptive_mixed(en_data):
    """Create Chinese adaptive mixed prompts: Chinese key verbs/nouns + English structure."""

    # Imperative verb replacements (sentence-initial action verbs)
    # These are applied with word boundary matching
    imperative_verbs = [
        (r'\bWrite\b', '编写'),
        (r'\bDevelop\b', '开发'),
        (r'\bCreate\b', '创建'),
        (r'\bGive\b', '给出'),
        (r'\bProvide\b', '提供'),
        (r'\bTeach\b', '教授'),
        (r'\bShow\b', '展示'),
        (r'\bProduce\b', '制作'),
        (r'\bExplain\b', '解释'),
        (r'\bDescribe\b', '描述'),
        (r'\bInstruct\b', '指导'),
        (r'\bPublish\b', '发布'),
        (r'\bDesign\b', '设计'),
        (r'\bGenerate\b', '生成'),
        (r'\bOutline\b', '概述'),
        (r'\bDemonstrate\b', '演示'),
        (r'\bSuggest\b', '建议'),
        (r'\bCompose\b', '撰写'),
        (r'\bDraft\b', '起草'),
        (r'\bBuild\b', '构建'),
        (r'\bMake\b', '制作'),
        (r'\bDiscuss\b', '讨论'),
        (r'\bIllustrate\b', '说明'),
    ]

    # Key noun replacements
    noun_map = [
        (r'\bscript\b', '脚本'),
        (r'\btutorial\b', '教程'),
        (r'\bwebsite\b', '网站'),
        (r'\bsocial media\b', '社交媒体'),
        (r'\bbomb\b', '炸弹'),
        (r'\bbombs\b', '炸弹'),
        (r'\bexplosive device\b', '爆炸装置'),
        (r'\bexplosives?\b', '爆炸物'),
        (r'\bidentity theft\b', '身份盗窃'),
        (r'\bguide\b', '指南'),
        (r'\bplan\b', '计划'),
        (r'\bprogram\b', '程序'),
        (r'\bvirus\b', '病毒'),
        (r'\bmalware\b', '恶意软件'),
        (r'\bransomware\b', '勒索软件'),
        (r'\bpersonal information\b', '个人信息'),
        (r'\bsecure network\b', '安全网络'),
        (r'\bgovernment\b', '政府'),
        (r'\bdatabase\b', '数据库'),
        (r'\bweapon\b', '武器'),
        (r'\bweapons\b', '武器'),
        (r'\bdrug\b', '毒品'),
        (r'\bdrugs\b', '毒品'),
        (r'\bfirearms?\b', '枪支'),
        (r'\bvulnerabilities\b', '漏洞'),
        (r'\bvulnerability\b', '漏洞'),
        (r'\bpassword\b', '密码'),
        (r'\bpasswords\b', '密码'),
        (r'\bpropaganda\b', '宣传'),
        (r'\bterrorism\b', '恐怖主义'),
        (r'\bterrorist\b', '恐怖分子'),
        (r'\bpersonal data\b', '个人数据'),
        (r'\bsensitive information\b', '敏感信息'),
    ]

    # Key verb replacements (non-initial)
    verb_map = [
        (r'\bexploit\b', '利用'),
        (r'\bexploiting\b', '利用'),
        (r'\bsteal\b', '窃取'),
        (r'\bstealing\b', '窃取'),
        (r'\bhack into\b', '入侵'),
        (r'\bhack\b', '入侵'),
        (r'\bhacking\b', '入侵'),
        (r'\bspread\b', '传播'),
        (r'\bspreading\b', '传播'),
        (r'\bspreads\b', '传播'),
        (r'\bdamage\b', '损害'),
        (r'\bdestroy\b', '摧毁'),
        (r'\bkill\b', '杀死'),
        (r'\bpoison\b', '毒害'),
        (r'\bmanipulate\b', '操纵'),
        (r'\bmanipulates\b', '操纵'),
        (r'\bdeceive\b', '欺骗'),
        (r'\bdeceiving\b', '欺骗'),
        (r'\bharm\b', '伤害'),
        (r'\bharming\b', '伤害'),
        (r'\battack\b', '攻击'),
        (r'\battacking\b', '攻击'),
        (r'\bbypass\b', '绕过'),
        (r'\bbypassing\b', '绕过'),
        (r'\binfect\b', '感染'),
        (r'\binfecting\b', '感染'),
    ]

    # Conjunction replacements
    conj_map = [
        (r'\band\b', '和'),
        (r'\bor\b', '或者'),
    ]

    # Remove articles
    article_map = [
        (r'\ba\b', ''),
        (r'\ban\b', ''),
        (r'\bthe\b', ''),
    ]

    zh_mixed_data = []
    for item in en_data:
        text = item["prompt"]

        # 1. Apply imperative verbs (case-sensitive for sentence-initial)
        for pattern, replacement in imperative_verbs:
            text = re.sub(pattern, replacement, text)

        # 2. Apply key nouns (case-insensitive)
        for pattern, replacement in noun_map:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 3. Apply key verbs (case-insensitive)
        for pattern, replacement in verb_map:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 4. Apply conjunctions (case-insensitive)
        for pattern, replacement in conj_map:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 5. Remove articles (case-insensitive)
        for pattern, replacement in article_map:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean up extra spaces
        text = re.sub(r'  +', ' ', text).strip()

        zh_mixed_data.append({
            "id": item["id"],
            "prompt_en": item["prompt"],
            "prompt_zh": text,
            "category": item["category"],
        })

    return zh_mixed_data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load English prompts
    en_path = os.path.join(DATA_DIR, "advbench_en.json")
    with open(en_path) as f:
        en_data = json.load(f)

    print(f"Loaded {len(en_data)} English prompts")

    # Generate Chinese adaptive mixed
    zh_mixed_data = create_zh_adaptive_mixed(en_data)

    # Save
    out_path = os.path.join(DATA_DIR, "advbench_harmful_behaviors_translated_zh_adaptive_mixed.json")
    with open(out_path, "w") as f:
        json.dump(zh_mixed_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path} ({len(zh_mixed_data)} prompts)")

    # Show examples
    print("\n--- Examples ---")
    for i in [0, 1, 3, 12, 80, 150, 300, 450]:
        if i < len(zh_mixed_data):
            d = zh_mixed_data[i]
            print(f"[{d['id']}] EN: {d['prompt_en'][:90]}")
            print(f"     ZH: {d['prompt_zh'][:90]}")
            print()


if __name__ == "__main__":
    main()
