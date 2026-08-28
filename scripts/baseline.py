# scripts/baseline.py
"""
Rule-based MCQ baseline (offline).

- Extracts lecture keywords using simple heuristics
- Filters out verbs / discourse words
- Inserts remaining keywords into fixed MCQ templates
- No LLMs, no UI, no adaptivity

This is intentionally shallow but clean and defensible.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from typing import List, Dict, Tuple

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# -------------------- CONFIG --------------------

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while",
    "is","are","was","were","be","been","being","to","of","in","on",
    "for","with","as","by","at","from","into","about","than","that",
    "this","these","those","it","its","they","their","we","our","you",
    "your","i","me","my","can","could","should","would","may","might",
    "will","do","does","did","done","not","no","yes","also","such"
}

# Generic discourse / filler words
DISCOURSE_WORDS = {
    "today","across","task","tasks","example","examples",
    "introduction","summary","overview","section","lecture",
    "figure","table","chapter","page","pages","slide","slides",
    "note","notes","content","contents","material","materials",
    "result","results","conclusion","conclusions","subset",
}

# Likely verb / adjective endings we want to avoid
BAD_SUFFIXES = (
    "ed", "ing", "es", "ly", "ive", "al", "able", "ible"
)

GENERIC_DISTRACTORS = [
    "a hardware-specific feature not discussed in the lecture",
    "an unrelated administrative procedure",
    "a historical topic outside the lecture scope",
]


# -------------------- TEXT IO --------------------

def extract_text_from_pdf(path: str) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed. Use --txt instead.")
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# -------------------- KEYWORD EXTRACTION --------------------

def is_bad_token(w: str) -> bool:
    if w in STOPWORDS or w in DISCOURSE_WORDS:
        return True
    if len(w) < 4:
        return True
    if w.isdigit():
        return True
    for suf in BAD_SUFFIXES:
        if w.endswith(suf):
            return True
    return False


def extract_keywords(text: str, k: int = 50) -> List[str]:
    """
    Heuristic keyword extraction:
    1) Prefer TitleCase multi-word phrases
    2) Fall back to frequent noun-like tokens
    """
    if not text:
        return []

    keywords: List[str] = []

    # --- 1. TitleCase phrases (likely concepts) ---
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
        phrase = m.group(1).strip()
        if phrase.lower() in DISCOURSE_WORDS:
            continue
        keywords.append(phrase)

    # --- 2. Frequent noun-like tokens ---
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    freq = Counter()

    for w in tokens:
        if not is_bad_token(w):
            freq[w] += 1

    keywords.extend([w.title() for w, _ in freq.most_common(k)])

    # Dedupe while preserving order
    seen = set()
    out = []
    for kw in keywords:
        key = kw.lower()
        if key not in seen:
            seen.add(key)
            out.append(kw)

    return out[:k]


# -------------------- BASELINE MCQ GENERATION --------------------

def _place_options(correct: str, distractors: List[str]) -> Tuple[Dict[str, str], str]:
    d = [x for x in distractors if x.strip()]
    while len(d) < 3:
        d.append(random.choice(GENERIC_DISTRACTORS))
    d = d[:3]

    pool = d + [correct]
    random.shuffle(pool)

    labels = ["A","B","C","D"]
    options = dict(zip(labels, pool))
    correct_label = next(k for k, v in options.items() if v == correct)
    return options, correct_label


def generate_one_mcq(text: str, difficulty: str, keywords: List[str]) -> str:
    term = random.choice(keywords) if keywords else "a key concept"

    correct = f"a concept discussed in the lecture related to {term}"

    distractors = []
    others = [k for k in keywords if k != term]
    random.shuffle(others)

    for o in others[:3]:
        distractors.append(f"a concept primarily concerned with {o}")

    options, correct_label = _place_options(correct, distractors)

    if difficulty == "Hard":
        q = f"Which option best characterizes {term} as described in the lecture?"
    elif difficulty == "Medium":
        q = f"What best describes {term} according to the lecture?"
    else:
        q = f"According to the lecture, what is {term}?"

    explanation = (
        f'The lecture text contains the term "{term}". '
        f"This rule-based baseline inserts extracted keywords into predefined templates."
    )

    return (
        f"Question: {q}\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n"
        f"Correct Answer: {correct_label}\n"
        f"Explanation: {explanation}"
    )


def generate_mcqs(text: str, difficulty: str, n: int) -> str:
    kws = extract_keywords(text)
    if not kws:
        kws = ["Artificial Intelligence"]

    return "\n\n".join(generate_one_mcq(text, difficulty, kws) for _ in range(n))


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser("Rule-based MCQ baseline")
    ap.add_argument("--pdf", type=str)
    ap.add_argument("--txt", type=str)
    ap.add_argument("--difficulty", choices=["Easy","Medium","Hard"], default="Easy")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", type=str)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.pdf:
        text = extract_text_from_pdf(args.pdf)
    elif args.txt:
        text = read_text(args.txt)
    else:
        raise SystemExit("Provide --pdf or --txt")

    text = clean_text(text)
    out = generate_mcqs(text, args.difficulty, args.n)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"[baseline] wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
