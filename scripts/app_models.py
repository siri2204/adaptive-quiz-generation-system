
import os
import re
import random
import tempfile
import time

import streamlit as st
import pdfplumber
from gpt4all import GPT4All

# ===================== PATHS =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_FILES = {
    "Nous-Hermes-2-Mistral-7B-DPO (Q5_K_M)": "Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf",
    "Meta-Llama-3-8B-Instruct (Q4_0)": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "Orca Mini 3B (Q4_0)": "q4_0-orca-mini-3b.gguf",
}

MODEL_GEN = {
    "Nous-Hermes-2-Mistral-7B-DPO (Q5_K_M)": {"max_tokens": 380, "temp": 0.45, "top_p": 0.9},
    "Meta-Llama-3-8B-Instruct (Q4_0)": {"max_tokens": 320, "temp": 0.45, "top_p": 0.9},
    "Orca Mini 3B (Q4_0)": "default",
}

DEFAULT_GEN = {"max_tokens": 300, "temp": 0.45, "top_p": 0.9}

# ===================== PAGE SETUP =====================
st.set_page_config(page_title="Adaptive AI Quiz Generator", layout="centered")
st.title("📘 Adaptive AI Quiz Generator")

# ===================== SESSION STATE =====================
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss("pdf_text", "")
ss("topics", [])
ss("topic_spans", {})
ss("selected_topics", [])

ss("difficulty", "Easy")
ss("difficulty_ui", "Easy")

ss("current_mcq", None)
ss("parsed_mcq", None)

ss("score", 0)
ss("total", 0)

ss("last_result", None)
ss("locked", False)

ss("model_choice", list(MODEL_FILES.keys())[0])

# timing: total LLM time per question (first gen + repair gen if any)
ss("last_llm_time_s", None)
ss("time_history", [])  # list of {model, difficulty, seconds, source}

# anti-repeat memory
ss("recent_questions", [])

# ===================== MODEL LOADING =====================
@st.cache_resource
def load_model(model_path: str):
    return GPT4All(model_path, allow_download=False)

def resolve_model_path(choice: str) -> str:
    fname = MODEL_FILES.get(choice)
    if not fname:
        raise FileNotFoundError(f"Unknown model choice: {choice}")
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found:\n{path}\n\n"
            f"Put the GGUF file in: {MODELS_DIR}\n"
            f"Or update MODEL_FILES in script/app.py"
        )
    return path

def get_gen_cfg(model_name: str):
    cfg = MODEL_GEN.get(model_name, "default")
    return DEFAULT_GEN if cfg == "default" else cfg

# ===================== PDF PROCESSING =====================
def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)

# ===================== TOPIC EXTRACTION =====================
def extract_topics(text):
    topics = []
    spans = {}
    lines = text.splitlines()
    offsets = []
    cur = 0

    for l in lines:
        offsets.append(cur)
        cur += len(l) + 1

    for i, l in enumerate(lines):
        if re.match(r"^\s*\d+\.\s+[A-Z]", l):
            topics.append(l.strip())

    for i, t in enumerate(topics):
        try:
            start_line_idx = lines.index(t)
            start = offsets[start_line_idx]
        except ValueError:
            continue

        if i + 1 < len(topics):
            try:
                next_line_idx = lines.index(topics[i + 1])
                end = offsets[next_line_idx]
            except ValueError:
                end = len(text)
        else:
            end = len(text)

        spans[t] = (start, end)

    return topics, spans

def get_selected_text(text):
    # full selected text (no truncation) — we will sample windows later
    if not st.session_state.selected_topics or not st.session_state.topic_spans:
        return text

    chunks = []
    for t in st.session_state.selected_topics:
        if t in st.session_state.topic_spans:
            a, b = st.session_state.topic_spans[t]
            chunks.append(text[a:b])

    merged = "\n".join(chunks).strip()
    return merged if merged else text

# ===================== ANTI-REPEAT + RANDOM WINDOW =====================
def normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^a-z0-9 ]+", "", q)
    return q

def recently_seen(q: str) -> bool:
    nq = normalize_question(q)
    return any(normalize_question(old) == nq for old in st.session_state.recent_questions[-10:])

def remember_question(q: str) -> None:
    if q and q.strip():
        st.session_state.recent_questions.append(q.strip())
        st.session_state.recent_questions = st.session_state.recent_questions[-15:]

def random_text_window(text: str, window: int = 1800) -> str:
    t = (text or "").strip()
    if len(t) <= window:
        return t
    start = random.randint(0, max(0, len(t) - window))
    return t[start:start + window]

# ===================== OUTPUT CLEANUP + VALIDATION =====================
def clean_llm_output(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("**", "")
    s = re.split(r"(?i)\b(let me know|hope this helps|if you need any further)\b", s)[0].strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def is_valid_mcq(raw: str) -> bool:
    if not re.search(r"(?im)^Question:\s+.+", raw):
        return False
    for k in ["A", "B", "C", "D"]:
        if not re.search(rf"(?im)^{k}[.)]\s+.+", raw):
            return False
    if not re.search(r"(?im)^Correct Answer:\s*[ABCD]\b", raw):
        return False
    if not re.search(r"(?im)^Explanation:\s+.+", raw):
        return False
    return True

# ===================== MCQ PARSING =====================
def parse_mcq(raw):
    q = ""
    opts = {}
    ans = ""
    exp = ""

    qm = re.search(r"Question:\s*(.+)", raw)
    if qm:
        q = qm.group(1).strip()

    for k in ["A", "B", "C", "D"]:
        m = re.search(rf"^{k}[.)]\s*(.+)$", raw, re.MULTILINE)
        if m:
            opts[k] = m.group(1).strip()

    am = re.search(r"Correct Answer:\s*([ABCD])", raw)
    if am:
        ans = am.group(1).strip().upper()

    em = re.search(r"Explanation:\s*(.+)", raw, re.DOTALL)
    if em:
        exp = em.group(1).strip()

    return {"question": q, "options": opts, "answer": ans, "explanation": exp}

def reconcile_answer_with_explanation(parsed: dict) -> dict:
    """
    If model gives inconsistent correct answer, choose the option that best
    matches the explanation using keyword overlap.
    """
    opts = parsed.get("options", {}) or {}
    exp = (parsed.get("explanation") or "").lower()
    current = (parsed.get("answer") or "").strip().upper()

    if not exp:
        return parsed

    for k in ["A", "B", "C", "D"]:
        if not opts.get(k):
            return parsed

    stop = {
        "the","a","an","and","or","to","of","in","is","are","that","this","it","as","on",
        "for","with","while","from","by","be","can","do","does","not","only","based","into"
    }
    exp_tokens = [t for t in re.findall(r"[a-z]+", exp) if t not in stop]
    exp_set = set(exp_tokens)

    def score(opt_text: str) -> int:
        t = (opt_text or "").lower()
        toks = [x for x in re.findall(r"[a-z]+", t) if x not in stop]
        return len(set(toks) & exp_set)

    scores = {k: score(opts.get(k, "")) for k in ["A", "B", "C", "D"]}
    best = max(scores, key=scores.get)

    if current not in ["A", "B", "C", "D"]:
        parsed["answer"] = best
        return parsed

    if scores.get(best, 0) >= scores.get(current, 0) + 2:
        parsed["answer"] = best

    return parsed

# ===================== REPAIR =====================
def repair_mcq(model, bad_raw: str, context: str, difficulty: str, top_p: float) -> str:
    prompt = f"""
Fix the MCQ below into EXACT required format and internal consistency.

REQUIRED FORMAT (EXACT):
Question: <question>
A. <option>
B. <option>
C. <option>
D. <option>
Correct Answer: <A/B/C/D>
Explanation: <1-3 sentences justifying the Correct Answer>

RULES:
- Output ONLY the MCQ (no greetings)
- Correct Answer MUST match the explanation
- Exactly one correct answer
- Based ONLY on the CONTENT
- No markdown symbols like **

Difficulty: {difficulty}

BAD MCQ:
{bad_raw}

CONTENT:
{context}
""".strip()

    with model.chat_session():
        fixed = model.generate(prompt, max_tokens=450, temp=0.2, top_p=top_p).strip()
    return clean_llm_output(fixed)

# ===================== GENERATORS =====================
def rule_based_mcq(_text):
    concept = random.choice(["Artificial Intelligence", "Machine Learning", "Narrow AI", "Neural Networks"])
    return f"""Question: What best describes {concept}?

A. A key concept discussed in the lecture
B. A hardware component
C. An obsolete system
D. An unrelated topic

Correct Answer: A
Explanation: {concept} is explained as a core idea in the lecture.
"""

def ai_mcq(full_text: str, difficulty: str):
    """
    Returns: raw_mcq, total_llm_time_seconds, ok
    TOTAL LLM time counts: first generate + repair generate (if used)
    Uses random context window + do-not-repeat list.
    """
    model_name = st.session_state.model_choice
    model_path = resolve_model_path(model_name)
    model = load_model(model_path)

    cfg = get_gen_cfg(model_name)
    max_tokens = int(cfg.get("max_tokens", 300))
    temp = float(cfg.get("temp", 0.45))
    top_p = float(cfg.get("top_p", 0.9))

    # change context each time
    context_text = random_text_window(full_text, window=1800)

    # do-not-repeat list
    prev_qs = st.session_state.recent_questions[-6:]
    prev_block = ""
    if prev_qs:
        prev_block = "DO NOT repeat these previous questions:\n" + "\n".join([f"- {q}" for q in prev_qs])

    seed = random.randint(1, 10_000_000)

    prompt = f"""
Return ONLY the MCQ. No extra words.

Create EXACTLY one NEW MCQ based ONLY on the content.

Difficulty: {difficulty}
Random seed: {seed}

FORMAT (EXACT):
Question: <question>
A. <option>
B. <option>
C. <option>
D. <option>
Correct Answer: <A/B/C/D>
Explanation: <1-3 sentences>

RULES:
- No markdown, no emojis, no greetings
- Exactly one correct answer
- Correct Answer must match Explanation
- Avoid repeating earlier questions (even if wording changes)

{prev_block}

CONTENT:
{context_text}
""".strip()

    total_llm = 0.0

    # first generation
    t0 = time.perf_counter()
    with model.chat_session():
        raw = model.generate(prompt, max_tokens=max_tokens, temp=temp, top_p=top_p).strip()
    total_llm += (time.perf_counter() - t0)

    raw = clean_llm_output(raw)

    if is_valid_mcq(raw):
        return raw, total_llm, True

    # repair once (count repair time too)
    t1 = time.perf_counter()
    fixed = repair_mcq(model, raw, context_text, difficulty, top_p=top_p)
    total_llm += (time.perf_counter() - t1)

    if is_valid_mcq(fixed):
        return fixed, total_llm, True

    return fixed, total_llm, False

def generate_new_question(generator_mode: str):
    full_context = get_selected_text(st.session_state.pdf_text)
    MAX_TRIES = 3

    with st.spinner("Generating question..."):
        if generator_mode == "AI":
            raw = None
            llm_time = None
            ok_final = False

            # try a few times to avoid repeats
            for _ in range(MAX_TRIES):
                cand_raw, cand_time, cand_ok = ai_mcq(full_context, st.session_state.difficulty)
                cand_parsed = parse_mcq(cand_raw)
                q_try = cand_parsed.get("question", "").strip()

                if cand_ok and q_try and not recently_seen(q_try):
                    raw = cand_raw
                    llm_time = cand_time
                    ok_final = True
                    break

            if not ok_final:
                raw = rule_based_mcq(full_context)
                llm_time = None
                source = "Fallback (AI failed or repeated)"
            else:
                source = "AI"

        else:
            raw = rule_based_mcq(full_context)
            llm_time = None
            source = "Baseline"

    st.session_state.current_mcq = raw

    parsed = parse_mcq(raw)
    parsed = reconcile_answer_with_explanation(parsed)
    st.session_state.parsed_mcq = parsed

    # remember question to prevent repeats
    remember_question(parsed.get("question", ""))

    # timing log (AI only)
    st.session_state.last_llm_time_s = llm_time
    if llm_time is not None:
        st.session_state.time_history.append({
            "model": st.session_state.model_choice,
            "difficulty": st.session_state.difficulty,
            "seconds": float(llm_time),
            "source": source,
        })

    # reset per-question state
    st.session_state.last_result = None
    st.session_state.locked = False

# ===================== UI =====================
st.subheader("Model")
st.session_state.model_choice = st.selectbox(
    "Choose local LLM",
    list(MODEL_FILES.keys()),
    index=list(MODEL_FILES.keys()).index(st.session_state.model_choice),
)

uploaded = st.file_uploader("Upload lecture notes (PDF)", type=["pdf"])
if uploaded:
    st.session_state.pdf_text = extract_text_from_pdf(uploaded)
    st.session_state.topics, st.session_state.topic_spans = extract_topics(st.session_state.pdf_text)
    st.success("PDF uploaded successfully!")

if not st.session_state.pdf_text:
    st.stop()

st.subheader("Optional topic selection")
st.session_state.selected_topics = st.multiselect("Topics", st.session_state.topics)

st.subheader("Difficulty")
st.session_state.difficulty_ui = st.radio(
    "Difficulty",
    ["Easy", "Medium", "Hard"],
    index=["Easy", "Medium", "Hard"].index(st.session_state.difficulty_ui),
    horizontal=True,
    key="difficulty_ui_radio",
)
st.session_state.difficulty = st.session_state.difficulty_ui

st.subheader("Question Generator")
generator_mode = st.radio("Generator", ["AI", "Baseline"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Question"):
        generate_new_question(generator_mode)
        st.rerun()

with col2:
    if st.button("Next Question", disabled=st.session_state.parsed_mcq is None):
        generate_new_question(generator_mode)
        st.rerun()

# ===================== QUIZ =====================
if st.session_state.parsed_mcq:
    mcq = st.session_state.parsed_mcq

    if st.session_state.last_llm_time_s is not None:
        st.caption(
            f"⏱ TOTAL LLM time: {st.session_state.last_llm_time_s:.2f}s "
            f"(Model: {st.session_state.model_choice})"
        )

    st.markdown(f"### {mcq['question']}")

    opt_keys = ["A", "B", "C", "D"]
    choice = st.radio(
        "Choose an answer",
        opt_keys,
        format_func=lambda k: f"{k}. {mcq['options'].get(k, '').strip()}",
        disabled=st.session_state.locked
    )

    if st.button("Submit Answer", disabled=st.session_state.locked):
        st.session_state.total += 1
        correct_ans = mcq.get("answer", "").upper().strip() or "A"

        if choice == correct_ans:
            st.session_state.score += 1
            was_correct = True
            if st.session_state.difficulty == "Easy":
                st.session_state.difficulty = "Medium"
            elif st.session_state.difficulty == "Medium":
                st.session_state.difficulty = "Hard"
        else:
            was_correct = False
            if st.session_state.difficulty == "Hard":
                st.session_state.difficulty = "Medium"
            elif st.session_state.difficulty == "Medium":
                st.session_state.difficulty = "Easy"

        st.session_state.last_result = {
            "was_correct": was_correct,
            "correct_answer": correct_ans,
            "explanation": mcq.get("explanation", "")
        }
        st.session_state.locked = True
        st.session_state.difficulty_ui = st.session_state.difficulty
        st.rerun()

    if st.session_state.last_result:
        r = st.session_state.last_result
        if r["was_correct"]:
            st.success("Correct ✅")
        else:
            st.error(f"Incorrect ❌ (Correct: {r['correct_answer']})")

        if r.get("explanation"):
            st.info(f"Explanation: {r['explanation']}")

        st.info(f"Score: {st.session_state.score}/{st.session_state.total}")
        st.caption(f"Next difficulty (adaptive): {st.session_state.difficulty}")

with st.expander("📊 Timing summary (this session)"):
    if not st.session_state.time_history:
        st.write("No timing data yet. Generate a few AI questions to see averages.")
    else:
        by_model = {}
        for r in st.session_state.time_history:
            by_model.setdefault(r["model"], []).append(r["seconds"])

        for m, arr in by_model.items():
            avg = sum(arr) / len(arr)
            st.write(f"- {m}: avg {avg:.2f}s over {len(arr)} question(s)")