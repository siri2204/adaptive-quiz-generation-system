# app.py
import json
import re
import tempfile
import time
import hashlib
import random
from collections import deque
from datetime import date

import streamlit as st
import pdfplumber
import gemini_api as gemini_api

# -------------------- FREE-TIER SAFETY SETTINGS --------------------
LLM_DAILY_LIMIT = 30
LLM_MAX_CALLS_PER_CLICK = 2
VERIFY_IF_BUDGET = True
RETRY_ON_503 = 2

TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "500", "INTERNAL")
QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE")


# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Adaptive AI Quiz Generator", layout="centered")
st.title("📘 Adaptive AI Quiz Generator")


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def harder(level: str) -> str:
    return {"Easy": "Medium", "Medium": "Hard", "Hard": "Hard"}.get(level, "Easy")


def easier(level: str) -> str:
    return {"Hard": "Medium", "Medium": "Easy", "Easy": "Easy"}.get(level, "Easy")


# -------------------- REPEAT-PREVENTION HELPERS --------------------
def normalize_question(q: str) -> str:
    if not q:
        return ""
    q = q.strip().lower()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def fingerprint_question(q: str) -> str:
    norm = normalize_question(q)
    if not norm:
        return ""
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def format_avoid_list(recent_questions, max_items: int = 20) -> str:
    if not recent_questions:
        return ""
    items = list(recent_questions)[-max_items:]
    return "\n".join([f"- {q}" for q in items if q.strip()])


# --------------------  TOPIC EXTRACTION (NEW FEATURE) --------------------
def extract_topics(text: str, max_topics: int = 25):
    """
    Extract topics using section headings + key phrases.
    Heuristic, fast, and user-friendly (no extra LLM calls).
    """
    if not text:
        return []

    # Work on a reasonable slice for speed + stability
    sample = text[:30000]

    # Split into lines if possible (pdfplumber sometimes returns space-joined text,
    # but headings still often appear near numbering patterns)
    lines = re.split(r"[\n\r]+", sample)

    candidates = []

    # 1) Numbered headings: "1. Introduction", "2 Types of AI", "3.1 Something"
    for line in lines:
        s = (line or "").strip()
        if not s:
            continue
        m = re.match(r"^(\d+(\.\d+)*)[\.\)]?\s+(.+)$", s)
        if m:
            title = m.group(3).strip()
            # keep short-ish titles
            if 3 <= len(title) <= 80:
                candidates.append(title)

    # 2) Common key phrases / headings-like patterns from running text:
    # detect "X:" patterns (e.g., "Machine Learning:")
    for m in re.finditer(r"\b([A-Z][A-Za-z\s]{2,40}):", sample):
        candidates.append(m.group(1).strip())

    # Clean + dedupe
    cleaned = []
    seen = set()
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        c = re.sub(r"[•\-\–\—]+", "", c).strip()
        if len(c) < 3:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(c)
        if len(cleaned) >= max_topics:
            break

    return cleaned


def get_text_for_topics(full_text: str, selected_topics, window_chars: int = 4500):
    """
    Build a focused text blob around occurrences of each selected topic.
    This is simple and keeps UI friendly without complex segmentation.
    """
    if not selected_topics:
        return full_text

    full_lower = full_text.lower()
    chunks = []
    for topic in selected_topics:
        t = (topic or "").strip()
        if not t:
            continue
        idx = full_lower.find(t.lower())
        if idx == -1:
            continue
        start = max(0, idx - window_chars // 3)
        end = min(len(full_text), idx + window_chars)
        chunks.append(full_text[start:end])

    # If none found, fallback to full text
    if not chunks:
        return full_text

    # Join with separators (helps LLM distinguish)
    return "\n\n---\n\n".join(chunks)


# -------------------- LLM BUDGET + ERROR HANDLING --------------------
def _reset_daily_budget_if_needed():
    today = str(date.today())
    if "budget_day" not in st.session_state:
        st.session_state.budget_day = today
        st.session_state.llm_calls_today = 0
    if st.session_state.budget_day != today:
        st.session_state.budget_day = today
        st.session_state.llm_calls_today = 0


def _can_spend_calls(n: int) -> bool:
    _reset_daily_budget_if_needed()
    return (st.session_state.llm_calls_today + n) <= LLM_DAILY_LIMIT


def _spend_calls(n: int):
    st.session_state.llm_calls_today += n


def _is_transient(e: Exception) -> bool:
    msg = str(e).upper()
    return any(m in msg for m in TRANSIENT_MARKERS)


def _is_quota(e: Exception) -> bool:
    msg = str(e).upper()
    return any(m in msg for m in QUOTA_MARKERS)


def safe_api_call(prompt: str, *, retries: int = RETRY_ON_503) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            return gemini_api.api_call(prompt)
        except Exception as e:
            last_err = e
            if _is_quota(e):
                raise RuntimeError("QUOTA") from e

            if not _is_transient(e) or attempt == retries:
                raise

            sleep_s = (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

    raise last_err


# -------------------- SESSION STATE --------------------
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Easy"

if "user_difficulty" not in st.session_state:
    st.session_state.user_difficulty = st.session_state.difficulty

if "use_user_override" not in st.session_state:
    st.session_state.use_user_override = False

if "score" not in st.session_state:
    st.session_state.score = 0

if "total" not in st.session_state:
    st.session_state.total = 0

if "source" not in st.session_state:
    st.session_state.source = "AI-generated (Gemini)"

if "mcqs" not in st.session_state:
    st.session_state.mcqs = []

if "mcq_index" not in st.session_state:
    st.session_state.mcq_index = 0

if "submitted" not in st.session_state:
    st.session_state.submitted = {}

if "last_gen_time" not in st.session_state:
    st.session_state.last_gen_time = None

if "set_correct" not in st.session_state:
    st.session_state.set_correct = 0

if "set_total" not in st.session_state:
    st.session_state.set_total = 0

if "last_set_summary" not in st.session_state:
    st.session_state.last_set_summary = ""

if "last_raw_generation" not in st.session_state:
    st.session_state.last_raw_generation = ""

if "asked_fingerprints" not in st.session_state:
    st.session_state.asked_fingerprints = set()

if "asked_questions_recent" not in st.session_state:
    st.session_state.asked_questions_recent = deque(maxlen=50)

# Topic state
if "topics" not in st.session_state:
    st.session_state.topics = []
if "selected_topics" not in st.session_state:
    st.session_state.selected_topics = []

_reset_daily_budget_if_needed()


# -------------------- PDF UPLOAD --------------------
uploaded_file = st.file_uploader("Upload lecture notes (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text.append(page.extract_text() or "")

    st.session_state.pdf_text = " ".join(extracted_text)
    st.success("PDF uploaded successfully!")

    st.subheader("📄 Extracted Text Preview")
    st.text(st.session_state.pdf_text[:900])

    # Extract topics right after upload
    st.session_state.topics = extract_topics(st.session_state.pdf_text)
    # Reset selection on new upload
    st.session_state.selected_topics = []

if not st.session_state.pdf_text.strip():
    st.info("Please upload a PDF to continue.")
    st.stop()


# -------------------- ✅ TOPIC UI (NEW) --------------------
st.subheader("Topics (optional)")
if st.session_state.topics:
    st.session_state.selected_topics = st.multiselect(
        "Select one or more topics for a focused quiz (leave empty for full lecture)",
        options=st.session_state.topics,
        default=st.session_state.selected_topics,
    )
else:
    st.caption("No clear section headings detected — quiz will cover full lecture.")


# -------------------- CONTROLS --------------------
st.subheader("Difficulty")

colX, colY = st.columns([2, 3])
with colX:
    st.caption("Adaptive difficulty (system)")
    st.write(f"**{st.session_state.difficulty}**")

with colY:
    st.session_state.use_user_override = st.checkbox(
        "Manually override difficulty for next generation",
        value=st.session_state.use_user_override,
    )
    st.session_state.user_difficulty = st.radio(
        "Select difficulty",
        ["Easy", "Medium", "Hard"],
        index=["Easy", "Medium", "Hard"].index(st.session_state.user_difficulty),
        label_visibility="collapsed",
        disabled=not st.session_state.use_user_override,
    )

st.metric("Score", f"{st.session_state.score}/{st.session_state.total}")
st.caption(f"LLM calls today: {st.session_state.llm_calls_today}/{LLM_DAILY_LIMIT}")

if st.session_state.last_set_summary:
    st.info(st.session_state.last_set_summary)


# -------------------- PARSER --------------------
def parse_mcqs(raw: str):
    if not raw or not isinstance(raw, str):
        return []

    blocks = re.split(r"\bQuestion:\s*", raw)
    blocks = [b.strip() for b in blocks if b.strip()]

    mcqs = []
    for b in blocks:
        m = re.search(
            r"^(?P<q>.*?)(?:\n|$)\s*"
            r"A[\.\)]\s*(?P<A>.*?)(?:\n|$)\s*"
            r"B[\.\)]\s*(?P<B>.*?)(?:\n|$)\s*"
            r"C[\.\)]\s*(?P<C>.*?)(?:\n|$)\s*"
            r"D[\.\)]\s*(?P<D>.*?)(?:\n|$)\s*"
            r"Correct\s*Answer:\s*(?P<ans>[ABCD])\b\s*"
            r"(?:\n|$)\s*"
            r"(?:Explanation:\s*(?P<exp>.*))?$",
            b,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not m:
            continue

        q = (m.group("q") or "").strip()
        opts = {
            "A": (m.group("A") or "").strip(),
            "B": (m.group("B") or "").strip(),
            "C": (m.group("C") or "").strip(),
            "D": (m.group("D") or "").strip(),
        }
        ans = (m.group("ans") or "").upper().strip()
        exp = (m.group("exp") or "").strip()

        if not q or ans not in opts:
            continue

        mcqs.append({"question": q, "options": opts, "correct": ans, "explanation": exp})

    return mcqs


# -------------------- GEMINI GENERATION --------------------
def gemini_generate_3(text, difficulty, avoid_questions_text: str = ""):
    avoid_block = ""
    if avoid_questions_text.strip():
        avoid_block = f"""
DO NOT repeat or paraphrase any of these previously asked questions:
{avoid_questions_text}
"""

    prompt = f"""
You are an academic quiz generator.
Create EXACTLY THREE multiple-choice questions based ONLY on the content below.
Difficulty: {difficulty}

{avoid_block}

FORMAT (repeat 3 times exactly):
Question: <question>
A. <option>
B. <option>
C. <option>
D. <option>
Correct Answer: <A/B/C/D>
Explanation: <1-2 sentences, consistent with the Correct Answer letter>

RULES:
- No numbering
- No extra text before/after the 3 blocks
- Ensure the questions are distinct from each other
- Ensure the letter matches the explanation (no contradictions)

CONTENT:
{text}
"""
    t0 = time.perf_counter()
    out = safe_api_call(prompt)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def gemini_verify_answers(text, mcqs):
    payload = []
    for idx, q in enumerate(mcqs):
        payload.append(
            {
                "id": idx,
                "question": q["question"],
                "options": q["options"],
                "given_correct": q["correct"],
                "given_explanation": q.get("explanation", ""),
            }
        )

    prompt = f"""
You are a strict academic verifier.

Given CONTENT and a list of MCQs, verify each question's correct answer letter.
Return ONLY valid JSON (no markdown) in the following format:

[
  {{"id":0, "correct":"A", "explanation":"..."}} ,
  {{"id":1, "correct":"C", "explanation":"..."}} ,
  {{"id":2, "correct":"B", "explanation":"..."}}
]

Rules:
- "correct" must be one of A/B/C/D
- Explanation must justify that letter using ONLY the CONTENT
- If the provided correct letter is wrong or inconsistent, fix it

CONTENT:
{text}

MCQS:
{json.dumps(payload, ensure_ascii=False)}
"""
    verified_raw = safe_api_call(prompt)

    try:
        verified = json.loads(verified_raw)
        by_id = {item["id"]: item for item in verified if "id" in item}
    except Exception:
        return mcqs

    for idx, q in enumerate(mcqs):
        item = by_id.get(idx)
        if not item:
            continue
        corr = (item.get("correct") or "").strip().upper()
        exp = (item.get("explanation") or "").strip()
        if corr in q["options"]:
            q["correct"] = corr
        if exp:
            q["explanation"] = exp

    return mcqs


def dedupe_mcqs_against_history(mcqs):
    unique = []
    seen_in_batch = set()
    for q in mcqs:
        fp = fingerprint_question(q.get("question", ""))
        if not fp:
            continue
        if fp in st.session_state.asked_fingerprints:
            continue
        if fp in seen_in_batch:
            continue
        seen_in_batch.add(fp)
        unique.append(q)
    return unique


# -------------------- GENERATE BUTTON --------------------
if st.button("Generate Quiz Set (3 Questions)"):
    with st.spinner("Generating..."):
        try:
            effective_difficulty = (
                st.session_state.user_difficulty
                if st.session_state.use_user_override
                else st.session_state.difficulty
            )

            # use topic-focused text if topics selected
            generation_text = get_text_for_topics(
                st.session_state.pdf_text,
                st.session_state.selected_topics
            )

            calls_this_click = 0
            if not _can_spend_calls(1):
                raise RuntimeError("DAILY_LIMIT")

            max_tries = 6
            collected = []
            total_elapsed = 0.0

            for attempt in range(max_tries):
                if calls_this_click >= LLM_MAX_CALLS_PER_CLICK:
                    break

                avoid_questions = list(st.session_state.asked_questions_recent)
                avoid_questions += [q["question"] for q in collected if q.get("question")]
                avoid_text = format_avoid_list(avoid_questions, max_items=20)

                if not _can_spend_calls(1):
                    raise RuntimeError("DAILY_LIMIT")
                _spend_calls(1)
                calls_this_click += 1

                raw, elapsed = gemini_generate_3(
                    generation_text,  # focused lecture text
                    effective_difficulty,
                    avoid_questions_text=avoid_text,
                )
                total_elapsed += elapsed
                st.session_state.last_raw_generation = raw

                parsed = parse_mcqs(raw)
                if len(parsed) < 1:
                    continue

                parsed = dedupe_mcqs_against_history(parsed)

                collected_fps = {fingerprint_question(q["question"]) for q in collected}
                for q in parsed:
                    fp = fingerprint_question(q.get("question", ""))
                    if fp and fp not in collected_fps:
                        collected.append(q)
                        collected_fps.add(fp)
                    if len(collected) >= 3:
                        break

                if len(collected) >= 3:
                    break

            if len(collected) < 3:
                raise ValueError(
                    f"Could not generate 3 unique questions within the free-tier budget "
                    f"({calls_this_click} call(s) used this click). Try again."
                )

            st.session_state.mcqs = collected[:3]

            if VERIFY_IF_BUDGET:
                if _can_spend_calls(1) and calls_this_click < LLM_MAX_CALLS_PER_CLICK:
                    _spend_calls(1)
                    calls_this_click += 1
                    # verifier should use the same focused text context
                    st.session_state.mcqs = gemini_verify_answers(generation_text, st.session_state.mcqs)

            st.session_state.mcq_index = 0
            st.session_state.last_gen_time = total_elapsed
            st.session_state.submitted = {}
            st.session_state.set_correct = 0
            st.session_state.set_total = 0
            st.session_state.last_set_summary = ""

            for q in st.session_state.mcqs:
                qtext = (q.get("question") or "").strip()
                fp = fingerprint_question(qtext)
                if fp:
                    st.session_state.asked_fingerprints.add(fp)
                if qtext:
                    st.session_state.asked_questions_recent.append(qtext)

            safe_rerun()

        except RuntimeError as e:
            if str(e) == "DAILY_LIMIT":
                st.warning(
                    f"Daily LLM limit reached ({st.session_state.llm_calls_today}/{LLM_DAILY_LIMIT}). "
                    f"Try again tomorrow or increase your self-imposed limit."
                )
            elif str(e) == "QUOTA":
                st.warning("API quota/rate limit hit (429/RESOURCE_EXHAUSTED). Try later.")
            else:
                st.error(f"Generator Error: {e}")

            st.session_state.mcqs = []
            st.session_state.mcq_index = 0
            st.session_state.last_gen_time = None

        except Exception as e:
            msg = str(e).upper()
            if any(m in msg for m in TRANSIENT_MARKERS):
                st.warning("Model overloaded (503/UNAVAILABLE). Try again in a moment.")
            elif any(m in msg for m in QUOTA_MARKERS):
                st.warning("API quota/rate limit hit. Try later.")
            else:
                st.error(f"Generator Error: {e}")

            st.session_state.mcqs = []
            st.session_state.mcq_index = 0
            st.session_state.last_gen_time = None


if st.session_state.last_gen_time is not None:
    st.caption(f"⏱️ LLM generation time: {st.session_state.last_gen_time:.2f} seconds")


# -------------------- DISPLAY QUESTIONS --------------------
if st.session_state.mcqs:
    i = st.session_state.mcq_index
    mcq = st.session_state.mcqs[i]

    st.subheader(f"📝 Quiz Question ({i+1}/{len(st.session_state.mcqs)})")
    st.write(mcq["question"])
    st.caption(f"Source: {st.session_state.source}")

    option_labels = [f"{k}. {mcq['options'][k]}" for k in ["A", "B", "C", "D"]]
    already_answered = i in st.session_state.submitted

    with st.form(key=f"answer_form_{i}", clear_on_submit=False):
        chosen = st.radio(
            "Your answer",
            option_labels,
            index=None,
            disabled=already_answered,
            key=f"answer_choice_{i}",
        )
        submitted_now = st.form_submit_button("Submit Answer", disabled=already_answered)

    if submitted_now:
        if not chosen:
            st.warning("Please select an option first.")
        else:
            user_letter = chosen.split(".")[0].strip()
            correct_letter = mcq["correct"]
            is_correct = (user_letter == correct_letter)

            st.session_state.total += 1
            if is_correct:
                st.session_state.score += 1

            st.session_state.set_total += 1
            if is_correct:
                st.session_state.set_correct += 1

            st.session_state.submitted[i] = {
                "user": user_letter,
                "correct": correct_letter,
                "is_correct": is_correct,
            }
            safe_rerun()

    if already_answered:
        result = st.session_state.submitted[i]
        correct_letter = result["correct"]
        correct_text = mcq["options"].get(correct_letter, "")

        if result["is_correct"]:
            st.success(f"Correct ✅ (You chose {result['user']})")
        else:
            st.error(
                f"Incorrect ❌ (You chose {result['user']}; "
                f"Correct: {correct_letter} — {correct_text})"
            )

        # updated UI text (system difficulty)
        st.info(
            f"Score: {st.session_state.score}/{st.session_state.total} | "
            f"Set score: {st.session_state.set_correct}/{st.session_state.set_total}  "
        )

        # show generation difficulty when manual override is ON
        if st.session_state.use_user_override:
            st.caption(
                f"Generation difficulty (manual override): {st.session_state.user_difficulty}"
            )

        exp_text = (mcq.get("explanation") or "").strip()
        if exp_text:
            st.markdown("**Explanation:** " + exp_text)
        else:
            st.markdown("**Explanation:** _(No explanation returned by the model.)_")

        colA, colB = st.columns(2)
        with colA:
            if i > 0 and st.button("⬅ Previous"):
                st.session_state.mcq_index -= 1
                safe_rerun()

        with colB:
            if i < len(st.session_state.mcqs) - 1:
                if st.button("Next ➡"):
                    st.session_state.mcq_index += 1
                    safe_rerun()
            else:
                if st.button("Finish ✅"):
                    set_correct = st.session_state.set_correct
                    set_total = max(st.session_state.set_total, 1)
                    prev = st.session_state.difficulty

                    if set_correct >= 3:
                        st.session_state.difficulty = harder(prev)
                        change = "⬆️ Increased"
                    elif set_correct == 2:
                        st.session_state.difficulty = prev
                        change = "➡️ Kept the same"
                    else:
                        st.session_state.difficulty = easier(prev)
                        change = "⬇️ Decreased"

                    st.session_state.user_difficulty = st.session_state.difficulty

                    
                    if st.session_state.use_user_override:
                        st.session_state.last_set_summary = (
                            f"Set finished: {set_correct}/{set_total} correct. "
                            f"{change} difficulty for the next set."
                        )
                    else:
                        st.session_state.last_set_summary = (
                            f"Set finished: {set_correct}/{set_total} correct. "
                            f"{change} difficulty for the next set: {prev} → {st.session_state.difficulty}."
                        )

                    st.session_state.mcqs = []
                    st.session_state.mcq_index = 0
                    st.session_state.submitted = {}
                    st.session_state.last_gen_time = None
                    st.session_state.set_correct = 0
                    st.session_state.set_total = 0

                    safe_rerun()
else:
    st.caption("Click **Generate Quiz Set (3 Questions)** to start.")
