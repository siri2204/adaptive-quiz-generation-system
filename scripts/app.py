# app.py
import streamlit as st
import pdfplumber
import tempfile
import os
import random
import re
from gpt4all import GPT4All

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Adaptive AI Quiz Generator",
    layout="centered"
)

st.title("📘 Adaptive AI Quiz Generator")

# -------------------- SESSION STATE --------------------
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Easy"

if "current_mcq" not in st.session_state:
    st.session_state.current_mcq = None

if "score" not in st.session_state:
    st.session_state.score = 0

if "total" not in st.session_state:
    st.session_state.total = 0

if "source" not in st.session_state:
    st.session_state.source = ""

# -------------------- LOAD MODEL (LOCAL, NO DOWNLOAD) --------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "orca_mini_3b.Q3_K_M.gguf"
    )
    return GPT4All(model_path, allow_download=False)

# -------------------- PDF UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload lecture notes (PDF)",
    type=["pdf"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:2]:
            extracted_text.append(page.extract_text() or "")

    st.session_state.pdf_text = " ".join(extracted_text)
    st.success("PDF uploaded successfully!")

    st.subheader("📄 Extracted Text Preview")
    st.text(st.session_state.pdf_text[:900])

# -------------------- STOP IF NO PDF --------------------
if not st.session_state.pdf_text.strip():
    st.info("Please upload a PDF to continue.")
    st.stop()

# -------------------- CONTROLS --------------------
st.subheader("Select difficulty")
st.session_state.difficulty = st.radio(
    "Difficulty",
    ["Easy", "Medium", "Hard"],
    label_visibility="collapsed"
)

st.subheader("Question generator")
generator = st.radio(
    "Generator",
    ["AI (GPT4All)", "Baseline (Rule-Based)"],
    label_visibility="collapsed"
)

# -------------------- RULE-BASED GENERATOR --------------------
def rule_based_question(text):
    concepts = ["variables", "Python", "machine learning", "neural networks"]
    concept = random.choice(concepts)

    return f"""Question: What best describes {concept} based on the lecture?

A. It is a key concept explained in the material  
B. It is a hardware component  
C. It is unrelated to the topic  
D. It is an obsolete technology  

Correct Answer: A
"""

# -------------------- AI GENERATOR (FIXED & ROBUST) --------------------
def ai_question(text, difficulty):
    model = load_model()

    prompt = f"""
You are an academic quiz generator.

Create EXACTLY ONE multiple-choice question based ONLY on the content below.

Difficulty: {difficulty}

FORMAT EXACTLY:

Question: <question>

A. <option>
B. <option>
C. <option>
D. <option>

Correct Answer: <A/B/C/D>

RULES:
- No explanations
- No meta-questions
- Exactly one correct answer

CONTENT:
{text[:700]}
"""

    with model.chat_session():
        output = model.generate(
            prompt,
            max_tokens=350,
            temp=0.2
        ).strip()

    # ---- RELAXED BUT SAFE VALIDATION ----
    has_correct = re.search(r"Correct Answer:\s*[ABCD]", output)
    options = re.findall(r"^[A-D][\.\)]\s+", output, re.MULTILINE)

    if not has_correct or len(options) < 4:
        raise ValueError("AI output invalid")

    return output

# -------------------- GENERATE QUESTION --------------------
if st.button(
    "Generate Question" if st.session_state.current_mcq is None else "Generate Next Question"
):
    with st.spinner("Generating question..."):
        try:
            if generator.startswith("AI"):
                mcq = ai_question(
                    st.session_state.pdf_text,
                    st.session_state.difficulty
                )
                source = "AI-generated (GPT4All)"
            else:
                mcq = rule_based_question(st.session_state.pdf_text)
                source = "Rule-Based"

        except Exception:
            mcq = rule_based_question(st.session_state.pdf_text)
            source = "Rule-Based (AI failed safely)"

    st.session_state.current_mcq = mcq
    st.session_state.source = source

# -------------------- DISPLAY QUESTION --------------------
if st.session_state.current_mcq:
    st.subheader("📝 Quiz Question")
    st.text(st.session_state.current_mcq)
    st.caption(f"Source: {st.session_state.source}")

    answer = st.radio(
        "Your answer",
        ["A", "B", "C", "D"]
    )

    if st.button("Submit Answer"):
        st.session_state.total += 1

        correct = re.search(
            r"Correct Answer:\s*([ABCD])",
            st.session_state.current_mcq
        ).group(1)

        if answer == correct:
            st.success("Correct ✅")
            st.session_state.score += 1
            if st.session_state.difficulty == "Easy":
                st.session_state.difficulty = "Medium"
            elif st.session_state.difficulty == "Medium":
                st.session_state.difficulty = "Hard"
        else:
            st.error(f"Incorrect ❌ (Correct: {correct})")
            if st.session_state.difficulty == "Hard":
                st.session_state.difficulty = "Medium"
            elif st.session_state.difficulty == "Medium":
                st.session_state.difficulty = "Easy"

        st.info(
            f"Score: {st.session_state.score}/{st.session_state.total} | "
            f"Next difficulty: {st.session_state.difficulty}"
        )

        st.session_state.current_mcq = None

