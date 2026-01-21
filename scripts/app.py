# app.py
import streamlit as st
import pdfplumber
import tempfile
import os
from gpt4all import GPT4All

st.set_page_config(page_title="AI Quiz Generator", layout="centered")
st.title("📘 AI-Powered Quiz Generator")

# Load model once
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "orca_mini_3b.Q3_K_M.gguf"
    )
    return GPT4All(model_path, allow_download=False)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_upload")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Extract text from first page and skip first 10 lines (metadata)
    with pdfplumber.open(temp_path) as pdf:
        text = ""
        page_text = pdf.pages[0].extract_text() or ""
        lines = page_text.splitlines()
        # Skip first 10 lines (title, authors, emails)
        if len(lines) > 10:
            lines = lines[10:]
        text = " ".join(lines)

    st.subheader("Extracted Text (preview)")
    st.text(text[:800])

    # Generate MCQ
    if st.button("Generate 1 MCQ"):
        with st.spinner("Generating MCQ..."):
            model = load_model()

            prompt = f"""
You are an educational quiz generator.

Based ONLY on the academic content below, create exactly ONE multiple-choice question that tests understanding of the material.

Rules:
- The question must be factual and content-based
- Provide 4 options labeled A, B, C, D
- Clearly indicate the correct answer at the end
- Do NOT ask meta-questions or repeat the title

Content:
{text[:700]}
"""

            try:
                with model.chat_session():
                    mcq = model.generate(prompt, max_tokens=400)

                # Parse model output to display neatly
                mcq_lines = mcq.strip().splitlines()
                if len(mcq_lines) < 2:
                    st.warning("⚠️ Model did not generate a proper MCQ. Try again or shorten PDF.")
                else:
                    st.subheader("Generated MCQ")
                    for line in mcq_lines:
                        st.text(line)

            except Exception as e:
                st.error(f"Error generating MCQ: {e}")

