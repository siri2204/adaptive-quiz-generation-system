# generate_mcqs_from_pdf.py (upgraded)
import os
import pdfplumber
import pandas as pd
from gpt4all import GPT4All

# ===== Paths =====
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "..", "data")  # PDFs folder
model_path = os.path.join(current_dir, "..", "models", "orca_mini_3b.Q3_K_M.gguf")  # GGUF model
output_csv = os.path.join(current_dir, "..", "outputs", "mcqs_output.csv")  # CSV output

# Ensure outputs folder exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ===== Load model =====
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print("Loading local model...")
model = GPT4All(model_path, allow_download=False)
print("Model loaded successfully!\n")

# ===== Get PDFs =====
pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
if not pdf_files:
    print("No PDFs found in the data folder.")
    exit()

print("PDFs found:", pdf_files)

all_mcqs = []

# ===== Process PDFs =====
for pdf_file in pdf_files:
    pdf_path = os.path.join(data_folder, pdf_file)
    print(f"\nProcessing PDF: {pdf_file}")

    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    print(f"Extracted {len(text)} characters.")

    # Generate 1 MCQ per PDF (fast for testing)
    prompt = f"Generate 1 multiple-choice question from the following text:\n{text[:2000]}"
    with model.chat_session():
        raw_mcq = model.generate(prompt, max_tokens=400).strip()

    # ===== Parse into structured columns =====
    question = ""
    options = {"A": "", "B": "", "C": "", "D": ""}
    correct = ""

    lines = raw_mcq.split("\n")
    if lines:
        question = lines[0].strip()
        for line in lines[1:]:
            line_lower = line.lower()
            if line_lower.startswith("a)"):
                options["A"] = line[2:].strip()
            elif line_lower.startswith("b)"):
                options["B"] = line[2:].strip()
            elif line_lower.startswith("c)"):
                options["C"] = line[2:].strip()
            elif line_lower.startswith("d)"):
                options["D"] = line[2:].strip()
            elif "answer:" in line_lower:
                correct = line.split("Answer:")[-1].strip()

    # Save MCQ
    all_mcqs.append({
        "PDF": pdf_file,
        "MCQ_Number": 1,
        "Question": question,
        "Option A": options["A"],
        "Option B": options["B"],
        "Option C": options["C"],
        "Option D": options["D"],
        "Correct Answer": correct
    })
    print(f"Generated MCQ for {pdf_file}")

# ===== Save to CSV =====
df = pd.DataFrame(all_mcqs)
df.to_csv(output_csv, index=False)
print(f"\nAll MCQs saved to {output_csv}")










