import os
from datetime import datetime
from google import genai

# =======================
# CONFIG
# =======================

API_KEY = "API_KEY"  # replace with actual api key
MODEL_NAME = "gemini-3-flash-preview"

TEXT_PATH = os.path.join("data", "seminar_input_2.txt")
OUT_DIR = "outputs"

# Safety limit: set to 20 for free-tier
CALL_BUDGET = 20

# Run plan:
# - Medium: 4 calls -> 12 MCQs (Human vs AI)
# - Easy:   1 call  -> 3 MCQs  (Quality ratings)
# - Hard:   1 call  -> 3 MCQs  (Quality ratings)
# Total = 6 calls
RUN_PLAN = [
    ("Medium", 4),
    ("Easy", 1),
    ("Hard", 1),
]

GEN_TEMPLATE = """
You are an academic quiz generator.
Create EXACTLY THREE multiple-choice questions based ONLY on the content below.
Difficulty: {difficulty}

FORMAT (repeat 3 times exactly):
Question: <question>
A. <option>
B. <option>
B. <option>
C. <option>
D. <option>
Correct Answer: <A/B/C/D>
Explanation: <1-2 sentences>

RULES:
- No numbering
- No extra text before or after the 3 blocks
- Each question must test a different concept from the content
- Options must be plausible and clearly distinct
- The correct answer must be unambiguous from the content
- The explanation must justify the chosen letter

CONTENT:
{content}
""".strip()

# =======================
# FUNCTIONS
# =======================

def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Text file not found at: {path}\n"
            f"Run scripts/extract_text.py first to create data/seminar_input_2.txt."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_client() -> genai.Client:
    if not API_KEY or API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError("API_KEY not set. Paste your Gemini key into API_KEY in this file.")
    return genai.Client(api_key=API_KEY)


def generate_3_mcqs(client: genai.Client, content: str, difficulty: str) -> str:
    prompt = GEN_TEMPLATE.format(difficulty=difficulty, content=content)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text or ""


def save_output(text: str, filename: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


# =======================
# MAIN
# =======================

if __name__ == "__main__":
    planned_calls = sum(n for _, n in RUN_PLAN)

    print(f"📊 Planned API calls: {planned_calls}")
    print(f"📊 Budget (your setting): {CALL_BUDGET}")

    if planned_calls > CALL_BUDGET:
        raise RuntimeError(
            f"ABORTING: Planned calls ({planned_calls}) exceed CALL_BUDGET ({CALL_BUDGET}). "
            f"Reduce RUN_PLAN or increase CALL_BUDGET."
        )

    content = load_text(TEXT_PATH)
    client = get_client()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    call_counter = 0
    for difficulty, num_calls in RUN_PLAN:
        for run_idx in range(1, num_calls + 1):
            call_counter += 1
            print(
                f"\n=== API CALL {call_counter}/{planned_calls} | "
                f"Difficulty={difficulty} | Run {run_idx}/{num_calls} ==="
            )

            output = generate_3_mcqs(client, content, difficulty=difficulty)

            if num_calls == 1:
                fname = f"gemini_{difficulty.lower()}_{timestamp}.txt"
            else:
                fname = f"gemini_{difficulty.lower()}_run{run_idx}_{timestamp}.txt"

            out_path = save_output(output, fname)
            print(f"✅ Saved: {out_path}")

    print("\n✅ Done. Check the outputs/ folder.")
    print("⚠️ Reminder: running this script again will consume the same number of API calls.")
