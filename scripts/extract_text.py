import os
import pdfplumber

PDF_PATH = os.path.join("data", "seminar input 2.pdf")
OUT_TXT = os.path.join("data", "seminar_input_2.txt")

def extract_pdf_text(pdf_path: str) -> str:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            chunks.append(f"\n\n--- PAGE {i+1} ---\n\n{page_text}")
    return "\n".join(chunks)

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    text = extract_pdf_text(PDF_PATH)

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Extracted text saved to: {OUT_TXT}")
    print(f"✅ Characters extracted: {len(text)}")
