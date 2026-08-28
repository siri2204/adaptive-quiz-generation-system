# Adaptive AI Quiz Generator

## MSc Seminar Project — Supplementary Materials

This repository contains the source code, evaluation artifacts, and supplementary materials for the MSc seminar project:

**“An Adaptive AI-Based Quiz Generation System for Educational Assessment”**

The project implements an AI-driven system for generating multiple-choice questions (MCQs) from lecture material using both cloud-based and locally deployed language models. The system incorporates adaptive difficulty control and was evaluated through automated experiments and human-centered studies.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Gemini API

The primary application uses the Gemini API. Set your API key as an environment variable before running the application.

For example:

```bash
export GEMINI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-api-key"
```

The API key is **not stored in the repository**.

### 3. Run the primary Gemini-based application

```bash
cd scripts
streamlit run app.py
```

### 4. Run the local-model exploratory application

```bash
cd scripts
streamlit run app_models.py
```

---

## 1. Main Application — Gemini API

**File:** `scripts/app.py`

This is the primary application evaluated in the project report. It uses the Gemini API to generate MCQs from uploaded lecture material.

### Key Features

* MCQ generation from uploaded lecture PDFs
* Easy / Medium / Hard difficulty levels
* Quiz-set-level adaptive difficulty
* Three questions per quiz set
* Optional topic-based quiz selection
* Answer explanations
* Structured question generation and validation

### Used For

The Gemini-based system was used in:

* Human-vs.-AI question identification study
* MCQ quality evaluation
* Student feedback user study

---

## 2. Local Model Application — Exploratory

**File:** `scripts/app_models.py`

This application implements an exploratory local-model version of the quiz generator. It was developed to investigate the feasibility and deployment trade-offs of running language models locally.

### Key Features

* Model selection through a dropdown interface
* Support for open-source GGUF models
* Local inference using llama.cpp-compatible backends
* Per-question adaptive difficulty
* Correct answer → difficulty increases
* Incorrect answer → difficulty decreases

### Purpose

The local-model experiments focused on:

* Inference latency
* Output quality
* Structural validity of generated MCQs
* Formatting reliability
* Feasibility of local deployment

The local-model application was **not used for the human-centered evaluation studies**. It was used exclusively for comparative feasibility and model trade-off analysis.

---

## 3. Local Models Evaluated

The following open-source GGUF models were evaluated using llama.cpp-compatible local inference.

Model binaries are **not included in this repository** due to their size. Links to the corresponding model repositories are provided below.

### Meta-Llama-3-8B-Instruct

* Quantization: `Q4_0`
* Source: [Hugging Face — QuantFactory/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)

### Nous-Hermes-2-Mistral-7B-DPO

* Quantization: `Q5_K_M`
* Source: [Hugging Face — NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF)

### Orca Mini 3B

* Quantization: `Q4_0`
* Expected file: `q4_0-orca-mini-3b.gguf`

These models were evaluated with respect to:

* Inference latency
* MCQ structural validity
* Formatting reliability
* Subjective output quality

The corresponding results and screenshots are included in the supplementary materials.

---

## 4. Supporting Scripts

| File                        | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| `extract_text.py`           | PDF text extraction                      |
| `generate_mcqs_from_pdf.py` | End-to-end PDF → MCQ generation          |
| `eval_mcq.py`               | Controlled MCQ generation for evaluation |
| `test_local_model.py`       | Local model testing and debugging        |
| `test_setup.py`             | Environment and dependency checks        |
| `gemini_api.py`             | Gemini API interaction logic             |

---

## 5. Data

The `data/` directory contains the lecture material used during evaluation.

```text
data/
├── seminar_input_2.pdf
└── seminar_input_2.txt
```

* `seminar_input_2.pdf` — lecture material used for evaluation
* `seminar_input_2.txt` — extracted text version

---

## 6. Generated Outputs

The `outputs/` directory contains raw MCQ generation outputs from the Gemini API across different difficulty levels and experimental runs.

These outputs were used during the evaluation process and are included to provide transparency into the generated data.

---

## 7. Evaluation Materials

The `evaluation_materials/` directory contains the human evaluation artifacts referenced in the project report.

```text
evaluation_materials/
├── raw/
├── collected_info/
└── final_summary/
```

* `raw/` — original evaluation documents
* `collected_info/` — collected evaluation responses
* `final_summary/` — aggregated results, accuracy tables, and qualitative summaries

Where applicable, identifying information has been removed from shared evaluation materials.

---

## 8. Reproducibility

To reproduce the main Gemini-based experiments:

1. Install the required Python dependencies.
2. Configure a valid Gemini API key through the `GEMINI_API_KEY` environment variable.
3. Run `scripts/app.py`.
4. Use the lecture material provided in `data/`.
5. Generated outputs can be compared with the experimental outputs provided in `outputs/`.

For local-model experiments, download the required GGUF model separately and place it in the location expected by the local-model scripts.

Local inference may require substantial CPU memory and computational resources, depending on the selected model and quantization.

---

## 9. Repository Structure

```text
.
├── scripts/
│   ├── app.py
│   ├── app_models.py
│   ├── extract_text.py
│   ├── generate_mcqs_from_pdf.py
│   ├── eval_mcq.py
│   ├── test_local_model.py
│   ├── test_setup.py
│   └── gemini_api.py
│
├── data/
│   ├── seminar_input_2.pdf
│   └── seminar_input_2.txt
│
├── outputs/
│   └── ...
│
├── evaluation_materials/
│   ├── raw/
│   ├── collected_info/
│   └── final_summary/
│
├── requirements.txt
└── README.md
```

---

## 10. Security and API Keys

API credentials are **not committed to this repository**.

The Gemini API key must be provided through the `GEMINI_API_KEY` environment variable when running the Gemini-based application.

Do not commit API keys, `.env` files, credentials, or other sensitive information to the repository.

---


