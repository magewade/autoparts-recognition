## Autoparts Recognition

An automated system for collecting, recognizing, and analyzing automotive parts using modern machine learning models and LLMs (Gemini).

---


## Features

* **Data Collection**: Automatic parsing of product listings from marketplaces (Yahoo, Goofish) using Selenium and requests.
* **Image Recognition**: Classification and selection of the most relevant part image using a convolutional neural network (MobileNetV3Small).
* **Part Number Extraction**: Using an LLM (Gemini) to extract and validate OEM numbers and brands from images, with strict output format enforcement.
* **Flexible Prompt Configuration**: Support for various brands and part number formats through `prompts.json`, including a universal fallback prompt for unknown brands.
* **Automatic Brand Inference**: The pipeline can infer the car brand/model from product descriptions using LLM, and automatically uses this for brand-specific prompt selection.
* **Strict LLM Output Format**: All LLM responses are required to follow a strict, parseable format: `<START> [Brand/Model Guess] | [Model/Part Number] | [Presumptive Model Number] | [Multiple? True/False] <END>`. This ensures robust downstream processing and validation.
* **Results Export**: Saving predictions and metadata to Excel/CSV for further analysis.
* **Robust Pipeline & Resumption**: The pipeline automatically resumes from the last completed step if interrupted, using intermediate CSVs. Partial results are saved every 10 rows for maximum reliability.
* **Error Handling & Retry Logic**: All key steps (including LLM inference) have built-in error handling and retry mechanisms. If the LLM fails twice for a product, the result is marked as empty ("nan | nan | nan | False") for that row, so you can easily identify and reprocess failures later.

---

## LLM Output Format & Prompting

**LLM output is always strictly enforced to be in the following format:**

```
<START> [Brand/Model Guess] | [Model/Part Number] | [Presumptive Model Number] | [Multiple? True/False] <END>
```

- `[Brand/Model Guess]`: The car brand/model (provided or inferred), or `nan` if unknown.
- `[Model/Part Number]`: The most likely validated part/model number, or `nan`.
- `[Presumptive Model Number]`: The most likely car model number, or `nan`.
- `[Multiple? True/False]`: `True` if multiple unique parts are visible, else `False`.

If a value is unknown, the model outputs `nan`. No extra text, tags, or formatting is allowed.

**Prompt selection logic:**

- If a brand is detected (from description or user input), the corresponding brand-specific prompt from `prompts.json` is used.
- If the brand is unknown, a universal fallback prompt (`all`) is used, which contains rules for all supported brands and enforces the strict output format.
- The pipeline automatically infers the car brand/model from product descriptions using LLM, and uses this for prompt selection.
- All prompt logic and fallback handling is robust to missing or unknown brands.

**Validation:**

- Each extracted part number is validated using brand-specific rules from `prompts.json`.
- If validation fails, the result is set to `nan | nan | nan | False`.

---

---


## Pipeline Robustness & Resumption

The system is designed to be robust against interruptions and failures:

- **Stepwise Saving**: At each major stage (link collection, parsing, inference), results are saved to CSV. If the process is interrupted, it will resume from the last completed row/file.
- **Partial Results**: Every 10 rows, partial results are saved, minimizing data loss in case of crashes or restarts.
- **LLM Retry Logic**: For each product, the LLM is called up to 2 times. If both attempts fail (due to errors or invalid responses), the result is left empty for that row, and the pipeline continues. This ensures that a single failure does not halt the entire process.
- **Strict Output Enforcement**: If the LLM returns a response not matching the required format, the pipeline retries and/or marks the result as invalid, ensuring only parseable outputs are saved.
- **Logging**: All steps, errors, and retries are logged for transparency and debugging.

---

## Quick Start (Colab)

You can run the full pipeline in Google Colab using the provided notebook:

👉 [Colab Notebook](https://colab.research.google.com/drive/1yoZfnOA_LHF3bwF3YSpRtM1frP9BEyK_?usp=sharing)

### Entry Point (CLI)

The main entry point for batch processing is:

```
python main_goofish.py --api-keys <YOUR_GEMINI_API_KEY> [--car-brand <brand>] [--max-steps 3 --max-links 90 --save-file-name result]
```

**Required arguments:**

- `--api-keys` — your Gemini API key(s)

**Optional:**

- `--car-brand` — brand for prompt (e.g. toyota, audi, honda, etc). If not provided, the pipeline will infer the brand/model from product descriptions automatically.
- `--max-steps` — how many pages to parse (default: 3)
- `--max-links` — how many product links to collect (default: 90)
- `--save-file-name` — output file name (default: recognized_data)

---
