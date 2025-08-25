## Autoparts Recognition

An automated system for collecting, recognizing, and analyzing automotive parts using modern machine learning models and LLMs (Gemini).

---

## Features

* **Data Collection**: Automatic parsing of product listings from marketplaces (Yahoo, Goofish) using Selenium and requests.
* **Image Recognition**: Classification and selection of the most relevant part image using a convolutional neural network (MobileNetV3Small).
* **Part Number Extraction**: Using an LLM (Gemini) to extract and validate OEM numbers and brands from images.
* **Flexible Prompt Configuration**: Support for various brands and part number formats through `prompts.json`.
* **Results Export**: Saving predictions and metadata to Excel/CSV for further analysis.
* **Robust Pipeline & Resumption**: The pipeline automatically resumes from the last completed step if interrupted, using intermediate CSVs. Partial results are saved every 10 rows for maximum reliability.
* **Error Handling & Retry Logic**: All key steps (including LLM inference) have built-in error handling and retry mechanisms. If the LLM fails twice for a product, the result is marked as empty ("nan | nan | nan | False") for that row, so you can easily identify and reprocess failures later.

---

## Pipeline Robustness & Resumption

The system is designed to be robust against interruptions and failures:

- **Stepwise Saving**: At each major stage (link collection, parsing, inference), results are saved to CSV. If the process is interrupted, it will resume from the last completed row/file.
- **Partial Results**: Every 10 rows, partial results are saved, minimizing data loss in case of crashes or restarts.
- **LLM Retry Logic**: For each product, the LLM is called up to 2 times. If both attempts fail (due to errors or invalid responses), the result is left empty for that row, and the pipeline continues. This ensures that a single failure does not halt the entire process.
- **Logging**: All steps, errors, and retries are logged for transparency and debugging.

---

## Quick Start (Colab)

You can run the full pipeline in Google Colab using the provided notebook:

👉 [Colab Notebook](https://colab.research.google.com/drive/1yoZfnOA_LHF3bwF3YSpRtM1frP9BEyK_?usp=sharing)

### Entry Point (CLI)

The main entry point for batch processing is:

```
python main_goofish.py --model gemini --api-keys <YOUR_GEMINI_API_KEY> --car-brand <brand> [--max-steps 3 --max-links 90 --save-file-name result]
```

**Required arguments:**

- `--model gemini` — use Gemini LLM for part number extraction
- `--api-keys` — your Gemini API key(s)
- `--car-brand` — brand for prompt (e.g. toyota, audi, honda, etc)

**Optional:**

- `--max-steps` — how many pages to parse (default: 3)
- `--max-links` — how many product links to collect (default: 90)
- `--save-file-name` — output file name (default: recognized_data)
