## Autoparts Recognition

An automated system for collecting, recognizing, and analyzing automotive parts using modern machine learning models and LLMs (Gemini).

---

## Features

* **Data Collection** : Automatic parsing of product listings from marketplaces (Yahoo, Goofish) using Selenium and requests.
* **Image Recognition** : Classification and selection of the most relevant part image using a convolutional neural network (MobileNetV3Small).
* **Part Number Extraction** : Using an LLM (Gemini) to extract and validate OEM numbers and brands from images.
* **Flexible Prompt Configuration** : Support for various brands and part number formats through `prompts.json`.
* **Results Export** : Saving predictions and metadata to Excel for further analysis.

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
