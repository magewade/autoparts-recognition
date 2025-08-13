
# Autoparts Recognition

An automated system for collecting, recognizing, and analyzing automotive parts using modern machine learning models and LLMs (Gemini).

---

## Features

* **Data Collection** : Automatic parsing of product listings from marketplaces (Yahoo, Goofish) using Selenium and requests.
* **Image Recognition** : Classification and selection of the most relevant part image using a convolutional neural network (MobileNetV3Small).
* **Part Number Extraction** : Using an LLM (Gemini) to extract and validate OEM numbers and brands from images.
* **Flexible Prompt Configuration** : Support for various brands and part number formats through `prompts.json`.
* **Results Export** : Saving predictions and metadata to Excel for further analysis.
