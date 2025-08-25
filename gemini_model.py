# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep
import random
import logging
import time
import json
import os
from PIL import Image
import requests
import re
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_PROMPT = """
Identify the most likely model or part number from the photo of any automotive part, using the following universal algorithm:

1. **Scan the Image Thoroughly:**
    - Examine all visible text, numbers, barcodes, and labels in the image.
    - Focus on stickers, embossed/engraved areas, and prominent alphanumeric sequences.
    - Pay special attention to numbers near brand names, barcodes, or in bold/large font.

2. **Understand Brand and Format Diversity:**
    - The part/model number may follow different formats depending on the brand (e.g., Toyota, Audi, Bosch, Denso, etc.).
    - Typical identifiers are 8–15 characters, often a mix of letters and digits, sometimes with hyphens or spaces.
    - If the brand is visible, prefer numbers that match known formats for that brand (if you know them).
    - If no brand is visible, select the most prominent, structured number.

3. **Selection Rules:**
    - If multiple candidates exist, choose the one that is:
      - Closest to the brand name or logo (if present)
      - In bold or larger font
      - Most structured (e.g., contains both letters and digits, or matches a known pattern)
    - If all candidates are equally likely, pick the first one (top-to-bottom, left-to-right).
    - If no plausible number is found, return NONE.

4. **Common Pitfalls:**
    - Be careful with character confusion: '1' vs 'I', '0' vs 'O', '8' vs 'B', '5' vs 'S', '2' vs 'Z'.
    - Ignore numbers that are clearly dates, serials, or batch codes unless no other candidates exist.
    - If the only visible identifier is a long string (e.g., 16+ characters) or a barcode, extract the most plausible substring.

5. **Contextual Verification:**
    - Consider the part's function and any visible brand/model context.
    - If the part is from a well-known brand, try to match the number to typical formats for that brand.
    - If the part is generic or the brand is unknown, select the most prominent number.

**Response Format:**
- If a model/part number is identified: `<START> [Model/Part Number] <END>`
- If no valid number is identified: `<START> NONE <END>`

If there are multiple numbers, return only the one that is most likely to be the main identifier for this part.
"""


class GeminiInference:
    def __init__(
        self,
        api_keys,
        model_name="gemini-1.5-flash",
        car_brand=None,
        prompt_override=None,
    ):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.car_brand = car_brand.lower() if car_brand else None
        self.prompts = self.load_prompts()

        # Используем override, если он задан, иначе берем из prompts.json или дефолт
        base_prompt = self.prompts.get(self.car_brand, {}).get(
            "main_prompt", DEFAULT_PROMPT
        )
        if prompt_override:
            self.system_prompt = prompt_override.strip() + "\n\n" + base_prompt
        else:
            self.system_prompt = base_prompt

        self.configure_api()
        generation_config = {
            "temperature": 1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 8192,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]

        self.system_prompt = self.prompts.get(self.car_brand, {}).get(
            "main_prompt", DEFAULT_PROMPT
        )

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=self.system_prompt,
        )

        self.validator_model = self.create_validator_model(model_name)
        self.incorrect_predictions = []
        self.message_history = []

    def load_prompts(self):
        try:
            with open("prompts.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("prompts.json not found. Using default prompts.")
            return {}

    def configure_api(self):
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_api()
        logging.info(f"Switched to API key index: {self.current_key_index}")

    def create_validator_model(self, model_name):
        genai.configure(api_key=self.api_keys[self.current_key_index])

        generation_config = {
            "temperature": 1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 8192,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]
        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    def get_response(self, img_data, retry=False):
        max_retries = 10
        base_delay = 5

        for attempt in range(max_retries):
            try:
                image_parts = [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": (
                                img_data.getvalue()
                                if isinstance(img_data, io.BytesIO)
                                else img_data.read_bytes()
                            ),
                        }
                    },
                ]

                prompt_parts = (
                    []
                    if not retry
                    else [
                        "It is not correct. Try again. Look for the numbers that are highly VAG number"
                    ]
                )

                full_prompt = image_parts + prompt_parts

                sleep(random.uniform(1, 3))

                chat = self.model.start_chat(history=self.message_history)
                response = chat.send_message(full_prompt)

                logging.info(f"Main model response: {response.text}")

                self.message_history.append({"role": "user", "parts": full_prompt})
                self.message_history.append({"role": "model", "parts": [response.text]})

                return response.text

            except Exception as e:
                if "quota" in str(e).lower():
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    if delay > 300:
                        self.switch_api_key()
                        delay = base_delay
                    logging.warning(
                        f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logging.error(f"Error in get_response: {str(e)}")
                    raise

        logging.error("Max retries reached. Unable to get a response.")
        raise Exception("Max retries reached. Unable to get a response.")

    def format_part_number(self, number):
        if self.car_brand == "audi" and re.match(
            r"^[A-Z0-9]{3}[0-9]{3}[0-9]{3,5}[A-Z]?$",
            number.replace(" ", "").replace("-", ""),
        ):
            number = number.replace("-", "").replace(" ", "")

            formatted_number = f"{number[:3]} {number[3:6]} {number[6:9]}"

            if len(number) > 9:
                formatted_number += f" {number[9:]}"

            return formatted_number.strip()
        else:
            return number

    def extract_number(self, response):
        number = response.split("<START>")[-1].split("<END>")[0].strip()
        if number.upper() != "NONE":
            return self.format_part_number(number)
        return number

    def validate_number(self, extracted_number, img_data):
        genai.configure(api_key=self.api_keys[self.current_key_index])

        formatted_number = self.format_part_number(extracted_number)

        image_parts = [
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": (
                        img_data.getvalue()
                        if isinstance(img_data, io.BytesIO)
                        else img_data.read_bytes()
                    ),
                }
            },
        ]

        validation_prompt = self.prompts.get(self.car_brand, {}).get(
            "validation_prompt", ""
        )
        incorrect_predictions_str = ", ".join(self.incorrect_predictions)
        prompt = validation_prompt.format(
            extracted_number=extracted_number,
            incorrect_predictions=incorrect_predictions_str,
        )

        prompt_parts = [
            image_parts[0],
            prompt,
        ]

        response = self.validator_model.generate_content(prompt_parts)

        logging.info(f"Validator model response: {response.text}")
        return response.text

    def reset_incorrect_predictions(self):
        self.incorrect_predictions = []
        self.message_history = []

    def __call__(self, image_path):
        self.configure_api()

        if image_path.startswith("http"):
            response = requests.get(image_path, stream=True)
            img_data = io.BytesIO(response.content)
        else:
            img = Path(image_path)
            if not img.exists():
                raise FileNotFoundError(f"Could not find image: {img}")
            img_data = img

        self.message_history = []

        max_attempts = 2
        for attempt in range(max_attempts):
            answer = self.get_response(img_data, retry=(attempt > 0))
            extracted_number = self.extract_number(answer)

            logging.info(f"Attempt {attempt + 1}: Extracted number: {extracted_number}")

            # Если LLM вернул специальный маркер о множестве номеров
            if extracted_number.strip().endswith("| True"):
                logging.warning("Multiple numbers detected, returning special marker.")
                self.reset_incorrect_predictions()
                return "nan | nan | nan | True"

            # Если хоть какой-то номер найден в presumptive_model_number
            try:
                presumptive = extracted_number.split("|")[2].strip()
            except Exception:
                presumptive = ""
            if presumptive and presumptive.lower() != "nan":
                logging.info(
                    f"Presumptive model number found: {presumptive}, accepting as valid result."
                )
                self.reset_incorrect_predictions()
                return extracted_number

            # Если ничего не найдено, пробуем дальше
            if attempt < max_attempts - 1:
                logging.info(
                    f"No valid number found in attempt {attempt + 1}, retrying..."
                )

        logging.warning("All attempts failed or only nan found.")
        self.reset_incorrect_predictions()
        # Возвращаем всегда 4 поля, если ничего не найдено
        return "nan | nan | nan | False"
