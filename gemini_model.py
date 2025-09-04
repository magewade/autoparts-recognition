import google.generativeai as genai
from pathlib import Path
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
You are an expert at extracting automotive part/model numbers from images. You are given the following information from a previous analysis of the product description:

[Description LLM output]
Brand: {car_brand}
Numbers: {desc_numbers}
One_or_many: {desc_one_many}

Your task:
1. Use all information above to help you analyze the image. If any field is missing or 'None', try to fill it using the image.
2. If a brand is given, use it to help identify relevant numbers or text in the image. If not, try to infer the brand/model from the image.
3. Your main goal is to extract the main serial (OEM) part number or model number for each physical object visible in the image. This number is unique for each part, usually 9-15 characters, contains both letters and digits, and is not a date, batch, or random short code. 
4. Always try to select the number that most closely matches the typical format of a model or OEM part number (for example, a mix of letters and digits, often with dashes or spaces, and not just a short code or serial/batch).
5. The OEM/serial/model number is most often located directly above or near the barcode, and usually just below the logo or brand name of the manufacturer. It is often printed in a larger or bolder font. 
6. Do NOT select numbers that are printed far from the brand or barcode, or that look like internal codes, batch numbers, or serials. Do NOT select numbers that are at the very bottom of the label or packaging unless they clearly match the OEM/model format and are near the brand/barcode. 
7. If there are several numbers, always prefer the one that is closest to the brand name/logo and barcode, and that matches the typical OEM/model number format. 
8. Ignore any numbers that are not plausible serial/OEM/model part numbers.
9. If the brand is Bosch, in addition to extracting the Bosch part number, try to infer or guess for which car model(s) this part might be intended, based on any visible information (text, numbers, context) in the image. If you can guess the car model, mention it in parentheses after the part number (e.g. 0 280 155 968 (for BMW 3 Series)).

8. If there is only one physical object (part) in the image, output its main serial/OEM part number in the second field, and set the last field to 'one'.
9. If there are clearly multiple separate physical objects (for example, several identical or different parts, or a set/kit of parts), output the main serial/OEM part number for each object (comma-separated), and set the last field to 'many'.
10. Do NOT set 'many' just because you see several numbers on one part. Only set 'many' if there are multiple distinct physical objects/parts visible.
11. If you are not sure, default to 'one'.
12. Always double-check for character confusion: '1' vs 'I', '0' vs 'O', etc.
13. Ignore numbers that are clearly dates, serials, or batch codes unless no other candidates exist.

Output strictly in this format (always in English, always 3 fields, always separated by |):
<START> [Brand/Model Guess] | [Model/Part Number(s)] | [one/many] <END>

- [Brand/Model Guess]: The car brand/model you used (from description or inferred from image), or None.
- [Model/Part Number(s)]: The main serial/OEM part number(s) found (from description or image), or None. If there are multiple physical objects, separate numbers with commas.
- [one/many]: Write 'many' only if there are multiple separate physical objects/parts visible in the image. If only one part is visible (even with several numbers), write 'one'. If unsure, write 'one'.

If you don't know a value, write None. Do not output anything else except the required 3 fields in the specified format. Always answer in English.
"""


class GeminiInference:
    def __init__(
        self,
        api_keys,
        model_name="gemini-2.5-flash",
        car_brand=None,
        desc_numbers=None,
        desc_one_many=None,
        prompt_override=None,
    ):
        logging.info(f"[GeminiInference] Using model: {model_name}")

        self.car_brand = car_brand
        self.desc_numbers = desc_numbers
        self.desc_one_many = desc_one_many

        self.api_keys = api_keys
        self.current_key_index = 0
        # Всегда используем только DEFAULT_PROMPT
        logging.info(
            f"[GeminiInference] Description info for prompt: brand='{car_brand}', numbers='{desc_numbers}', one_many='{desc_one_many}'"
        )
        prompt_filled = DEFAULT_PROMPT.format(
            car_brand=car_brand if car_brand is not None else "None",
            desc_numbers=desc_numbers if desc_numbers is not None else "None",
            desc_one_many=desc_one_many if desc_one_many is not None else "None",
        )
        if prompt_override:
            self.system_prompt = prompt_override.strip() + "\n\n" + prompt_filled
        else:
            self.system_prompt = prompt_filled

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

        # Всегда используем только DEFAULT_PROMPT

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
                image_part = {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": (
                            img_data.getvalue()
                            if isinstance(img_data, io.BytesIO)
                            else img_data.read_bytes()
                        ),
                    }
                }
                # Всегда первым идёт текстовый промпт, затем картинка
                prompt_parts = [self.system_prompt, image_part]
                if retry:
                    prompt_parts.append(
                        "It is not correct. Try again. Look for the numbers that are highly OEM number"
                    )

                time.sleep(random.uniform(1, 3))

                chat = self.model.start_chat(history=self.message_history)
                response = chat.send_message(prompt_parts)

                logging.info(f"Main model response: {response.text}")

                self.message_history.append({"role": "user", "parts": prompt_parts})
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
        # Replace all 'nan' with 'None' in the output for consistency
        number = number.replace("nan", "None")
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
            # Для retry добавляем жёсткое сообщение к system_prompt
            if attempt == 1:
                orig_prompt = self.system_prompt
                self.system_prompt = (
                    "Previous answer did not match required format (must contain exactly 2 pipe | characters and 3 fields). STRICTLY follow the output format!\n\n"
                    + orig_prompt
                )
            answer = self.get_response(img_data, retry=(attempt > 0))
            # Проверка формата: должно быть ровно 2 pipe (|) и <START>/<END>
            if answer.count("|") != 2:
                logging.info(
                    f"LLM output format invalid (pipes: {answer.count('|')}), retrying..."
                    if attempt == 0
                    else "All attempts failed or only nan found."
                )
                if attempt == max_attempts - 1:
                    break
                continue
            extracted_number = self.extract_number(answer)

            logging.info(f"Attempt {attempt + 1}: Extracted number: {extracted_number}")

            # Если результат не "None | None | unknown", считаем его валидным и возвращаем сразу
            if extracted_number.strip().lower() != "none | none | none":
                self.reset_incorrect_predictions()
                return extracted_number

            # Если ничего не найдено, пробуем дальше
            if attempt < max_attempts - 1:
                logging.info(
                    f"No valid number found in attempt {attempt + 1}, retrying..."
                )

        logging.warning("All attempts failed or only None found.")
        self.reset_incorrect_predictions()
        # Возвращаем всегда 3 поля, если ничего не найдено
        return "None | None | None"
