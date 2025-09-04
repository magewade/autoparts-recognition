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
3. If part/model numbers are given, check if they appear in the image. If not, try to find a new plausible OEM-like number (9-15 characters, mix of letters and digits, not a date or batch code). If you find a better or additional number, update the field accordingly.
4. If the description said 'many', check if the image also shows multiple different part/model numbers or items. If so, set the last field to True. If only one, set to False.
5. Always double-check for character confusion: '1' vs 'I', '0' vs 'O', etc.
6. Ignore numbers that are clearly dates, serials, or batch codes unless no other candidates exist.

Output strictly in this format (always in English, always 3 fields, always separated by |):
<START> [Brand/Model Guess] | [Model/Part Number(s)] | [one/many] <END>

- [Brand/Model Guess]: The car brand/model you used (from description or inferred from image), or None.
- [Model/Part Number(s)]: The most likely part/model number(s) found (from description or image), or None.
- [one/many]: Write 'many' if you see more than one plausible part/model number or item in the image, or if the numbers field contains more than one number (comma, space, or 'and'). Also set 'many' if there are indirect signs of multiple items (e.g. 'set', 'kit', 'several', '2 pcs', 'for different models', 'multiple', 'набор', 'комплект', 'несколько', '2 шт', etc.). If you are not sure, write 'one'.

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

        self.api_keys = api_keys
        self.current_key_index = 0
        self.prompts = self.load_prompts()
        # Подставляем значения в промпт
        base_prompt = self.prompts.get("all", {}).get("main_prompt", DEFAULT_PROMPT)
        logging.info(
            f"[GeminiInference] Description info for prompt: brand='{car_brand}', numbers='{desc_numbers}', one_many='{desc_one_many}'"
        )
        prompt_filled = base_prompt.format(
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

        # self.system_prompt = self.prompts.get(self.car_brand, {}).get(
        #     "main_prompt"
        # ) or self.prompts.get("all", {}).get("main_prompt", DEFAULT_PROMPT)
        self.system_prompt = self.prompts.get("all", {}).get(
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
                        "It is not correct. Try again. Look for the numbers that are highly OEM number"
                    ]
                )

                full_prompt = image_parts + prompt_parts

                time.sleep(random.uniform(1, 3))

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

            # Если результат не "None | None | False", считаем его валидным и возвращаем сразу
            if extracted_number.strip().lower() != "none | none | false":
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
        return "None | None | False"
